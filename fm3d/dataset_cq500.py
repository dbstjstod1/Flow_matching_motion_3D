"""CQ500 head CT -> simulated cone-beam CBCT. The dataset the head-motion literature standardized on.

CQ500 (Qure.ai / CARING, New Delhi) is 491 non-contrast head CT scans, DICOM, CC BY-NC-SA 4.0.
It is DIAGNOSTIC MDCT, not CBCT: every paper in this area takes its volumes as the CLEAN ground
truth and FORWARD-PROJECTS them in cone-beam geometry to synthesize the motion-corrupted CBCT.
That is what this module does, so we sit exactly where Thies et al. and JRM-ADM sit.

It replaces `dataset_slab.py` (the AAPM slice archive), and it retires three of that file's debts:
`dz` stops being a guess (DICOM carries the spacing), the interloper-slice and patient-boundary
heuristics are gone (CQ500 is organized by patient and series), and we get whole heads instead of
64-slice slabs.

SELECTION, after Thies et al. (IEEE TMI 2025, arXiv:2401.09283), Sec. III:

    "We first filter the data for scans which have been reconstructed with a small slice
     thickness. From the remaining scans we further exclude those which have considerably fewer
     or more slices than the average sample. This results in 320 scans which we then split
     sequentially on patient level into training set (150), validation set (50), and test set
     (120). After filtering, all scans have a reconstructed slice thickness of 0.625 mm and an
     isotropic in-slice spacing between 0.38 mm and 0.58 mm (mean of 0.472 mm)."

Two things that sentence does NOT pin down, so they are ours and are flagged as such:

  * **"considerably fewer or more"** has no published threshold. We keep series whose slice count
    lies within `count_tol` of the MEDIAN (median, not mean -- the outliers we are removing are
    exactly what drags a mean around). Their rule landed on 320 of 491; if ours lands elsewhere,
    that is the knob.
  * **A patient can have more than one thin series** (CQ500 ships several reconstructions per
    study). We take the one with the most slices, so the choice is deterministic and needs no seed.

THE SPLIT IS SEQUENTIAL AND PATIENT-LEVEL, NOT RANDOM. Patients are ordered by their CQ500 index
(CQ500-CT-0, -1, -2, ...) and cut 150 / 50 / rest. There is no RNG in it, so it is reproducible
without a seed and cannot leak a patient across the cut. `--split_counts` overrides.

NORMALIZATION. Physics in mu [1/mm], network in [-1,1], same contract as `AAPMSlabGenerator`:

    mu     = (HU/1000 + 1) * mu_water
    to_net = 2 (mu - mu_lo) / (mu_hi - mu_lo) - 1        over the HU window `hu_norm`

with one DELIBERATE DIFFERENCE: HU is CLIPPED to the window before conversion. `AAPMSlabGenerator`
does not clip. JRM-ADM does ("intensity range clipped between -1000 and 2000 HU") and CQ500,
unlike the AAPM archive, contains dental amalgam and surgical metal whose HU runs to +3000 and
beyond -- unclipped, one such voxel sets the scale for the whole normalization.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from .filters import calibrate_scale
from .geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                          view_angular_weights)
from .projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched
from .rigid_motion import params_to_Pmot, random_motion

MU_WATER = 0.02
_PATIENT_RE = re.compile(r"CQ500[-_ ]?CT[-_ ]?(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------------------
def patient_id(path: str) -> int | None:
    """The CQ500 patient index out of any path component ('CQ500CT23 CQ500CT23/...' -> 23)."""
    m = _PATIENT_RE.search(path)
    return int(m.group(1)) if m else None


def index_cq500(root: str, *, cache: str | None = None, verbose: bool = True) -> list[dict]:
    """Every DICOM series under `root`, as {patient, path, n_slices, thickness_mm, spacing_mm}.

    Walks once and caches to JSON -- a full CQ500 tree is ~200k files and re-reading the headers
    on every run is minutes of I/O for information that never changes.
    """
    import SimpleITK as sitk

    cache = cache or os.path.join(root, "_cq500_index.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    recs = []
    reader = sitk.ImageSeriesReader()
    for dirpath, _dirnames, files in os.walk(root):
        if not any(f.lower().endswith(".dcm") for f in files):
            continue
        pid = patient_id(dirpath)
        if pid is None:
            continue
        for sid in reader.GetGDCMSeriesIDs(dirpath):
            names = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if len(names) < 2:
                continue
            f0 = sitk.ReadImage(names[0])
            sp = f0.GetSpacing()                                  # (x, y, z) mm
            th = _tag(f0, "0018|0050", sp[2])                     # SliceThickness
            recs.append({"patient": pid, "path": dirpath, "series": sid,
                         "n_slices": len(names), "thickness_mm": float(th),
                         "spacing_mm": [float(sp[0]), float(sp[1])],
                         "desc": _tag(f0, "0008|103e", "")})
    recs.sort(key=lambda r: (r["patient"], r["series"]))
    with open(cache, "w") as f:
        json.dump(recs, f, indent=1)
    if verbose:
        print(f"[cq500] indexed {len(recs)} series from "
              f"{len({r['patient'] for r in recs})} patients -> {cache}")
    return recs


def _tag(img, tag: str, default):
    try:
        v = img.GetMetaData(tag).strip()
        return float(v) if isinstance(default, float) else v
    except Exception:
        return default


# ---------------------------------------------------------------------------------------
# selection + split (Thies et al., TMI 2025, Sec. III)
# ---------------------------------------------------------------------------------------
def select_series(recs: list[dict], *, thin_mm: float = 0.7, count_tol: float = 0.5,
                  min_slices: int = 64) -> list[dict]:
    """Thin-slice filter, then a slice-count outlier cut, then one series per patient.

    thin_mm   keep series with SliceThickness <= this. 0.7 admits the 0.625 mm reconstructions
              Thies reports and rejects the 5 mm ones CQ500 also ships.
    count_tol keep |n_slices / median - 1| <= count_tol. OUR interpretation of their
              "considerably fewer or more slices than the average sample" -- they publish no
              threshold. Median, not mean: the outliers being cut are what would move a mean.
    """
    thin = [r for r in recs if r["thickness_mm"] <= thin_mm and r["n_slices"] >= min_slices]
    if not thin:
        raise RuntimeError(f"no series with thickness <= {thin_mm} mm and >= {min_slices} slices")

    med = float(np.median([r["n_slices"] for r in thin]))
    kept = [r for r in thin if abs(r["n_slices"] / med - 1.0) <= count_tol]

    by_patient: dict[int, dict] = {}
    for r in kept:                                    # most slices wins -> deterministic, no seed
        p = r["patient"]
        if p not in by_patient or r["n_slices"] > by_patient[p]["n_slices"]:
            by_patient[p] = r
    return [by_patient[p] for p in sorted(by_patient)]


def split_patients(sel: list[dict], counts=(150, 50, 120)) -> dict[str, list[dict]]:
    """SEQUENTIAL, patient-level, no RNG: patients ordered by CQ500 index, cut n_tr / n_va / rest.

    Test takes the REMAINDER rather than `counts[2]`, so a filtered set that is not exactly 320
    degrades gracefully instead of silently dropping patients off the end. With Thies' 320 the
    remainder IS 120 and this reproduces their split exactly.
    """
    n_tr, n_va = int(counts[0]), int(counts[1])
    if len(sel) < n_tr + n_va + 1:
        raise RuntimeError(f"only {len(sel)} patients survive selection; "
                           f"the {counts} split needs > {n_tr + n_va}")
    return {"train": sel[:n_tr], "val": sel[n_tr:n_tr + n_va], "test": sel[n_tr + n_va:]}


# ---------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------
class CQ500Generator:
    """Loads CQ500 volumes on an isotropic grid and projects them through the standard geometry.

    Same operator contract as `AAPMSlabGenerator` -- `volume / to_net / from_net / project / fdk /
    sample_motion / shape / fbp_scale` -- so `train_fm3d.py` and `run_posterior3d.py` take it as a
    drop-in.
    """

    def __init__(self, root: str, cfg: ConeBeam3DConfig | None = None, *, device="cuda",
                 split: str = "train", shape=(256, 256, 256), voxel_mm: float = 1.0,
                 hu_norm=(-1000.0, 2000.0), clip: bool = True, mu_water: float = MU_WATER,
                 thin_mm: float = 0.7, count_tol: float = 0.5,
                 split_counts=(150, 50, 120), n_samples_fwd: int = 512,
                 angle_weight: bool = True,
                 cache_dir: str | None = None, verbose: bool = True):
        self.device = torch.device(device)
        self.root = root
        # Per-view angular weights read out of Pmat instead of FDK's uniform angle_span/V.
        # Under rigid motion the views stop being equiangular and the uniform weight is the wrong
        # Riemann sum (-2.01 dB at 5 deg about the gantry axis). Exact no-op on the nominal orbit.
        self.angle_weight = bool(angle_weight)
        self.shape_dhw = tuple(int(s) for s in shape)
        self.dx = self.dy = self.dz = float(voxel_mm)
        self.mu_water = mu_water
        self.clip = clip
        self.n_samples_fwd = n_samples_fwd
        self.cache_dir = cache_dir or os.path.join(root, "_vol_cache")

        lo, hi = hu_norm
        self.hu_lo, self.hu_hi = float(lo), float(hi)
        self.mu_lo = (lo / 1000.0 + 1.0) * mu_water
        self.mu_hi = (hi / 1000.0 + 1.0) * mu_water

        recs = index_cq500(root, verbose=verbose)
        sel = select_series(recs, thin_mm=thin_mm, count_tol=count_tol)
        parts = split_patients(sel, counts=split_counts)
        self.splits = parts
        if split not in parts:
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.records = parts[split]
        self.split = split
        if verbose:
            print(f"[cq500] {len(sel)} patients selected | "
                  + " ".join(f"{k} {len(v)}" for k, v in parts.items())
                  + f" | using '{split}' ({len(self.records)})")

        self.cfg = cfg or ConeBeam3DConfig.thies()
        self.P_nom = build_conebeam_orbit(self.cfg, device=self.device)
        self.u_coords, self.v_coords = detector_coords_3d(self.cfg, device=self.device)
        self.fbp_scale = self._calibrate()

    # -- volumes ------------------------------------------------------------------------
    def volume(self, idx: int = 0) -> torch.Tensor:
        """(1,1,D,H,W) mu volume: patient `idx` of this split, resampled to the isotropic grid."""
        r = self.records[idx % len(self.records)]
        os.makedirs(self.cache_dir, exist_ok=True)
        npy = os.path.join(self.cache_dir, f"p{r['patient']:04d}_"
                                           f"{'x'.join(map(str, self.shape_dhw))}"
                                           f"_{self.dx:g}mm.npy")
        if os.path.exists(npy):
            hu = np.load(npy)
        else:
            hu = self._load_hu(r)
            np.save(npy, hu)
        if self.clip:
            hu = np.clip(hu, self.hu_lo, self.hu_hi)
        mu = (hu / 1000.0 + 1.0) * self.mu_water
        return torch.from_numpy(mu).float()[None, None].to(self.device)

    def _load_hu(self, r: dict) -> np.ndarray:
        """DICOM series -> HU on an isotropic `voxel_mm` grid, centre-cropped/padded to `shape`."""
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(r["path"], r["series"]))
        img = reader.Execute()                                    # HU already (rescale applied)

        # resample to an isotropic grid -- the whole reason dz stops being an assumption
        sp_in = np.array(img.GetSpacing(), dtype=np.float64)       # (x, y, z)
        sz_in = np.array(img.GetSize(), dtype=np.int64)            # (x, y, z)
        sp_out = np.array([self.dx, self.dy, self.dz], dtype=np.float64)
        sz_out = np.maximum(np.round(sz_in * sp_in / sp_out).astype(int), 1)

        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(sp_out.tolist())
        rs.SetSize([int(s) for s in sz_out])
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1000.0)                           # air, not zero (= water)
        vol = sitk.GetArrayFromImage(rs.Execute(img)).astype(np.float32)   # (z, y, x) = (D,H,W)
        return _centre_fit(vol, self.shape_dhw, pad_value=-1000.0)

    # -- net <-> mu ---------------------------------------------------------------------
    def to_net(self, mu):
        return 2.0 * (mu - self.mu_lo) / (self.mu_hi - self.mu_lo) - 1.0

    def from_net(self, x):
        return (x + 1.0) * 0.5 * (self.mu_hi - self.mu_lo) + self.mu_lo

    # -- operators ----------------------------------------------------------------------
    @property
    def shape(self):
        return self.shape_dhw

    @property
    def n_slabs(self) -> int:              # name kept so the training script's print still works
        return len(self.records)

    def project(self, vols: torch.Tensor, Pmat: torch.Tensor, **kw) -> torch.Tensor:
        return forward_project_3d_batched(
            vols, Pmat, self.u_coords, self.v_coords,
            dx=self.dx, dy=self.dy, dz=self.dz,
            n_samples=kw.pop("n_samples", self.n_samples_fwd),
            view_chunk=kw.pop("view_chunk", 4), row_chunk=kw.pop("row_chunk", 64), **kw)

    def fdk(self, sino: torch.Tensor, Pmat: torch.Tensor, *, scale=None,
            angle_weight: bool | None = None, **kw) -> torch.Tensor:
        D, H, W = self.shape_dhw
        aw = self.angle_weight if angle_weight is None else bool(angle_weight)
        # an explicitly-passed view_weight wins (and must not collide with the one we derive)
        vw = kw.pop("view_weight", None)
        if vw is None and aw:
            vw = view_angular_weights(Pmat)                        # (B,V), from the ACTUAL orbit
        return fdk_conebeam_3d_batched(
            sino, Pmat, self.u_coords, self.v_coords, self.cfg,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz,
            scale=self.fbp_scale if scale is None else scale,
            view_chunk=kw.pop("view_chunk", 8), view_weight=vw, **kw)

    def _calibrate(self) -> float:
        from .geometry_3d import measured_region_mask
        v = self.volume(0)
        with torch.no_grad():
            y = self.project(v, self.P_nom[None])
            raw = fdk_conebeam_3d_batched(
                y, self.P_nom[None], self.u_coords, self.v_coords, self.cfg,
                D=self.shape_dhw[0], H=self.shape_dhw[1], W=self.shape_dhw[2],
                dx=self.dx, dy=self.dy, dz=self.dz, scale=1.0, view_chunk=8)[0]
        m = measured_region_mask(self.shape_dhw, (self.dz, self.dy, self.dx), self.cfg,
                                 device=self.device)
        return calibrate_scale(raw, v[0, 0], m)

    def sample_volumes(self, batch: int = 1, generator=None) -> torch.Tensor:
        idx = torch.randint(len(self.records), (batch,), generator=generator)
        return torch.cat([self.volume(int(i)) for i in idx], 0)

    def sample_motion(self, batch: int = 1, *, trans_mm: float = 5.0, rot_deg: float = 5.0,
                      generator=None):
        """-> (y (B,V,nv,nu), theta (B,V,6), vols (B,1,D,H,W)). Defaults are the literature's
        evaluation amplitudes (Thies: 5 mm / 5 deg; JRM-ADM: +-5 mm / +-5 deg)."""
        vols = self.sample_volumes(batch, generator=generator)
        th = torch.stack([
            random_motion(self.cfg.n_views, trans_mm=trans_mm, rot_deg=rot_deg,
                          device=self.device, generator=generator)
            for _ in range(batch)], 0)
        P = torch.stack([params_to_Pmot(th[b], self.P_nom) for b in range(batch)], 0)
        with torch.no_grad():
            y = self.project(vols, P)
        return y, th, vols


def _centre_fit(vol: np.ndarray, shape, pad_value: float = -1000.0) -> np.ndarray:
    """Centre-crop and/or centre-pad `vol` (D,H,W) to `shape`. Padding is AIR, not zero: zero HU
    is water, so padding with it would wrap the head in a shell of soft tissue and the forward
    projection would integrate straight through it."""
    out = np.full(tuple(shape), pad_value, dtype=np.float32)
    src, dst = [], []
    for n_in, n_out in zip(vol.shape, shape):
        n = min(n_in, n_out)
        s0 = (n_in - n) // 2
        d0 = (n_out - n) // 2
        src.append(slice(s0, s0 + n))
        dst.append(slice(d0, d0 + n))
    out[tuple(dst)] = vol[tuple(src)]
    return out
