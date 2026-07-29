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
import math
import os
import re

import numpy as np
import torch

from .filters import calibrate_scale
from .geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                          view_angular_weights, view_angular_weights_dot)
from .projector_3d import (adjoint_project_3d_batched, fdk_conebeam_3d_batched,
                           fdk_conebeam_3d_tangent, forward_project_3d_batched)
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
    sample_motion / shape` -- so `train_fm3d.py` and `run_posterior3d.py` take it as a
    drop-in.
    """

    def __init__(self, root: str, cfg: ConeBeam3DConfig | None = None, *, device="cuda",
                 split: str = "train", shape=(256, 256, 256), voxel_mm: float = 1.0,
                 hu_norm=(-1000.0, 2000.0), clip: bool = True, mu_water: float = MU_WATER,
                 thin_mm: float = 0.7, count_tol: float = 0.5,
                 split_counts=(150, 50, 120),
                 angle_weight: bool = True, sim_native: bool = True,
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
        # (There used to be an `n_samples_fwd` ray-march quadrature knob here. The operator has
        # had no ray samples since the SF switch and has none now that it is LEAP's separable
        # footprint: the footprint is integrated analytically. Removed 2026-07-29.)
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
        # NO CALIBRATION (user, 2026-07-28; the 4DCT sibling deleted the same thing 2026-07-21).
        # The FDK is SELF-NORMALIZED by the pure geometry constant SOD*SDD/2
        # (`projector_3d._fdk_physical_norm`, phantom-verified to 0.2%), exactly like RTK --
        # SOD and SDD are GIVEN, so there is nothing to fit and no scale attribute at all. The old least-squares scale silently absorbed any operator or
        # level error instead of surfacing it.
        self._anchor_cache: dict[int, torch.Tensor] = {}   # idx -> NET static FDK, on CPU

        # ---- SIMULATION GRID (2026-07-29, user directive: "순결한 LEAP 일대일 대응") ----------
        # Every measurement y in this project is now simulated by forward-projecting the volume
        # resampled to the geometry's NATIVE voxel size -- du * SOD/SDD, the detector pitch back-
        # projected to the isocentre, which is LEAP's own reconstruction-grid convention (their
        # `set_default_volume`) -- while the INVERSION (FDK, CG, the estimator's dP, the bridge)
        # keeps running on the coarse `voxel_mm` grid. Same operator, same call path:
        # THE ONLY CHANGE IS THE GRID THE TRUTH IS SAMPLED ON.
        #
        # WHY. Simulating on the grid you invert on is the classic inverse crime, and here it had a
        # visible cost: the SF operator integrates the CUBE voxel basis, whose faces carry real
        # energy above the voxel Nyquist, and the detector RESOLVES it (0.4187 mm at iso vs 1 mm
        # voxels = 2.39x finer), so the static FDK came out with a crosshatch texture the user
        # spotted in the validation montages. MEASURED on p218 (excess sd in a homogeneous brain
        # ROI, static FDK): 1 mm simulation 21.3 HU under a bare ram-lak / 2.9 HU under the
        # deployed `shepphann`; NATIVE simulation 1.9 HU / 0.0 HU at equal-or-better sharpness.
        # The installed LEAP reproduces the same defect end-to-end when driven at 1 mm (9.5 HU with
        # all their defaults) and is clean at its native grid -- so this is the field's convention,
        # not a workaround. Thies et al. simulate from the native MDCT volumes for the same reason.
        # CQ500 is ALREADY scanner-native at ~0.41 mm in-plane, so nothing is invented here: we
        # simply stop throwing that resolution away before projecting.
        #
        # COST. Voxel-driven SF at 612^3 x 360 views is 2.88 s (the installed LEAP's own forward
        # needs 2.76 s on the same task, i.e. we are at parity) and the DICOM read + resample is
        # 1.4 s, which is why `volume_fine` keeps only a 1-deep in-RAM cache and NO disk cache:
        # caching 150 patients as fp32 would cost 131 GB to save 1 s, and fp16 would quantize the
        # truth by up to 1 HU.
        self.sim_native = bool(sim_native)
        du = float(self.u_coords[1] - self.u_coords[0])
        dv = float(self.v_coords[1] - self.v_coords[0])
        self.sim_voxel_mm = min(du, dv) * self.cfg.SOD / self.cfg.SDD
        # the fine box must COVER the coarse one (same physical FOV, centred), hence ceil
        self.sim_shape_dhw = tuple(
            int(math.ceil(n * d / self.sim_voxel_mm - 1e-9))
            for n, d in zip(self.shape_dhw, (self.dz, self.dy, self.dx)))
        self._fine_cache: tuple[int, torch.Tensor] | None = None  # (key, PINNED fp32 hu)
        self._fine_futures: dict[int, object] = {}  # key -> Future -- see `prefetch_fine`
        self._fine_pool = None     # lazy 1-thread executor backing `prefetch_fine`
        if verbose and self.sim_native:
            print(f"[cq500] simulation grid: {'x'.join(map(str, self.sim_shape_dhw))} @ "
                  f"{self.sim_voxel_mm:.5f} mm (native = du*SOD/SDD) | "
                  f"inversion grid: {'x'.join(map(str, self.shape_dhw))} @ {self.dx:g} mm")

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

    def _raw_series(self, r: dict):
        """The DICOM series as an sitk image on ITS OWN grid, disk-cached LOSSLESSLY as int16.

        Reading a CQ500 series costs 1-15 s and it is DECOMPRESSION, not IO (measured: a warm
        re-read costs the same as a cold one), which the native simulation grid can no longer
        absorb -- `volume_fine` is called on every bridge draw. So the decoded series is cached
        once, and every later resample (coarse OR fine) starts from the cache.

        int16 is EXACT here, not an approximation: CQ500's HU are integers after the DICOM
        rescale (verified max|HU - round(HU)| = 0 across patients) and span [-3024, 3071].
        ~120-150 MB per patient, vs 874 MB if the fine grid itself were cached."""
        import SimpleITK as sitk

        os.makedirs(self.cache_dir, exist_ok=True)
        npz = os.path.join(self.cache_dir, f"p{r['patient']:04d}_raw_i16.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            img = sitk.GetImageFromArray(z["hu"].astype(np.float32))
            img.SetSpacing([float(v) for v in z["spacing"]])
            img.SetOrigin([float(v) for v in z["origin"]])
            img.SetDirection([float(v) for v in z["direction"]])
            return img
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(r["path"], r["series"]))
        img = reader.Execute()                                    # HU already (rescale applied)
        hu = sitk.GetArrayFromImage(img).astype(np.float32)
        if np.abs(hu - np.round(hu)).max() == 0.0 and hu.min() >= -32768 and hu.max() <= 32767:
            tmp = npz + f".tmp{os.getpid()}"                       # atomic: many jobs share this
            with open(tmp, "wb") as fh:                             # a path arg would get ".npz"
                np.savez(fh, hu=hu.astype(np.int16), spacing=np.array(img.GetSpacing()),
                         origin=np.array(img.GetOrigin()), direction=np.array(img.GetDirection()))
            os.replace(tmp, npz)
        return img

    def _load_hu(self, r: dict, voxel_mm: float | None = None,
                 shape_dhw: tuple[int, int, int] | None = None) -> np.ndarray:
        """DICOM series -> HU on an isotropic grid, centre-cropped/padded to `shape`.

        Defaults to the INVERSION grid (`voxel_mm`, `shape`); `volume_fine` passes the finer
        SIMULATION grid instead. Both go through the identical resample, so the two grids differ
        only in sampling density -- not in interpolation, orientation, origin or air padding."""
        import SimpleITK as sitk

        vm = self.dx if voxel_mm is None else float(voxel_mm)
        shp = self.shape_dhw if shape_dhw is None else tuple(shape_dhw)
        img = self._raw_series(r)

        # resample to an isotropic grid -- the whole reason dz stops being an assumption
        sp_in = np.array(img.GetSpacing(), dtype=np.float64)       # (x, y, z)
        sz_in = np.array(img.GetSize(), dtype=np.int64)            # (x, y, z)
        sp_out = np.array([vm, vm, vm], dtype=np.float64)
        sz_out = np.maximum(np.round(sz_in * sp_in / sp_out).astype(int), 1)

        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing(sp_out.tolist())
        rs.SetSize([int(s) for s in sz_out])
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1000.0)                           # air, not zero (= water)
        vol = sitk.GetArrayFromImage(rs.Execute(img)).astype(np.float32)   # (z, y, x) = (D,H,W)
        return _centre_fit(vol, shp, pad_value=-1000.0)

    def _fine_load_pinned(self, key: int) -> torch.Tensor:
        """The CPU side of `volume_fine`: load + resample + clip, returned PAGE-LOCKED.

        This is what `prefetch_fine` runs on its worker thread. Pinning here (a host memcpy)
        rather than in `volume_fine` keeps the training thread's cost to one async DMA; every
        stage releases the GIL (npz IO, the sitk resample, the pin memcpy), so the worker
        genuinely overlaps with the training step."""
        hu = self._load_hu(self.records[key], voxel_mm=self.sim_voxel_mm,
                           shape_dhw=self.sim_shape_dhw)
        if self.clip:                                      # idempotent, so do it once per load
            np.clip(hu, self.hu_lo, self.hu_hi, out=hu)
        return torch.from_numpy(hu).pin_memory()

    def prefetch_fine(self, idx: int) -> None:
        """Start loading patient `idx`'s native-grid volume on a background thread.

        WHY: the trainer refreshes one bridge draw every `--refresh` steps, and with 150 train
        patients the 1-entry fine cache below misses essentially every time. The synchronous
        cost was measured at ~1 s per draw (0.19 s npz+f32, 0.48 s sitk resample, ~0.2 s
        pageable H2D) -- a 0% GPU dip every 6 s of wall clock, ~15% of the whole run. The
        trainer samples the NEXT draw's patient one draw ahead and calls this, so the load
        rides under the 12 training steps in between. No-op when the volume is already cached,
        already in flight, or `sim_native` is off (nothing fine to load).

        Futures live in a DICT keyed by patient, not a single slot: the trainer prefetches
        draw k+1's patient at the START of draw k, i.e. BEFORE draw k consumes its own
        prefetch -- a single slot gets overwritten right there and every join degrades to an
        inline load (the first deployment shipped that bug; the draw profiler caught it as a
        constant ~1.5 s inside sim_y). Entries are popped on use, so the dict only ever holds
        what is in flight (a resume can strand one entry; it is still a valid load and gets
        consumed whenever that patient comes up again)."""
        if not self.sim_native:
            return
        key = int(idx) % len(self.records)
        if self._fine_cache is not None and self._fine_cache[0] == key:
            return
        if key in self._fine_futures:
            return
        if self._fine_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._fine_pool = ThreadPoolExecutor(max_workers=1)
        self._fine_futures[key] = self._fine_pool.submit(self._fine_load_pinned, key)

    def volume_fine(self, idx: int = 0) -> torch.Tensor:
        """(1,1,Ds,Hs,Ws) mu volume of patient `idx` on the NATIVE SIMULATION grid.

        This is the object that gets forward-projected to make y (see `simulate`); the coarse
        `volume(idx)` remains the reconstruction target and the metric's ground truth. Not
        disk-cached on purpose -- see the simulation-grid note in `__init__` -- but the last
        volume is kept in RAM (pinned), which is what makes a first visit to a patient (y under
        motion AND the static anchor's y0) cost one resample instead of two. If `prefetch_fine`
        was called for this patient, the load has been running on the worker thread and this
        only joins it."""
        key = int(idx) % len(self.records)
        if self._fine_cache is not None and self._fine_cache[0] == key:
            hu = self._fine_cache[1]
        else:
            fut = self._fine_futures.pop(key, None)
            if fut is not None:
                hu = fut.result()
            else:                                          # cold / mispredicted: load inline
                hu = self._fine_load_pinned(key)
            self._fine_cache = (key, hu)
        # HU -> mu ON THE GPU: at 612^3 the same arithmetic on the host costs ~0.5 s per draw.
        # `hu` is pinned, so this H2D is a true async DMA, not a pageable sync copy.
        v = hu[None, None].to(self.device, non_blocking=True)
        return (v / 1000.0 + 1.0) * self.mu_water

    @torch.no_grad()
    def simulate(self, idx: int, Pmat: torch.Tensor, **kw) -> torch.Tensor:
        """THE measurement operator of this project: y = A(patient `idx`; Pmat), simulated on the
        native grid. (B,V,3,4) or (V,3,4) -> (B,V,nv,nu).

        Every y in training, validation and inference comes from here, so the simulation grid can
        never drift between them. `sim_native=False` degrades it to the old inverse-crime path
        (project the coarse volume) -- kept ONLY as a gate/ablation switch.

        No autograd: y is data. The estimator differentiates its OWN forward model of y on the
        coarse grid, which is the point of the split (see the `__init__` note)."""
        P = Pmat if Pmat.dim() == 4 else Pmat[None]
        if not self.sim_native:
            return self.project(self.volume(idx), P, **kw)
        vol = self.volume_fine(idx)
        y = forward_project_3d_batched(
            vol, P, self.u_coords, self.v_coords,
            dx=self.sim_voxel_mm, dy=self.sim_voxel_mm, dz=self.sim_voxel_mm, **kw)
        del vol
        return y

    # -- net <-> mu ---------------------------------------------------------------------
    def to_net(self, mu):
        return 2.0 * (mu - self.mu_lo) / (self.mu_hi - self.mu_lo) - 1.0

    def to_net_tangent(self, dmu):
        """to_net is affine, so a DERIVATIVE maps with the gain only (no -1 shift)."""
        return 2.0 * dmu / (self.mu_hi - self.mu_lo)

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
            dx=self.dx, dy=self.dy, dz=self.dz, **kw)

    def adjoint(self, sino: torch.Tensor, Pmat: torch.Tensor, **kw) -> torch.Tensor:
        """A^T sino -- LEAP's backprojection, NOT the exact transpose of `project`
        (`leap_projector.ADJOINT_MODE`). (B,V,nv,nu) -> (B,1,D,H,W)."""
        D, H, W = self.shape_dhw
        return adjoint_project_3d_batched(
            sino, Pmat, self.u_coords, self.v_coords,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz, **kw)


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
            scale=scale,
            view_chunk=kw.pop("view_chunk", 8), view_weight=vw, **kw)

    def fdk_tangent(self, sino: torch.Tensor, Pmat: torch.Tensor, Pdot: torch.Tensor, *,
                    scale=None, angle_weight: bool | None = None, **kw):
        """(FDK(sino, Pmat), its exact directional derivative along Pdot) -- one fused pass.

        The tangent twin of `fdk`: same scale, same Voronoi angular weighting (plus the
        weight's OWN derivative, which is s-dependent through the gantry-axis rotation).
        Used by `train_fm3d.bridge_pair` with (Pmat, Pdot) = rigid_motion.bridge_P_and_dP."""
        D, H, W = self.shape_dhw
        aw = self.angle_weight if angle_weight is None else bool(angle_weight)
        vw = vwd = None
        if aw:
            vw, vwd = view_angular_weights_dot(Pmat, Pdot)
        return fdk_conebeam_3d_tangent(
            sino, Pmat, Pdot, self.u_coords, self.v_coords, self.cfg,
            D=D, H=H, W=W, dx=self.dx, dy=self.dy, dz=self.dz,
            scale=scale,
            view_weight=vw, view_weight_dot=vwd, **kw)


    def sample_volumes(self, batch: int = 1, generator=None) -> torch.Tensor:
        idx = torch.randint(len(self.records), (batch,), generator=generator)
        return torch.cat([self.volume(int(i)) for i in idx], 0)

    @torch.no_grad()
    def static_anchor_net(self, idx: int) -> torch.Tensor:
        """NET-space motion-free FDK of volume `idx` -- x_anchor = to_net(FDK(project(vol,
        P_nom), P_nom)) -- memoized on CPU (150 vols ~= 10 GB fp32).

        This is the bridge's t=1 anchor (see train_fm3d.bridge_pair). It depends ONLY on the
        volume: neither the sampled motion nor t enter it, so recomputing it on every draw was
        pure waste. Caching it removes one full forward projection (~0.9 s) and one FDK (~0.18 s)
        from every bridge draw of a volume already seen once -- the dominant cost after the
        forward-projection twin. Numerically identical to the inline compute (same ops, a
        lossless fp32 CPU round-trip)."""
        key = int(idx) % len(self.records)
        a = self._anchor_cache.get(key)
        if a is None:
            y0 = self.simulate(key, self.P_nom[None])
            a = self.to_net(self.fdk(y0, self.P_nom[None])[0]).cpu()
            self._anchor_cache[key] = a
        return a.to(self.device)

    def sample_motion(self, batch: int = 1, *, trans_mm: float = 10.0, rot_deg: float = 10.0,
                      amp_mode: str = "fixed", generator=None):
        """-> (y (B,V,nv,nu), theta (B,V,6), vols (B,1,D,H,W)). Defaults are the literature's
        evaluation amplitudes, PEAK-TO-PEAK (Thies evaluates at 5/5; we run 10/10, i.e. 2x -- see
        the rigid_motion module header).

        `amp_mode="thies"` switches to the paper's TRAINING amplitude protocol, where those
        numbers become per-DoF maxima -- see `rigid_motion.akima_motion`."""
        idx = torch.randint(len(self.records), (batch,), generator=generator)
        vols = torch.cat([self.volume(int(i)) for i in idx], 0)
        th = torch.stack([
            random_motion(self.cfg.n_views, trans_mm=trans_mm, rot_deg=rot_deg,
                          amp_mode=amp_mode, device=self.device, generator=generator)
            for _ in range(batch)], 0)
        P = torch.stack([params_to_Pmot(th[b], self.P_nom) for b in range(batch)], 0)
        # y comes from the NATIVE grid, one patient at a time (the fine volumes are ~900 MB each,
        # so they are never batched); the returned `vols` stay on the coarse inversion grid.
        with torch.no_grad():
            y = torch.cat([self.simulate(int(idx[b]), P[b:b + 1]) for b in range(batch)], 0)
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
