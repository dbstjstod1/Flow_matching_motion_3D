"""Gate the CQ500 setup: the standard geometry, and the selection/split rules -- WITHOUT the data.

The real CQ500 is not on disk yet, and this project has already been bitten once by shipping a
script that was "syntactically valid, nothing more". So this gate does not wait for the download:
it SYNTHESIZES a small CQ500-shaped DICOM tree (real DICOM, written and re-read through SimpleITK,
with thin series, thick series and slice-count outliers deliberately planted in it) and runs the
actual `index_cq500 / select_series / split_patients / CQ500Generator` code over it.

  [1] geometry: the derived numbers of `ConeBeam3DConfig.thies()`, and the one thing the papers
      never say -- WHICH axis of the "500 x 700" panel is lateral. Only nu=700 gives an FOV that
      contains a head; nu=500 truncates it at every view. Checked, not assumed.
  [2] indexing: a synthetic tree round-trips through DICOM (thickness, spacing, slice count)
  [3] selection: 5 mm series dropped, slice-count outliers dropped, one series per patient,
      and the survivor is the one with the most slices
  [4] split: sequential, patient-level, DISJOINT, no RNG (two calls agree bit-for-bit), and the
      Thies counts (150/50/rest) reproduce on a 320-patient set
  [5] volumes: HU clipped, padded with AIR (not water), mu conversion, to_net/from_net roundtrip
  [6] operator: project -> FDK through the CQ500 generator actually reconstructs the phantom

  python scripts/gate_cq500.py          # ~1 min, no data, no checkpoint
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.dataset_cq500 import (CQ500Generator, _centre_fit, index_cq500, patient_id,
                                select_series, split_patients)
from fm3d.geometry_3d import ConeBeam3DConfig

FAIL = []


def check(idx, name, ok, detail=""):
    print(f"[{idx}] {name:<56s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAIL.append(name)


def head_phantom(nz, ny, nx, dz=1.0, dy=1.0, dx=1.0) -> np.ndarray:
    """A crude head in HU: air outside, a skull shell at 1200, brain at 40, a 2500 HU implant.

    The semi-axes are FRACTIONS OF THE VOLUME'S OWN PHYSICAL EXTENT, not absolute millimetres --
    the synthetic series here are far smaller than a real head, and an absolute 85 mm ellipsoid
    would swallow the whole box, leaving a volume of pure brain with no skull and no implant in
    it (which is exactly how the first version of this gate lied)."""
    ext = np.array([nz * dz, ny * dy, nx * dx]) * 0.5           # half-extent per axis [mm]
    z = (np.arange(nz) - (nz - 1) / 2) * dz
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt((Z / (0.85 * ext[0])) ** 2 + (Y / (0.85 * ext[1])) ** 2
                + (X / (0.85 * ext[2])) ** 2)
    hu = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    hu[r < 1.00] = 1200.0                                       # skull
    hu[r < 0.90] = 40.0                                         # brain
    imp = (X ** 2 + (Y + 0.6 * ext[1]) ** 2 + Z ** 2) < (0.10 * ext[2]) ** 2
    hu[imp] = 2500.0                                            # dental implant (metal)
    return hu


def write_series(out_dir, hu, thickness, in_plane_mm, desc, uid_seed):
    """Write `hu` (D,H,W) as a real DICOM series SimpleITK's GDCM reader will group."""
    import SimpleITK as sitk
    os.makedirs(out_dir, exist_ok=True)
    series_uid = f"1.2.826.0.1.3680043.2.1125.{uid_seed}"
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for i in range(hu.shape[0]):
        sl = sitk.GetImageFromArray(hu[i:i + 1].astype(np.int16))
        sl.SetSpacing((in_plane_mm, in_plane_mm, thickness))
        for k, v in [("0008|0060", "CT"), ("0008|103e", desc),
                     ("0018|0050", f"{thickness}"),               # SliceThickness
                     ("0020|000e", series_uid),                   # SeriesInstanceUID
                     ("0020|0013", str(i)),                       # InstanceNumber
                     ("0020|0032", f"0\\0\\{i * thickness}"),      # ImagePositionPatient
                     ("0020|0037", "1\\0\\0\\0\\1\\0"),
                     ("0028|1052", "0"), ("0028|1053", "1")]:     # rescale intercept/slope
            sl.SetMetaData(k, v)
        w.SetFileName(os.path.join(out_dir, f"s{i:04d}.dcm"))
        w.Execute(sl)


def build_tree(root, n_patients=8, n_slices=96):
    """A CQ500-shaped tree with the three things `select_series` must survive."""
    hu = head_phantom(n_slices, 96, 96, dz=0.625, dy=0.6, dx=0.6)
    for p in range(n_patients):
        base = os.path.join(root, f"CQ500CT{p} CQ500CT{p}", "Unknown Study")
        # every patient: a 0.625 mm thin series ...
        n = n_slices
        if p == 0:
            n = n_slices // 4                 # slice-count outlier (too few)  -> must be dropped
        if p == 1:
            n = n_slices * 3                  # slice-count outlier (too many) -> must be dropped
        h = hu if n <= n_slices else np.concatenate([hu] * 3, 0)
        write_series(os.path.join(base, "CT thin"), h[:n], 0.625, 0.6, "CT 0.625mm", 100 + p)
        # ... plus a 5 mm thick series, which must be filtered out
        write_series(os.path.join(base, "CT plain"), hu[::8], 5.0, 0.6, "CT plain", 200 + p)
        # patient 2 also gets a SECOND thin series with fewer slices: the fuller one must win
        if p == 2:
            write_series(os.path.join(base, "CT thin b"), hu[:n_slices - 10], 0.625, 0.6,
                         "CT 0.625mm b", 300 + p)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- [1] geometry ------------------------------------------------------------------
    cfg = ConeBeam3DConfig.thies()
    fov, axial, pitch = cfg.fov_diameter_mm(), cfg.axial_coverage_mm(), cfg.iso_pitch_mm()
    check(1, "thies(): SOD 785 / SDD 1200 / 360 views",
          (cfg.SOD, cfg.SDD, cfg.n_views) == (785.0, 1200.0, 360))
    check(1, "thies(): 700x500 panel @ 0.64 mm",
          (cfg.nu, cfg.nv, cfg.du) == (700, 500, 0.64))
    check(1, "magnification 1.529", abs(cfg.magnification - 1.5287) < 1e-3,
          f"M={cfg.magnification:.4f}")
    check(1, "FOV 288 mm (a 180 mm head fits)", 285.0 < fov < 292.0 and fov > 200.0,
          f"{fov:.1f} mm")
    check(1, "axial coverage 209 mm on-axis", 206.0 < axial < 212.0, f"{axial:.1f} mm")
    check(1, "iso pitch 0.419 mm", abs(pitch - 0.4187) < 1e-3, f"{pitch:.4f} mm")

    # the assignment the papers never state: 500 x 700 -- which one is lateral?
    swapped = ConeBeam3DConfig(SOD=785.0, SDD=1200.0, det_nu=500, det_nv=700,
                               det_pixel_mm=0.64, n_views=360)
    check(1, "nu=700 is FORCED: nu=500 would truncate a head",
          swapped.fov_diameter_mm() < 210.0 < fov,
          f"nu=500 -> FOV {swapped.fov_diameter_mm():.1f} mm  vs  nu=700 -> {fov:.1f} mm")
    check(1, "256 mm recon box EXCEEDS the 209 mm axial coverage (physical truncation)",
          axial < 256.0, f"{256.0 - axial:.0f} mm of a 256^3 @1mm box is unmeasured")

    jrm = ConeBeam3DConfig.jrm_adm()
    check(1, "jrm_adm(): 0.5 mm pitch, 120 views, tighter FOV",
          (jrm.du, jrm.n_views) == (0.5, 120) and jrm.fov_diameter_mm() < fov,
          f"FOV {jrm.fov_diameter_mm():.1f} mm, axial {jrm.axial_coverage_mm():.1f} mm")

    # ---- [2..6] the dataset code, on a SYNTHETIC CQ500 tree -----------------------------
    root = tempfile.mkdtemp(prefix="cq500_gate_")
    try:
        build_tree(root, n_patients=8, n_slices=96)

        check(2, "patient_id() parses the CQ500 folder convention",
              patient_id("/x/CQ500CT23 CQ500CT23/Unknown Study/CT thin") == 23
              and patient_id("/x/nope") is None)

        recs = index_cq500(root, verbose=False)
        thin = [r for r in recs if r["thickness_mm"] < 1.0]
        thick = [r for r in recs if r["thickness_mm"] > 4.0]
        check(2, "index_cq500 round-trips thickness through real DICOM",
              len(thin) == 9 and len(thick) == 8, f"{len(thin)} thin, {len(thick)} thick series")
        # The cache claim, tested for real: monkeypatch the module's os.walk so any re-scan is
        # observable, re-call, and demand (a) zero walks and (b) records identical to the scan's.
        # Merely checking the JSON exists would pass even if the 2nd call ignored it.
        import fm3d.dataset_cq500 as _dsq
        walks = {"n": 0}
        _orig_walk = _dsq.os.walk

        def _counting_walk(*a, **k):
            walks["n"] += 1
            return _orig_walk(*a, **k)

        _dsq.os.walk = _counting_walk
        try:
            recs2 = index_cq500(root, verbose=False)
        finally:
            _dsq.os.walk = _orig_walk
        check(2, "index is cached (2nd call reads JSON, no re-scan)",
              os.path.exists(os.path.join(root, "_cq500_index.json"))
              and walks["n"] == 0 and recs2 == recs,
              f"walk calls: {walks['n']}, records equal: {recs2 == recs}")

        sel = select_series(recs, thin_mm=0.7, count_tol=0.5, min_slices=16)
        pats = [r["patient"] for r in sel]
        check(3, "5 mm series dropped", all(r["thickness_mm"] <= 0.7 for r in sel))
        check(3, "slice-count outliers dropped (patients 0 and 1)",
              0 not in pats and 1 not in pats, f"survivors: {pats}")
        check(3, "one series per patient", len(pats) == len(set(pats)))
        check(3, "the FULLER of patient 2's two thin series wins",
              next(r["n_slices"] for r in sel if r["patient"] == 2) == 96)

        parts = split_patients(sel, counts=(2, 1, 3))
        ids = {k: [r["patient"] for r in v] for k, v in parts.items()}
        check(4, "split is sequential by patient index",
              ids["train"] == [2, 3] and ids["val"] == [4] and ids["test"] == [5, 6, 7], str(ids))
        check(4, "splits are patient-DISJOINT",
              not (set(ids["train"]) & set(ids["val"]) | set(ids["val"]) & set(ids["test"])
                   | set(ids["train"]) & set(ids["test"])))
        check(4, "no RNG: two calls are identical",
              split_patients(sel, counts=(2, 1, 3)) == parts)
        fake = [{"patient": i, "n_slices": 300} for i in range(320)]
        p320 = split_patients(fake, counts=(150, 50, 120))
        check(4, "Thies' 320 -> 150 / 50 / 120 exactly",
              (len(p320["train"]), len(p320["val"]), len(p320["test"])) == (150, 50, 120))

        # ---- [5] volumes -----------------------------------------------------------------
        small = _centre_fit(np.zeros((4, 4, 4), np.float32), (8, 8, 8))
        check(5, "_centre_fit pads with AIR (-1000), not water (0)",
              small[0, 0, 0] == -1000.0 and small[4, 4, 4] == 0.0)

        gen = CQ500Generator(root, cfg=ConeBeam3DConfig.thies(n_views=90), device=dev,
                             split="train", shape=(96, 128, 128), voxel_mm=1.0,
                             split_counts=(2, 1, 3), thin_mm=0.7, count_tol=0.5, verbose=False)
        v = gen.volume(0)
        hu_back = v[0, 0].cpu().numpy() / gen.mu_water * 1000.0 - 1000.0
        check(5, "HU clipped to the window (the 2500 HU implant -> 2000)",
              hu_back.max() <= 2000.5 and hu_back.max() > 1900.0, f"max {hu_back.max():.0f} HU")
        check(5, "air stays air after resample+pad", hu_back.min() < -950.0,
              f"min {hu_back.min():.0f} HU")
        rt = float((gen.from_net(gen.to_net(v)) - v).abs().max())
        check(5, "to_net/from_net roundtrip", rt < 1e-8, f"max|d|={rt:.1e}")
        check(5, "split membership: train = patients 2,3",
              [r["patient"] for r in gen.records] == [2, 3])

        # ---- [6] the operator through the CQ500 generator ---------------------------------
        with torch.no_grad():
            y = gen.project(v, gen.P_nom[None])
            rec = gen.fdk(y, gen.P_nom[None])[0]
        from fm3d.geometry_3d import measured_region_mask
        m = measured_region_mask(gen.shape, (gen.dz, gen.dy, gen.dx), gen.cfg, device=v.device)
        err = (rec - v[0, 0])[m]
        rng = float(v[0, 0][m].max() - v[0, 0][m].min())
        psnr = 20 * np.log10(rng / float(err.pow(2).mean().sqrt()) + 1e-12)
        check(6, "project -> FDK reconstructs the phantom (in the measured region)",
              torch.isfinite(rec).all() and psnr > 20.0, f"{psnr:.1f} dB (90 views)")
        check(6, "fbp_scale calibrated finite", np.isfinite(gen.fbp_scale) and gen.fbp_scale > 0,
              f"{gen.fbp_scale:.5g}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    n = len(FAIL)
    print(f"\n{'ALL PASS' if n == 0 else f'{n} FAILURE(S): ' + ', '.join(FAIL)}")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
