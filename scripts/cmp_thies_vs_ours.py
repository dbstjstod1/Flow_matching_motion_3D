"""PAIRED comparison, ours vs the Thies baseline, over the 30-patient test cohort.

    python scripts/cmp_thies_vs_ours.py

WHAT MAKES IT PAIRED, and it is the only thing that does: both sides were run over the same
(split=test, run=i, seed=1000+i) triples, so `run_posterior3d.build_world` handed each method the
byte-identical patient, geometry, motion draw and natively-simulated sinogram. The two drivers
are `scripts/drivers/drive_test30.sh` (ours) and `scripts/drivers/drive_thies_test30.sh`
(theirs); this script refuses to compare a patient that is missing from either side rather than
quietly averaging over different cohorts.

BECAUSE IT IS PAIRED, THE TEST IS A PAIRED ONE. Per-patient differences are reported with a
Wilcoxon signed-rank test (no normality assumption, n=30) alongside the means -- a two-sample
test would throw away exactly the patient-to-patient variance the pairing was set up to remove.

THREE THINGS THAT ARE EASY TO COMPARE WRONGLY HERE
--------------------------------------------------
1. **Which volume is "our output".** This project carries TWO deliverables and they rank
   oppositely against the two references (see the metrics note in the repo docs): `output` is
   FDK(theta_hat), `x_t` is the carried PnP state. Thies has only one deliverable -- one
   reconstruction with the final estimate -- so BOTH of ours are printed against it and neither
   is silently chosen.
2. **Which reference.** Ours are habitually quoted vs the GT VOLUME; Thies quotes vs the
   MOTION-FREE RECONSTRUCTION. Only the vs-static-reconstruction column is on the same footing as
   his published SSIM 0.94, and the two references flip the ranking. Both are printed.
   NOTE the "static" reference is not literally the same volume on the two sides: ours is our
   FDK of the motion-free scan, theirs is THEIR unweighted-sum reconstruction of it. That is
   correct -- each method is scored against what its own operator can produce of a still patient
   -- but it is not a common yardstick, so the vs-GT column is the one that compares operators.
3. **RPE.** `rpe_mm` (raw) is the number comparable to the paper's 0.61 mm. The gauge-quotiented
   one is ours and is not what he reports; both are carried because blind CT motion has an exact
   SE(3) gauge (see the gauge note in the repo docs).

OUR RPE IS COMPUTED HERE, NOT READ OFF DISK. `run_posterior3d.py` scores theta with
`motion_error` (a parameter-domain split) and never computes an RPE, so the comparison would
otherwise have a Thies-only column. It is recoverable offline and exactly: RPE is a function of
(theta_hat, theta_true, P_nom) alone, all three of which are either saved in `result.pt` or pure
geometry. `P_nom` is rebuilt from `ConeBeam3DConfig.thies(n_views=V)` with V read off the saved
theta -- the same call `build_world` makes for a CQ500 checkpoint -- so no volume, sinogram,
checkpoint or GPU is touched. THE ONE ASSUMPTION is that both sides ran the Thies geometry; the
loader asserts V matches on the two sides and `bench_thies_estimate` builds its world through the
very same `build_world`, so a geometry mismatch cannot survive the theta_true pairing check below.

WHICH theta IS "OURS". `result.pt` carries `theta` (the K-step average that produced the reported
output) and `theta_last`. The average is the deliverable, so it is the one scored. `theta_last`
is carried too and the script says whether the two agree: at the deployed `--theta_avg 1` they are
the SAME tensor (K=1 averages one step), so a difference in that row means the cohort was run at
some other K and the deliverable is the averaged one.

THE INITIAL RPE IS ALSO PRINTED (theta = 0, i.e. no compensation). The paper's own table leads
with it -- "initial median RPE ~3 mm" at 5/5 -- and it is the only way to read a corrected RPE as
a reduction factor rather than as a bare millimetre count, which matters here because our cohort
runs at 2x his amplitude.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit   # noqa: E402
from fm3d.rigid_motion import reprojection_error, zero_centre_gauge   # noqa: E402

OURS = "data/fm3d_test30"
THEIRS = "data/bench_thies_test30"
N = 30

_P_NOM: dict[int, "torch.Tensor"] = {}


def p_nom(n_views: int):
    """The nominal orbit, memoized. Pure geometry -- no data, no checkpoint, no GPU."""
    if n_views not in _P_NOM:
        _P_NOM[n_views] = build_conebeam_orbit(
            ConeBeam3DConfig.thies(n_views=n_views), device="cpu").double()
    return _P_NOM[n_views]


def _wilcoxon(d: np.ndarray):
    """Two-sided Wilcoxon signed-rank on the paired differences -> (statistic, p)."""
    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(d)
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


def load():
    rows = []
    missing = []
    for i in range(N):
        tag = f"p{i:02d}"
        ours_p = os.path.join(OURS, tag, "result.pt")
        theirs_p = os.path.join(THEIRS, tag, "result.json")
        if not (os.path.exists(ours_p) and os.path.exists(theirs_p)):
            missing.append((tag, os.path.exists(ours_p), os.path.exists(theirs_p)))
            continue
        o = torch.load(ours_p, map_location="cpu", weights_only=False)
        t = json.load(open(theirs_p))
        # THE PAIRING ASSERTION. Everything below assumes the two methods saw the same scan, and
        # the only way that fails is silently: a mismatched --seed still runs, still converges and
        # still prints a plausible table. Both sides save `theta_true`, so compare them. A
        # mismatch here voids the comparison, so it raises rather than warns.
        th_o = o["theta_true"]
        tp = torch.load(os.path.join(THEIRS, tag, "result.pt"),
                        map_location="cpu", weights_only=False)
        th_t = tp["theta_true"]
        dev = float((th_o - th_t).abs().max())
        if dev > 1e-6:
            raise SystemExit(
                f"{tag}: the two runs did NOT see the same motion (max |dtheta| = {dev:.3e}). "
                f"The cohorts are not paired -- check that both drivers use "
                f"--split test --run i --seed 1000+i.")
        # ---- OUR RPE, computed here (see the module docstring). fp64 throughout: the raw RPE is
        # a difference of two nearly-equal projected point sets, and at 0.6 mm on a 0.64 mm pixel
        # the fp32 cancellation is a percent-level effect on the very digit being compared.
        P = p_nom(th_o.shape[0])
        tt64 = th_o.double()                       # th_o IS theta_true (checked identical above)
        th64 = o["theta"].double()                 # the K-step average = the reported deliverable
        o_rpe = reprojection_error(th64, tt64, P)
        o_rpe_zc = reprojection_error(zero_centre_gauge(th64), tt64, P)
        o_rpe_last = reprojection_error(o["theta_last"].double(), tt64, P)
        o_rpe0 = reprojection_error(torch.zeros_like(tt64), tt64, P)   # no compensation at all
        # THEIR RPE is likewise recomputed here in fp64 from the saved theta_hat, not read off
        # result.json -- pre-2026-08-06 jsons carry an fp32 value, and a precision asymmetry
        # between the two sides of a paired table is exactly the kind of thumb on the scale this
        # script exists to prevent.
        t64 = tp["theta_hat"].double()
        t_rpe = reprojection_error(t64, tt64, P)
        t_rpe_zc = reprojection_error(zero_centre_gauge(t64), tt64, P)
        rows.append(dict(
            tag=tag,
            # --- ours: two deliverables, two references
            o_out_gt=o["final"]["ssim_aligned"], o_out_gt_psnr=o["final"]["psnr_aligned"],
            o_xt_gt=o["final_xt"]["ssim_aligned"], o_xt_gt_psnr=o["final_xt"]["psnr_aligned"],
            o_out_st=o["final_s"]["ssim_aligned"], o_xt_st=o["final_xt_s"]["ssim_aligned"],
            # --- theirs: one deliverable
            t_out_gt=t["vs_gt"]["output"]["ssim_aligned"],
            t_out_gt_psnr=t["vs_gt"]["output"]["psnr_aligned"],
            t_in_gt=t["vs_gt"]["input"]["ssim_aligned"],
            t_out_st=t["vs_thies_static"]["output"]["ssim_aligned"],
            t_in_st=t["vs_thies_static"]["input"]["ssim_aligned"],
            t_rpe=t_rpe["rpe_mm"], t_rpe_g=t_rpe_zc["rpe_mm"],
            t_sec=t["est_seconds"],
            # --- RPE, both sides, on the one convention each is allowed to be quoted on
            o_rpe=o_rpe["rpe_mm"], o_rpe_g=o_rpe_zc["rpe_mm"],
            o_rpe_last=o_rpe_last["rpe_mm"], rpe_init=o_rpe0["rpe_mm"],
            _theta_true=th_o,
        ))
    return rows, missing


def main():
    rows, missing = load()
    if missing:
        print("MISSING (not compared -- the cohorts must match patient for patient):")
        for tag, a, b in missing:
            print(f"  {tag}: ours={'ok' if a else 'MISSING'} thies={'ok' if b else 'MISSING'}")
    n = len(rows)
    if n == 0:
        raise SystemExit("nothing to compare yet")
    print(f"\nPAIRED over {n}/{N} patients (test split, seed 1000+i, 10 mm / 10 deg p2p "
          f"= 2x Thies' own evaluation amplitude)\n")

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print("=" * 78)
    print("SSIM (rigid-aligned) vs the GT VOLUME  -- this repo's reference, compares operators")
    print("=" * 78)
    for name, k in (("Thies  input (uncompensated)", "t_in_gt"),
                    ("Thies  output", "t_out_gt"),
                    ("ours   output = FDK(theta_hat)", "o_out_gt"),
                    ("ours   x_t    = carried PnP state", "o_xt_gt")):
        v = col(k)
        print(f"  {name:34s} {v.mean():.4f} +- {v.std(ddof=1):.4f}   "
              f"[{v.min():.4f}, {v.max():.4f}]")
    for label, a, b in (("ours x_t  - Thies output", "o_xt_gt", "t_out_gt"),
                        ("ours FDK  - Thies output", "o_out_gt", "t_out_gt")):
        d = col(a) - col(b)
        s, p = _wilcoxon(d)
        print(f"  paired diff  {label:26s} {d.mean():+.4f}  "
              f"(win {int((d > 0).sum())}/{n})  Wilcoxon p = {p:.2e}")

    print()
    print("=" * 78)
    print("SSIM (rigid-aligned) vs each method's OWN motion-free reconstruction")
    print("  (Thies' protocol; his published number is 0.94 at HALF this motion amplitude)")
    print("=" * 78)
    for name, k in (("Thies  input (uncompensated)", "t_in_st"),
                    ("Thies  output", "t_out_st"),
                    ("ours   output", "o_out_st"),
                    ("ours   x_t", "o_xt_st")):
        v = col(k)
        print(f"  {name:34s} {v.mean():.4f} +- {v.std(ddof=1):.4f}")

    print()
    print("=" * 78)
    print("PSNR (rigid-aligned) vs the GT VOLUME [dB]")
    print("=" * 78)
    for name, k in (("Thies  output", "t_out_gt_psnr"),
                    ("ours   output", "o_out_gt_psnr"),
                    ("ours   x_t", "o_xt_gt_psnr")):
        v = col(k)
        print(f"  {name:34s} {v.mean():6.2f} +- {v.std(ddof=1):.2f}")
    print()
    print("=" * 78)
    print("RPE [mm] -- 300 points at r = 25/50/100 mm, recovered vs target geometry")
    print("  RAW is the paper's definition (his 0.61 mm, at HALF this amplitude).")
    print("  ZERO-CENTRED uses no ground truth -- it is the gauge the simulator and the paper")
    print("  both already write theta in, so it is a legitimate second reading, not a fit.")
    print("=" * 78)
    for name, k in (("no compensation (theta = 0)", "rpe_init"),
                    ("Thies  raw", "t_rpe"),
                    ("Thies  zero-centred", "t_rpe_g"),
                    ("ours   raw          (theta avg)", "o_rpe"),
                    ("ours   raw          (theta last)", "o_rpe_last"),
                    ("ours   zero-centred (theta avg)", "o_rpe_g")):
        v = col(k)
        print(f"  {name:34s} {v.mean():7.3f} +- {v.std(ddof=1):6.3f}  "
              f"median {np.median(v):7.3f}  [{v.min():.3f}, {v.max():.3f}]")
    if float(np.abs(col("o_rpe") - col("o_rpe_last")).max()) < 1e-9:
        print("  (avg == last on every patient: the cohort ran at --theta_avg 1)")
    for label, a, b in (("ours raw   - Thies raw", "o_rpe", "t_rpe"),
                        ("ours zeroc - Thies zeroc", "o_rpe_g", "t_rpe_g")):
        d = col(a) - col(b)
        s, p = _wilcoxon(d)
        print(f"  paired diff  {label:26s} {d.mean():+.3f} mm  "
              f"(ours lower on {int((d < 0).sum())}/{n})  Wilcoxon p = {p:.2e}")
    v = col("t_sec")
    print(f"  Thies estimation wall time [s]     {v.mean():7.1f} +- {v.std(ddof=1):.1f}")

    print()
    print("per-patient (SSIM vs GT):  tag   thies_in  thies_out   ours_out   ours_x_t")
    for r in rows:
        print(f"  {r['tag']}  {r['t_in_gt']:.4f}   {r['t_out_gt']:.4f}     "
              f"{r['o_out_gt']:.4f}     {r['o_xt_gt']:.4f}")

    out = os.path.join(THEIRS, "paired_vs_ours.json")
    json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
