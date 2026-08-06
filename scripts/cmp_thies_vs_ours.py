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
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OURS = "data/fm3d_test30"
THEIRS = "data/bench_thies_test30"
N = 30


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
        th_t = torch.load(os.path.join(THEIRS, tag, "result.pt"),
                          map_location="cpu", weights_only=False)["theta_true"]
        dev = float((th_o - th_t).abs().max())
        if dev > 1e-6:
            raise SystemExit(
                f"{tag}: the two runs did NOT see the same motion (max |dtheta| = {dev:.3e}). "
                f"The cohorts are not paired -- check that both drivers use "
                f"--split test --run i --seed 1000+i.")
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
            t_rpe=t["rpe"]["rpe_mm"], t_rpe_g=t["rpe_zero_centred"]["rpe_mm"],
            t_sec=t["est_seconds"],
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
    print("PSNR (rigid-aligned) vs GT [dB]        |  Thies motion estimate")
    print("=" * 78)
    for name, k in (("Thies  output", "t_out_gt_psnr"),
                    ("ours   output", "o_out_gt_psnr"),
                    ("ours   x_t", "o_xt_gt_psnr")):
        v = col(k)
        print(f"  {name:34s} {v.mean():6.2f} +- {v.std(ddof=1):.2f}")
    for name, k in (("Thies RPE raw [mm]  (his 0.61 @ half amp)", "t_rpe"),
                    ("Thies RPE zero-centred [mm]", "t_rpe_g"),
                    ("Thies estimation wall time [s]", "t_sec")):
        v = col(k)
        print(f"  {name:42s} {v.mean():7.3f} +- {v.std(ddof=1):.3f}  "
              f"median {np.median(v):.3f}")

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
