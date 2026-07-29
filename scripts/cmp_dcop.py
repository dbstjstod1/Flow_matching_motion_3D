"""Side-by-side table of the DATA-STEP comparison runs (data/dcop_*/result.pt).

The five conditions, all otherwise identical (500k ckpt, CQ500 val 0, l2si, N=50, PER=50,
uniform K=2, kappa 0):

    A adj   normalized raw-adjoint soft step      -- the 2D recipe, the incumbent
    B sart  SART row/column-normalized update     -- a DIAGONAL preconditioner
    C sart + ASD-POCS adaptive TV coupling
    D fdk   filtered-residual FDK-preconditioned  -- a SPECTRAL preconditioner
    E cg    DDS-style short CG on the normal eqs  -- a SPECTRAL preconditioner (matched adjoint)

FOUR AXES, because no single number decides this (and PSNR ranks blind recons wrong -- the SE(3)
gauge, see fm3d/reg_metric.py). The table prints, per run:

    FBP(theta_hat)  the OUTPUT: aligned PSNR / SSIM at the last step
    x_t             the CARRIED state -- what the motion estimator actually reads, via one FM
                    step. The x_t <-> FBP gap is the bottleneck diagnostic: a negative gap means
                    the carried volume has OVERTAKEN the filtered reconstruction, which flips the
                    question of which one should be the deliverable.
    theta           rot RMSE (deg) and trans_obs (mm) -- NEVER the raw translation RMSE, which is
                    dominated by the unobservable depth direction (see the project memory).
    steps-to        how many ODE steps until rot RMSE first drops below 0.5 deg: the CONVERGENCE
                    SPEED axis, which a final-value table hides.

    python scripts/cmp_dcop.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = [("A", "adj", "data/dcop_A_adj"), ("B", "sart", "data/dcop_B_sart"),
        ("C", "sart+asd", "data/dcop_C_asdpocs"), ("D", "fdk", "data/dcop_D_fdk"),
        ("E", "cg", "data/dcop_E_cg")]
ROT_TARGET = 0.5      # deg; "converged enough to stop hurting the reference image"


def first_below(hist, key, thr):
    for h in hist:
        if h[key] < thr:
            return h["step"]
    return None


def main():
    print(f"{'':2} {'dc_op':<9} | {'FBP(theta) out':>16} | {'x_t carried':>16} | "
          f"{'gap':>6} | {'rot':>5} {'obs':>5} | {'->0.5deg':>8}")
    print("-" * 88)
    best = []
    for tag, name, d in RUNS:
        p = os.path.join(d, "result.pt")
        if not os.path.isfile(p):
            print(f"{tag:2} {name:<9} | (not finished: {d})")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        h, f = r["hist"], r["final"]
        last = h[-1]
        k = first_below(h, "rot_rmse_deg", ROT_TARGET)
        print(f"{tag:2} {name:<9} | {f['psnr_aligned']:8.2f} dB {f['ssim_aligned']:.3f} | "
              f"{last['xt_psnr_aligned']:8.2f} dB {last['xt_ssim_aligned']:.3f} | "
              f"{last['psnr_aligned'] - last['xt_psnr_aligned']:+6.2f} | "
              f"{last['rot_rmse_deg']:5.2f} {last['trans_obs_mm']:5.2f} | "
              f"{(str(k) if k is not None else '--'):>8}")
        best.append((tag, name, f, last))
    if len(best) < 2:
        return
    print("\nBest OUTPUT (aligned SSIM):  " +
          max(best, key=lambda b: b[2]["ssim_aligned"])[1])
    print("Best CARRIED x_t (aligned SSIM): " +
          max(best, key=lambda b: b[3]["xt_ssim_aligned"])[1])
    print("Best theta (rot RMSE):       " + min(best, key=lambda b: b[3]["rot_rmse_deg"])[1])
    print("\nA NEGATIVE gap means x_t beat FBP(theta_hat) -- the deliverable should be revisited.")
    print("Judge the montages by eye before adopting anything (data/dcop_*/final.png).")


if __name__ == "__main__":
    main()
