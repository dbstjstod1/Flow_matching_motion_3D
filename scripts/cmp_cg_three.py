"""Summarize the three-patient cg deliverable set (data/cg_v{0,1,2}).

Reports, per patient, BOTH volumes the loop computes -- the nominal output FDK(theta_hat) and the
carried x_t -- because with a spectral data step the carried state overtakes the output, which is
the open question about what this loop should return.

Also re-derives the THETA-AVERAGED readout offline for several K. The thetas are saved (see
run_posterior3d's `theta_hist`), so this costs nothing and answers "what would K have bought"
without re-running anything. Runs that predate the theta_hist change are reported as last-step
only rather than silently skipped.

    python scripts/cmp_cg_three.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig
from fm3d.rigid_motion import motion_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["cg_v0", "cg_v1", "cg_v2"])
    ap.add_argument("--dir", default="data")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--views", type=int, default=360)
    args = ap.parse_args()
    cfg = ConeBeam3DConfig.thies(n_views=args.views)

    print(f"{'run':8} | {'OUTPUT FDK(th)':>16} | {'carried x_t':>16} | {'gap':>6} | "
          f"{'rot':>5} {'obs':>5}")
    print("-" * 74)
    outs, xts = [], []
    rows = []
    for tag in args.runs:
        p = os.path.join(args.dir, tag, "result.pt")
        if not os.path.isfile(p):
            print(f"{tag:8} | (not finished)")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        h, f = r["hist"], r["final"]
        fx = r.get("final_xt") or {"psnr_aligned": h[-1]["xt_psnr_aligned"],
                                   "ssim_aligned": h[-1]["xt_ssim_aligned"]}
        last = h[-1]
        print(f"{tag:8} | {f['psnr_aligned']:8.2f} {f['ssim_aligned']:.3f} | "
              f"{fx['psnr_aligned']:8.2f} {fx['ssim_aligned']:.3f} | "
              f"{f['psnr_aligned'] - fx['psnr_aligned']:+6.2f} | "
              f"{last['rot_rmse_deg']:5.2f} {last['trans_obs_mm']:5.2f}")
        outs.append((f["psnr_aligned"], f["ssim_aligned"]))
        xts.append((fx["psnr_aligned"], fx["ssim_aligned"]))
        rows.append((tag, r))
    if len(outs) > 1:
        n = len(outs)
        print("-" * 74)
        print(f"{'MEAN':8} | {sum(a for a, _ in outs)/n:8.2f} {sum(b for _, b in outs)/n:.3f} | "
              f"{sum(a for a, _ in xts)/n:8.2f} {sum(b for _, b in xts)/n:.3f} | "
              f"{(sum(a for a, _ in outs) - sum(a for a, _ in xts))/n:+6.2f}")

    # ---- offline theta-averaging: what would K have bought?
    print(f"\ntheta readout, rot RMSE [deg] (and trans_obs [mm]) vs averaging window K")
    print(f"{'run':8} | " + " | ".join(f"{'K='+str(k):>16}" for k in args.ks))
    for tag, r in rows:
        th = r.get("theta_hist")
        if th is None:
            print(f"{tag:8} | (no theta_hist -- run predates the change)")
            continue
        tt = r["theta_true"]
        cells = []
        for k in args.ks:
            kk = max(1, min(k, th.shape[0]))
            me = motion_error(th[-kk:].mean(0), tt, cfg=cfg)
            cells.append(f"{me['rot_rmse_deg']:7.3f} ({me['trans_obs_mm']:.3f})")
        print(f"{tag:8} | " + " | ".join(f"{c:>16}" for c in cells))
    print("\nK>1 helps a run whose tail OSCILLATES (landing lottery) and hurts one still "
          "DESCENDING\n(it averages in stale thetas) -- check the two regimes before picking K.")


if __name__ == "__main__":
    main()
