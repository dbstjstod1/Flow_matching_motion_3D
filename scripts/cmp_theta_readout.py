"""Stage 0: what would a different theta READOUT have bought? Offline, on finished runs.

Costs nothing to ask -- run_posterior3d saves the whole `theta_hist` (N,V,6), so every readout
rule can be replayed against a run that has already happened. Three rules are compared:

    last        theta at the final step -- what the loop returns when --theta_avg 1
    mean K      the 2D project's winning readout: mean of the last K thetas. It cancels a
                period-2 limit cycle, so it wins when the tail OSCILLATES and LOSES when the
                tail is still descending (it averages in stale, worse thetas).
    oracle      the single best step, chosen with knowledge of theta_true. NOT a usable rule --
                it is the ceiling a blind best-step selector could reach, and the gap between
                `last` and `oracle` is the size of the landing lottery.

Reported per run so the two regimes stay visible; a mean over runs would hide exactly the thing
that decides whether K>1 is safe.

    python scripts/cmp_theta_readout.py --runs fair_cg fair_adj fair_sart
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
    ap.add_argument("--runs", nargs="+", default=["fair_cg", "fair_adj", "fair_sart"])
    ap.add_argument("--dir", default="data")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 8])
    ap.add_argument("--views", type=int, default=360)
    args = ap.parse_args()
    cfg = ConeBeam3DConfig.thies(n_views=args.views)

    hdr = " | ".join(f"{'K=' + str(k):>13}" for k in args.ks)
    print(f"{'run':10} | {hdr} | {'oracle step':>13} | tail")
    print("-" * (14 + 16 * len(args.ks) + 34))
    for tag in args.runs:
        p = os.path.join(args.dir, tag, "result.pt")
        if not os.path.isfile(p):
            print(f"{tag:10} | (not finished)")
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        th, tt = r.get("theta_hist"), r["theta_true"]
        if th is None:
            print(f"{tag:10} | (no theta_hist -- run predates the change)")
            continue
        cells = []
        for k in args.ks:
            kk = max(1, min(k, th.shape[0]))
            me = motion_error(th[-kk:].mean(0), tt, cfg=cfg)
            cells.append(f"{me['rot_rmse_deg']:6.3f} deg")
        # the ceiling a perfect best-step selector would reach
        per_step = [motion_error(th[i], tt, cfg=cfg)["rot_rmse_deg"] for i in range(th.shape[0])]
        bi = min(range(len(per_step)), key=lambda i: per_step[i])
        # is the tail oscillating or still descending? count sign flips of successive changes
        tail = per_step[-12:]
        d = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        flips = sum(1 for i in range(len(d) - 1) if d[i] * d[i + 1] < 0)
        regime = "OSCILLATING" if flips >= 6 else ("descending" if sum(d) < -0.01 else "flat")
        print(f"{tag:10} | " + " | ".join(f"{c:>13}" for c in cells) +
              f" | {per_step[bi]:6.3f}@{bi:<3d} | {flips}/{len(d)-1} {regime}")
    print("\nK>1 is a VARIANCE fix (kills the period-2 landing lottery), not a convergence fix.\n"
          "Pick K per regime: averaging a still-DESCENDING tail makes it worse.")


if __name__ == "__main__":
    main()
