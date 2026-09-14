"""Report RPE (Thies' headline motion metric) for finished posterior runs.

`run_posterior3d.py` saves `theta` and `theta_true`, so RPE can be computed after the fact for
every run we have ever done -- no re-running. Thies' gradient-based method reports a MEAN RPE of
0.61 mm at akima 5 mm / 5 deg (uncorrected ~3 mm), and that is the only motion number of his we
can be compared against; per-parameter MAEs in his Table I are for a different parameterization.

Quote `rpe_mm` (raw) against him -- he does not quotient the SE(3) gauge. `rpe_mm_gauged` is the
honest estimator error and is the one to use when comparing OUR configs to each other.

    python scripts/rpe_report.py data/runs/akima55/thies_v*/result.pt
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import motion_error, reprojection_error, zero_centre_gauge


def _row(name, e, m):
    print(f"{name:22} {e['rpe_mm']:8.3f} {e['rpe_mm_zc']:8.3f} {e['rpe_mm_gauged']:8.3f} | "
          f"{e['rpe_mm_gauged_r25']:6.2f} {e['rpe_mm_gauged_r50']:6.2f} "
          f"{e['rpe_mm_gauged_r100']:6.2f} | {m['rot_rmse_deg']:7.3f} {m['trans_obs_mm']:6.3f}",
          flush=True)


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = P_nom = None
    print(f"{'run':22} {'RPE raw':>8} {'RPE zc':>8} {'RPE gaug':>8} | {'g-r25':>6} "
          f"{'g-r50':>6} {'g-r100':>6} | {'rot deg':>7} {'obs mm':>6}")
    rows = []
    for p in paths:
        r = torch.load(p, map_location=dev, weights_only=False)
        th, tt = r["theta"].to(dev).float(), r["theta_true"].to(dev).float()
        if P_nom is None or cfg.n_views != th.shape[0]:
            cfg = ConeBeam3DConfig.thies(n_views=th.shape[0])
            P_nom = build_conebeam_orbit(cfg, device=dev)
        e = reprojection_error(th, tt, P_nom)
        # The zero-mean gauge uses NO ground truth, so this column is honestly reportable --
        # unlike `rpe_mm_gauged`, which fits the optimal G against theta_true.
        e["rpe_mm_zc"] = reprojection_error(zero_centre_gauge(th), tt, P_nom)["rpe_mm"]
        m = motion_error(th, tt, cfg=cfg)
        name = os.path.basename(os.path.dirname(p))[:22]
        rows.append((name, e, m))
        _row(name, e, m)
    if len(rows) > 1:
        n = len(rows)
        avg = lambda k: sum(e[k] for _, e, _ in rows) / n            # noqa: E731
        _row("MEAN", {k: avg(k) for k in rows[0][1]},
             {k: sum(m[k] for _, _, m in rows) / n for k in rows[0][2]})
    print("\nThe paper reports `RPE zc` (mean pose offset removed, no ground truth needed). "
          "Distances are in detector-plane millimeters.")


if __name__ == "__main__":
    main()
