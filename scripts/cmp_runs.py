"""One table for every finished posterior run: the four cells, theta, and RPE.

THE FOUR CELLS ARE NOT INTERCHANGEABLE and a config that lifts one while dropping another is not
a win (user's rule, 2026-07-26). They are laid out here in the order they matter:

    x_t vs GT      **THE MAIN DELIVERABLE.** x_t is a prior+data reconstruction, not an FDK, so
                   the ground truth is its natural reference -- and it beats the static FDK's own
                   fidelity to GT (0.8256), which is why scoring it against the static FDK
                   penalises it for removing artefacts the reference still has.
    OUT vs sFDK    **THE EVIDENCE CHANNEL.** FDK(theta_hat) against the motion-free FDK: both are
                   FDKs, so this is the volume that is comparable in KIND to Thies' 0.94, and it
                   is the honest readout of whether the estimated Pmat is right. Its ceiling with
                   PERFECT theta is 0.7935 / 36.27 dB (val 0) -- read every number against that,
                   not against 1.0.
    x_t vs sFDK / OUT vs GT   the two off-diagonal cells, reported so a trade cannot hide.

`RPE zc` is the headline motion number: RPE after moving theta onto the zero-mean gauge, which
uses no ground truth and matches the convention the simulator and Thies both use. Thies: 0.61 mm.

    python scripts/cmp_runs.py data/runs/akima55/*/result.pt
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import motion_error, reprojection_error, zero_centre_gauge

A = "ssim_aligned"
P = "psnr_aligned"


def main():
    paths = [p for p in sys.argv[1:] if os.path.isfile(p)]
    if not paths:
        raise SystemExit(__doc__)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P_nom = cfg = None
    print(f"{'run':20} | {'x_t vs GT (MAIN)':>17} | {'OUT vs sFDK (EVID)':>18} | "
          f"{'x_t vs sFDK':>13} | {'OUT vs GT':>13} | {'rot':>6} {'RPEzc':>6}")
    print("-" * 108)
    for p in sorted(paths):
        r = torch.load(p, map_location=dev, weights_only=False)
        if any(k not in r for k in ("final", "final_s", "final_xt", "final_xt_s")):
            print(f"{os.path.basename(os.path.dirname(p))[:20]:20} | (incomplete)")
            continue
        th, tt = r["theta"].to(dev).float(), r["theta_true"].to(dev).float()
        if P_nom is None:
            cfg = ConeBeam3DConfig.thies(n_views=th.shape[0])
            P_nom = build_conebeam_orbit(cfg, device=dev)
        rot = motion_error(th, tt, cfg=cfg)["rot_rmse_deg"]
        zc = reprojection_error(zero_centre_gauge(th), tt, P_nom)["rpe_mm"]
        c = [r["final_xt"], r["final_s"], r["final_xt_s"], r["final"]]
        cells = " | ".join(f"{m[P]:6.2f}/{m[A]:.4f}" for m in c)
        print(f"{os.path.basename(os.path.dirname(p))[:20]:20} | {cells} | {rot:6.3f} {zc:6.3f}",
              flush=True)
    print("\nreference points (val 0): oracle-theta ceiling  x_t 40.45/0.9889, OUT-vs-sFDK "
          "36.27/0.7935 | static FDK vs GT 33.36/0.8256 | Thies RPE 0.61 mm")


if __name__ == "__main__":
    main()
