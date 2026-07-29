"""Gate: `reprojection_error` (RPE) is Thies' metric, in his units.

RPE is the number this project will eventually be judged on -- Thies' gradient-based method
reports a mean of 0.61 mm, down from ~3 mm uncorrected -- so it has to be right before it is
quoted, and "right" here means two separate things:

  * the UNITS are detector millimetres, so a known object translation must appear MAGNIFIED by
    M = SDD/SOD = 1.529. Reporting object-space millimetres instead would silently make every
    number we publish 1.53x too small, and nothing else in the pipeline would notice.
  * the GAUGE split is real: an injected global pose must show up entirely in `rpe_mm` and not
    at all in `rpe_mm_gauged`.

Checks (all analytic, no reference implementation needed):
  1. theta_hat == theta_true  ->  RPE == 0.
  2. A pure z translation of dz, on points at the isocentre, gives EXACTLY M*dz -- z is the
     rotation axis, always perpendicular to the beam and parallel to detector v, so the
     magnification is exact rather than depth-dependent.
  3. A pure rotation error grows LINEARLY with shell radius: r100 / r25 == 4.
  4. An injected SE(3) gauge (theta_hat = theta_true composed with a global G) shows up in
     `rpe_mm` but is removed by `rpe_mm_gauged`.
  5. The point set is what the paper says: 300 points, 100 on each of the 25/50/100 mm shells.

    python scripts/gate_rpe.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit
from fm3d.rigid_motion import (apply_rigid_motion, reprojection_error, rigid_motion_matrices,
                               sphere_points)

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ConeBeam3DConfig.thies(n_views=360)
    P_nom = build_conebeam_orbit(cfg, device=dev)
    V = cfg.n_views
    M = cfg.SDD / cfg.SOD
    zero = torch.zeros(V, 6, device=dev)

    # ---- 5. the point set --------------------------------------------------------------
    pts = sphere_points(device=dev)
    r = pts.norm(dim=-1)
    check("300 points", pts.shape == (300, 3), f"{tuple(pts.shape)}")
    check("shells 25/50/100 mm, 100 each",
          all(torch.allclose(r[k * 100:(k + 1) * 100],
                             torch.full((100,), v, device=dev), atol=1e-3)
              for k, v in enumerate((25.0, 50.0, 100.0))),
          f"radii {sorted(set(round(float(x), 3) for x in r))}")

    # ---- 1. exact recovery -------------------------------------------------------------
    th = torch.zeros(V, 6, device=dev)
    th[:, 0] = 2.0                                   # any non-trivial true motion
    th[:, 5] = math.radians(3.0)
    e = reprojection_error(th, th, P_nom)
    check("RPE == 0 for a perfect estimate", e["rpe_mm"] < 1e-4, f"{e['rpe_mm']:.3e} mm")

    # ---- 2. units: a z shift is magnified by M -----------------------------------------
    dz = 1.0
    th_z = zero.clone()
    th_z[:, 2] = dz
    e = reprojection_error(th_z, zero, P_nom, radii=(1e-6,), n_per=100)
    check("a 1 mm z shift reads M*dz on the detector",
          abs(e["rpe_mm"] - M * dz) < 1e-3,
          f"{e['rpe_mm']:.4f} mm vs M*dz = {M * dz:.4f} (M = {M:.3f})")

    # ---- 3. a rotation error scales with radius ----------------------------------------
    th_r = zero.clone()
    th_r[:, 5] = math.radians(1.0)
    e = reprojection_error(th_r, zero, P_nom)
    ratio = e["rpe_mm_r100"] / max(e["rpe_mm_r25"], 1e-12)
    check("rotation error scales linearly with shell radius", 3.8 < ratio < 4.2,
          f"r25 {e['rpe_mm_r25']:.3f} / r50 {e['rpe_mm_r50']:.3f} / r100 "
          f"{e['rpe_mm_r100']:.3f} mm -> ratio {ratio:.3f}")

    # ---- 4. the gauge split ------------------------------------------------------------
    # theta_hat is theta_true with ONE global pose G applied on the right: T_hat_v = T_true_v G.
    # Blind motion cannot see G, so it must vanish from the gauged number.
    g6 = torch.tensor([1.5, -2.0, 0.8, 0.0, 0.0, math.radians(1.5)], device=dev)
    G = rigid_motion_matrices(g6[None])[0]
    T_true = rigid_motion_matrices(th)
    T_hat = T_true @ G
    from fm3d.rigid_motion import so3_log
    th_hat = torch.cat([T_hat[:, :3, 3], so3_log(T_hat[:, :3, :3])], dim=-1)
    e = reprojection_error(th_hat, th, P_nom)
    check("an injected gauge shows up raw", e["rpe_mm"] > 1.0, f"{e['rpe_mm']:.3f} mm")
    check("...and is removed by the gauge fit", e["rpe_mm_gauged"] < 0.02,
          f"{e['rpe_mm_gauged']:.4f} mm (share {100 * e['rpe_gauge_share']:.1f}%)")

    # sanity: the projection really is the same one the projector uses
    P_a = apply_rigid_motion(P_nom, T_true)
    check("apply_rigid_motion keeps (V,3,4)", P_a.shape == (V, 3, 4), f"{tuple(P_a.shape)}")

    print("\nGATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
