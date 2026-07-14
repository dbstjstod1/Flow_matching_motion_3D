"""Gate the per-view ANGULAR WEIGHT. Phantom only -- no CQ500, no checkpoint. ~2 min.

FDK ends with one scalar, `angle_span / V`: "the views are equiangular". Rigid patient motion
about the GANTRY AXIS breaks exactly that and nothing else -- rotating the object about z maps
the source circle onto ITSELF, so the orbit stays a circle and the ramp stays along u; the views
merely stop being evenly spaced. The uniform weight is then the wrong Riemann sum, and it is
worth -2.0 dB at 5 deg on real CQ500 heads.

`geometry_3d.view_angular_weights` reads each view's true angular share out of Pmat. It is a
VORONOI PARTITION OF THE CIRCLE, not a difference along the view index, and that distinction is
the whole gate: real head motion contains STEPS (one view rotates 5 deg while the gantry advances
1 deg, so the effective angle jumps BACKWARDS by 9 deg), and a central difference hands that view
a 4x weight and makes the reconstruction WORSE. Sorting the views around the circle cannot do
that -- every share is non-negative and they sum to 2*pi whatever the motion does.

  [1] source_positions recovers the orbit from P alone (nominal: radius == SOD, angles equiangular)
  [2] the weights are a partition: >= 0, sum == 2*pi exactly, for arbitrary motion incl. steps
  [3] NO-OP on the nominal orbit -- weights uniform, and the reconstruction is unchanged
  [4] a GLOBAL rigid motion keeps the orbit circular: oracle FDK == static FDK (the claim that
      "P absorbs 6-DoF exactly")
  [5] rz-only motion: the uniform weight loses dB, the Voronoi weight gets them ALL back
  [6] a STEP in rz: the naive central-difference weight makes it WORSE; Voronoi does not

    python scripts/gate_angle_weight.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.filters import calibrate_scale
from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              measured_region_mask, source_positions, view_angular_weights)
from fm3d.phantom import head_phantom
from fm3d.projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched
from fm3d.rigid_motion import params_to_Pmot

FAIL = []


def check(i, name, ok, detail=""):
    print(f"[{i}] {name:<56s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAIL.append(name)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V = 360                    # the real view count: at 1 deg/view a 5 deg motion step can
    cfg = ConeBeam3DConfig.thies(n_views=V, det_bin=2)      # reverse the effective view angle
    P_nom = build_conebeam_orbit(cfg, device=dev)
    uc, vc = detector_coords_3d(cfg, device=dev)
    shape, sp = (96, 128, 128), (1.5, 1.5, 1.5)
    uni = cfg.angle_span / V

    vol = head_phantom(shape, sp, device=dev)[None, None]
    meas = measured_region_mask(shape, sp, cfg, device=dev)

    def project(v, P):
        return forward_project_3d_batched(v, P, uc, vc, dx=sp[2], dy=sp[1], dz=sp[0],
                                          n_samples=384, view_chunk=8)

    def fdk(y, P, w=None, scale=None):
        return fdk_conebeam_3d_batched(y, P, uc, vc, cfg, D=shape[0], H=shape[1], W=shape[2],
                                       dx=sp[2], dy=sp[1], dz=sp[0], scale=scale,
                                       view_weight=w, view_chunk=8)[0]

    with torch.no_grad():
        y_nom = project(vol, P_nom[None])
        scale = calibrate_scale(fdk(y_nom, P_nom[None], scale=1.0), vol[0, 0], meas)

    def psnr(a):
        e = (a - vol[0, 0])[meas]
        rng = float(vol[0, 0][meas].max() - vol[0, 0][meas].min())
        return float(20 * np.log10(rng / (e.pow(2).mean().sqrt().item() + 1e-12)))

    # ---- [1] the orbit, recovered from P alone ------------------------------------------
    S = source_positions(P_nom)
    r = S[:, :2].norm(dim=-1)
    check(1, "source_positions: |S| == SOD on the nominal orbit",
          abs(float(r.mean()) - cfg.SOD) < 1e-2 and float(r.std()) < 1e-2,
          f"{float(r.mean()):.3f} +- {float(r.std()):.1e} mm (SOD {cfg.SOD})")
    check(1, "source_positions: the source is in the z=0 plane",
          float(S[:, 2].abs().max()) < 1e-3, f"max|Sz| = {float(S[:, 2].abs().max()):.1e} mm")

    # ---- [2] the weights are a partition of the circle -----------------------------------
    torch.manual_seed(0)
    for tag, th in [
        ("smooth", torch.stack([5 * torch.sin(2 * math.pi * 1.5 * torch.arange(V, device=dev) / V)
                                if k == 5 else torch.zeros(V, device=dev) for k in range(6)], -1)),
        ("with a 5 deg STEP", torch.stack(
            [torch.where(torch.arange(V, device=dev) > V // 2,
                         torch.full((V,), 5 * math.pi / 180, device=dev),
                         torch.zeros(V, device=dev)) if k == 5 else torch.zeros(V, device=dev)
             for k in range(6)], -1)),
        ("random 6-DoF", torch.randn(V, 6, device=dev) * torch.tensor(
            [5, 5, 5, 0.09, 0.09, 0.09], device=dev))]:
        w = view_angular_weights(params_to_Pmot(th, P_nom))
        ok = bool((w >= 0).all()) and abs(float(w.sum()) / (2 * math.pi) - 1.0) < 1e-5
        check(2, f"partition of the circle ({tag})", ok,
              f"min {float(w.min()) / uni:.3f}x  max {float(w.max()) / uni:.3f}x  "
              f"sum/2pi {float(w.sum()) / (2 * math.pi):.7f}")

    # ---- [3] NO-OP on the nominal orbit ---------------------------------------------------
    w_nom = view_angular_weights(P_nom)
    check(3, "nominal orbit -> the weights come back uniform",
          float((w_nom / uni - 1).abs().max()) < 1e-4,
          f"max dev {float((w_nom / uni - 1).abs().max()):.2e}")
    with torch.no_grad():
        a = fdk(y_nom, P_nom[None], scale=scale)
        b = fdk(y_nom, P_nom[None], w=w_nom[None], scale=scale)
    rel = float((a - b).abs().max() / a.abs().max())
    check(3, "... so the static reconstruction is unchanged", rel < 1e-4, f"rel {rel:.1e}")
    p_static = psnr(a)
    print(f"    static FDK = {p_static:.2f} dB")

    # ---- [4] a GLOBAL rigid motion keeps the orbit a circle --------------------------------
    th = torch.zeros(V, 6, device=dev)
    th[:, :3] = torch.tensor([5.0, -3.0, 2.0], device=dev)
    th[:, 5] = 5 * math.pi / 180
    P = params_to_Pmot(th, P_nom)[None]
    with torch.no_grad():
        p = psnr(fdk(project(vol, P), P, scale=scale))
    check(4, "GLOBAL rigid (6 mm + 5 deg): oracle FDK == static", abs(p - p_static) < 0.35,
          f"{p:.2f} dB vs static {p_static:.2f} ({p - p_static:+.2f})")

    # ---- [5] rz-only: the uniform weight loses dB, the Voronoi weight recovers them --------
    th = torch.zeros(V, 6, device=dev)
    th[:, 5] = (5 * math.pi / 180) * torch.sin(2 * math.pi * 1.5 * torch.arange(V, device=dev) / V)
    P = params_to_Pmot(th, P_nom)[None]
    with torch.no_grad():
        y = project(vol, P)
        p_uni = psnr(fdk(y, P, scale=scale))
        p_vor = psnr(fdk(y, P, w=view_angular_weights(P), scale=scale))
    check(5, "rz 5 deg: the UNIFORM weight demonstrably loses quality", p_static - p_uni > 0.5,
          f"uniform {p_uni:.2f} dB ({p_uni - p_static:+.2f} vs static)")
    check(5, "rz 5 deg: the VORONOI weight gets it back", abs(p_vor - p_static) < 0.35,
          f"voronoi {p_vor:.2f} dB ({p_vor - p_static:+.2f} vs static)  "
          f"[recovered {p_vor - p_uni:+.2f}]")

    # ---- [6] a STEP: the naive central difference makes it WORSE ---------------------------
    # The REAL profile that broke it, not a hand-made one: `mixed` seed 2 rotates the patient
    # 5 deg between two consecutive views while the gantry advances 1 deg, so the effective view
    # angle jumps BACKWARDS by 9 deg. (A gentler step would not reproduce the failure -- an
    # earlier version of this gate used a single 5 deg jump at V=180, i.e. 2 deg/view, where the
    # central difference never exceeds 1.0x and is harmless. The gate has to hurt to be a gate.)
    from fm3d.rigid_motion import make_motion
    th = torch.zeros(V, 6, device=dev)
    th[:, 5] = make_motion("mixed", V, device=dev, seed=2, trans_mm=(5.,) * 3,
                           rot_deg=(5.,) * 3)[:, 5]
    P = params_to_Pmot(th, P_nom)[None]
    S = source_positions(P[0])
    beta = torch.atan2(S[:, 1], S[:, 0])
    d = beta[1:] - beta[:-1]
    d = torch.atan2(torch.sin(d), torch.cos(d))
    w_cd = torch.empty_like(beta)
    w_cd[1:-1] = 0.5 * (d[:-1] + d[1:])
    w_cd[0], w_cd[-1] = d[0], d[-1]
    w_cd = w_cd.abs()
    with torch.no_grad():
        y = project(vol, P)
        p_uni = psnr(fdk(y, P, scale=scale))
        p_cd = psnr(fdk(y, P, w=w_cd[None], scale=scale))
        p_vor = psnr(fdk(y, P, w=view_angular_weights(P), scale=scale))
    check(6, "STEP: the naive central-difference weight is HARMFUL", p_cd < p_uni,
          f"uniform {p_uni:.2f} -> central-diff {p_cd:.2f} ({p_cd - p_uni:+.2f})  "
          f"[max weight {float(w_cd.max()) / uni:.1f}x]")
    check(6, "STEP: the VORONOI weight is not", p_vor >= p_uni - 0.05,
          f"uniform {p_uni:.2f} -> voronoi {p_vor:.2f} ({p_vor - p_uni:+.2f})")

    n = len(FAIL)
    print(f"\n{'ALL PASS' if n == 0 else f'{n} FAILURE(S): ' + ', '.join(FAIL)}")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
