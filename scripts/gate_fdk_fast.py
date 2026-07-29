"""Gate the DEPLOYED FDK backprojection: LEAP's modular VD kernel under our FDK algorithm.

Since 2026-07-30 the static FDK's distance-weighted backprojection runs through
`leap_projector.leap_fdk_backproject` (the operator-unification decision: our
cosine/Wang/Ohnesorge/ramp/Voronoi algorithm, LEAP's backprojector under it, with LEAP's
geometric ray weight folded back to our exact 1/w^2 convention -- the algebra is in that
function's docstring). The torch loop survives as the reference; FM3D_FDK_LEAP=0 routes to it.

WHAT IS CLAIMED:
  * the LEAP-backed FDK computes the SAME reconstruction as the torch reference up to the
    KNOWN model gap: tex3D bilinear (9-bit lerp fractions) vs grid_sample, and the folded
    1/(sdd*dist) weight being interpolated WITH the data instead of applied per voxel
    (second-order in the detector cell). Measured at adoption: rel-to-peak max ~1e-3-level,
    3.1e-3 in norm at 256^3. An actual semantic error (offset, mask edge, weight power)
    shows up at O(1e-1..1) and fails loudly.
  * determinism and batch-consistency of the deployed path.

G1  rigid-motion P + Voronoi weights, LEAP vs torch:    rel max < 5e-2 (panel-edge model, see G1 note) and rel RMS < 1e-3
G2  determinism:                                        two runs bit-identical (max|d| == 0)
G3a identical rows in one batch:                        bit-identical rows
G3b batched B=2 vs 2x B=1:                              rel max < 1e-6
G4  head-phantom round-trip FDK(A(phantom)):            backends agree, rel max < 1e-2
G5  full CQ500/Thies-scale timing (360 views, 500x700 panel, 256^3 @ 1 mm): report only.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              view_angular_weights)
from fm3d.phantom import head_phantom
from fm3d.projector_3d import fdk_conebeam_3d_batched, forward_project_3d_batched
from fm3d.rigid_motion import params_to_Pmot, random_motion

dev = "cuda"
FAIL = []


def gate(name, ok, detail):
    print(f"[gate] {name}: {detail} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        FAIL.append(name)


def fdk(sino, P, u, v, cfg, leap: bool, **kw):
    os.environ["FM3D_FDK_LEAP"] = "1" if leap else "0"
    try:
        return fdk_conebeam_3d_batched(sino, P, u, v, cfg, **kw)
    finally:
        os.environ.pop("FM3D_FDK_LEAP", None)


def smooth_sino(B, V, nv, nu, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.rand((B, V, nv, nu), generator=g).to(dev)
    s = F.avg_pool2d(s.view(B * V, 1, nv, nu), 9, stride=1, padding=4)
    s = F.avg_pool2d(s, 9, stride=1, padding=4).view(B, V, nv, nu)
    return s.contiguous()


def rel(a, b):
    d = (a - b).abs()
    ref = b.abs().amax().clamp_min(1e-30)
    return float(d.amax() / ref), float(d.pow(2).mean().sqrt() / ref)


def main():
    torch.manual_seed(0)

    # ---- small config: Thies geometry, 60 views, 96^3 @ 2 mm -------------------------------
    cfg = ConeBeam3DConfig.thies(n_views=60)
    P_nom = build_conebeam_orbit(cfg, device=dev)
    u, v = detector_coords_3d(cfg, device=dev)
    D = H = W = 96
    vox = dict(D=D, H=H, W=W, dx=2.0, dy=2.0, dz=2.0)

    theta = random_motion(cfg.n_views, trans_mm=10.0, rot_deg=10.0, device=dev)
    P = params_to_Pmot(theta, P_nom)[None]                       # (1, V, 3, 4)
    vw = view_angular_weights(P)
    sino = smooth_sino(1, cfg.n_views, cfg.nv, cfg.nu)

    # G1: rigid-motion geometry + Voronoi weights, LEAP vs torch. The max bar is set by the
    # PANEL-EDGE handling difference (our hard half_u mask vs LEAP's tex border, which fades
    # over the last half pixel): this synthetic sinogram has O(1) values AT the edge, so a
    # few edge voxels read ~2.7e-2 of peak (measured) while the volume RMS sits at 3.6e-4.
    # A physical sinogram decays at the panel edge -- G4's head phantom reads max 5.4e-4.
    r_t = fdk(sino, P, u, v, cfg, True, view_weight=vw, **vox)
    r_c = fdk(sino, P, u, v, cfg, False, view_weight=vw, **vox)
    m, r = rel(r_t, r_c)
    gate("G1 leap-vs-torch", m < 5e-2 and r < 1e-3, f"rel max {m:.3e} rms {r:.3e}")

    # G2: determinism
    r_t2 = fdk(sino, P, u, v, cfg, True, view_weight=vw, **vox)
    d = (r_t - r_t2).abs().amax()
    gate("G2 determinism", float(d) == 0.0, f"max|d| {float(d):.3e}")

    # G3a: identical rows in one batch are bit-identical
    sino2 = torch.cat([sino, sino], 0)
    P2 = torch.cat([P, P], 0)
    rb = fdk(sino2, P2, u, v, cfg, True, view_weight=torch.cat([vw, vw], 0), **vox)
    d = (rb[0] - rb[1]).abs().amax()
    gate("G3a same-rows", float(d) == 0.0, f"max|d| {float(d):.3e}")

    # G3b: batched vs single calls (different rows)
    sino_b = smooth_sino(2, cfg.n_views, cfg.nv, cfg.nu, seed=1)
    theta_b = random_motion(cfg.n_views, trans_mm=10.0, rot_deg=10.0, device=dev)
    P_b = torch.stack([params_to_Pmot(theta, P_nom), params_to_Pmot(theta_b, P_nom)], 0)
    vw_b = view_angular_weights(P_b)
    rb = fdk(sino_b, P_b, u, v, cfg, True, view_weight=vw_b, **vox)
    r0 = fdk(sino_b[:1], P_b[:1], u, v, cfg, True, view_weight=vw_b[:1], **vox)
    r1 = fdk(sino_b[1:], P_b[1:], u, v, cfg, True, view_weight=vw_b[1:], **vox)
    m0, _ = rel(rb[0], r0[0])
    m1, _ = rel(rb[1], r1[0])
    gate("G3b batched", max(m0, m1) < 1e-6, f"rel max {max(m0, m1):.3e}")

    # G4: physical round-trip, both backends
    vol = head_phantom((D, H, W), (2.0, 2.0, 2.0), device=dev)[None, None]
    y = forward_project_3d_batched(vol, P_nom[None], u, v, dx=2.0, dy=2.0, dz=2.0)
    q_t = fdk(y, P_nom[None], u, v, cfg, True, **vox)
    q_c = fdk(y, P_nom[None], u, v, cfg, False, **vox)
    m, r = rel(q_t, q_c)
    gate("G4 round-trip", m < 1e-2, f"rel max {m:.3e} rms {r:.3e}")

    # G5: full-scale timing (the training/inference config: 360 views, 256^3 @ 1 mm)
    cfgF = ConeBeam3DConfig.thies(n_views=360)
    P_f = build_conebeam_orbit(cfgF, device=dev)
    uF, vF = detector_coords_3d(cfgF, device=dev)
    thF = random_motion(cfgF.n_views, trans_mm=10.0, rot_deg=10.0, device=dev)
    PF = params_to_Pmot(thF, P_f)[None]
    vwF = view_angular_weights(PF)
    sF = smooth_sino(1, cfgF.n_views, cfgF.nv, cfgF.nu, seed=2)
    voxF = dict(D=256, H=256, W=256, dx=1.0, dy=1.0, dz=1.0)

    def clock(fn, n=3):
        fn()                                                     # warm-up / autotune
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t_tri = clock(lambda: fdk(sF, PF, uF, vF, cfgF, True, view_weight=vwF, **voxF))
    t_tor = clock(lambda: fdk(sF, PF, uF, vF, cfgF, False, view_weight=vwF, **voxF), n=1)
    gate("G5 timing", True,
         f"full FDK 256^3/360v: torch {t_tor:.2f}s vs leap {t_tri:.2f}s "
         f"({t_tor / t_tri:.1f}x)")

    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
