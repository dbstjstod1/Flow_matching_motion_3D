"""Gate the SF (separable-footprint) projector pair with geometry gradients (fm3d/triton_sf).

What it must prove, in order of what would silently break the loop if wrong:

  T1  forward parity     vs the GRIDSAMPLE ray-march reference (the retired Triton ray march
                         routes to SF now, so gridsample is the independent check) on a MOVED
                         geometry.
                         Different discretizations (footprint vs sampled march), so the bar is
                         the LEAP-vs-raymarch level (~2e-3), not machine precision.
  T2  matched transpose  <A f, s> == <f, A^T s> to float-accumulation precision: forward and
                         transpose share the SAME weight code, so unlike T1 this one has no
                         model-difference excuse.
  T3  theta gradient     d loss/d theta against the ray-march reference's EXACT gradient
                         (autograd through `reference_project_3d_batched`). SF drops the footprint-width
                         and path-length derivative terms (second order), so direction (cosine)
                         is the claim, per DoF block.
  T4  speed              24-view estimator shape (fwd+theta-bwd) and 360-view CG shape.

    python scripts/gate_sf_projector.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from fm3d.phantom import head_phantom
from fm3d.projector_3d import reference_project_3d_batched
from fm3d.rigid_motion import params_to_Pmot
from fm3d.triton_sf import sf_backproject, sf_project_3d_batched

DEV = "cuda"
CFG = ConeBeam3DConfig(det_bin=4, n_views=64)
SHAPE = (96, 128, 128)
SPACING = (1.5, 1.5, 1.5)

_fails: list[str] = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")
    if not ok:
        _fails.append(name)


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def main():
    dz, dy, dx = SPACING
    D, H, W = SHAPE
    P_nom = build_conebeam_orbit(CFG, device=DEV)
    u, v = detector_coords_3d(CFG, device=DEV)
    vol = head_phantom(SHAPE, SPACING, device=DEV)
    rkw = dict(dx=dx, dy=dy, dz=dz, n_samples=256, view_chunk=8, row_chunk=64)
    skw = dict(dx=dx, dy=dy, dz=dz)
    du = float(u[1] - u[0])
    dv = float(v[1] - v[0])

    torch.manual_seed(0)
    th_true = torch.randn(CFG.n_views, 6, device=DEV) * torch.tensor(
        [2., 2., 2., .02, .02, .02], device=DEV)
    P = params_to_Pmot(0.5 * th_true, P_nom)[None]               # moved geometry throughout

    # ---- T1 forward parity (model difference allowed, must be small)
    print("T1  forward parity vs the gridsample reference (moved geometry)")
    with torch.no_grad():
        g_ray = reference_project_3d_batched(vol[None, None], P, u, v, **rkw)
        g_sf = sf_project_3d_batched(vol[None, None], P, u, v, **skw)
    rel = float((g_sf - g_ray).norm() / g_ray.norm())
    # The bar is the SF MODEL CLASS, not machine precision: attribution (2026-07-28) put the
    # ray-march's own quadrature at 4e-4 (n_samples 256 vs 768), so this residual is the
    # separable-footprint approximation itself -- the same class as LEAP's SF, which measured
    # 2.4e-3 against our ray-march on the smooth deploy sphere (obliques run ~2-3x the
    # axis-aligned views; this gate's coarse 1.55 mm panel and bone edges sit at the high end).
    check("forward parity", rel < 1.5e-2, f"rel = {rel:.2e} (SF-class; raymarch quadrature 4e-4)")

    # ---- T2 the matched pair
    print("\nT2  adjoint identity (same weight code both directions)")
    s = torch.randn_like(g_ray)
    with torch.no_grad():
        ATs = sf_backproject(s, P, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
                             u0=float(u[0] + u[-1]) * 0.5, v_off=float(v[0] + v[-1]) * 0.5)
    lhs = float((g_sf * s).sum())
    rhs = float((vol[None] * ATs).sum())
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-20)
    check("matched transpose", rel < 1e-4,
          f"<Af,s> = {lhs:.6e}  <f,ATs> = {rhs:.6e}  rel = {rel:.2e}")

    # ---- T3 the geometry gradient
    print("\nT3  d(loss)/d(theta) vs the gridsample-autograd exact gradient")
    with torch.no_grad():
        y = reference_project_3d_batched(vol[None, None],
                                         params_to_Pmot(th_true, P_nom)[None], u, v, **rkw)

    def theta_grad(op):
        th = (0.5 * th_true).detach().clone().requires_grad_(True)
        Pm = params_to_Pmot(th, P_nom)[None]
        pred = op(Pm)
        (0.5 * ((pred - y) ** 2).mean()).backward()
        return th.grad.clone()

    g_ref = theta_grad(lambda Pm: reference_project_3d_batched(
        vol[None, None], Pm, u, v, **rkw))
    g_new = theta_grad(lambda Pm: sf_project_3d_batched(vol[None, None], Pm, u, v, **skw))
    for name, sl in [("translation", slice(0, 3)), ("rotation", slice(3, 6))]:
        a, b = g_ref[:, sl], g_new[:, sl]
        c = cos(a, b)
        r = float(b.norm() / a.norm())
        check(f"{name} block", c > 0.98,
              f"cos = {c:.4f}  |sf|/|ray| = {r:.3f}")

    # ---- T4 speed at deploy scale
    print("\nT4  speed (Thies panel, 256^3)")
    tcfg = ConeBeam3DConfig.thies(n_views=360)
    tP_nom = build_conebeam_orbit(tcfg, device=DEV)
    tu, tv = detector_coords_3d(tcfg, device=DEV)
    f = torch.rand(1, 1, 256, 256, 256, device=DEV)
    tdu, tdv = float(tu[1] - tu[0]), float(tv[1] - tv[0])
    tkw = dict(dx=1.0, dy=1.0, dz=1.0)

    def t(fn, reps=3):
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / reps

    with torch.no_grad():
        tf = t(lambda: sf_project_3d_batched(f, tP_nom[None], tu, tv, **tkw))
        gg = sf_project_3d_batched(f, tP_nom[None], tu, tv, **tkw)
        tb = t(lambda: sf_backproject(gg, tP_nom[None], D=256, H=256, W=256,
                                      dx=1.0, dy=1.0, dz=1.0, du=tdu, dv=tdv))
    print(f"      360-view: SF fwd {tf*1e3:7.1f} ms   SF^T {tb*1e3:7.1f} ms   "
          f"(retired anchors: ray fwd 662 / scatter-adj 3923 / unmatched-bp 112)")

    # estimator shape: 24 views, fwd + theta backward
    idx = torch.arange(0, 360, 15, device=DEV)[:24]
    eP_nom = tP_nom[idx]
    y24 = torch.randn(1, 24, len(tv), len(tu), device=DEV)

    def est_iter(op):
        th = torch.zeros(24, 6, device=DEV, requires_grad=True)
        def run():
            th.grad = None
            pred = op(params_to_Pmot(th, eP_nom)[None])
            (0.5 * ((pred - y24) ** 2).mean()).backward()
        return t(run)

    te_sf = est_iter(lambda Pm: sf_project_3d_batched(f, Pm, tu, tv, **tkw))
    # historical anchors (retired ray-march, same GPU, 2026-07-28): fwd 662 ms,
    # matched scatter adjoint 3923 ms, est iter (fwd+theta-bwd) 108 ms.
    print(f"      24-view est iter (fwd+theta-bwd): SF {te_sf*1e3:7.1f} ms   "
          f"(retired ray-march anchor: 108 ms)")
    # THIS IS A WALL-CLOCK BUDGET, SO RUN IT ON AN IDLE GPU. Measured 115.0 ms on a free A6000
    # and 276 ms on the same card while a training job held it at 93% -- if this check fails,
    # confirm the GPU is quiet before believing it.
    # The 150 ms figure predates `_footprint_window`, which now DERIVES FP/FPV from the geometry
    # and returns 6/5 here where the old hardcoded default was 6/4. That default was TRUNCATING:
    # 5.6e-4 relative in the forward, and the dP gradient only cos 0.99955 / 3.2e-2 rel-L2 against
    # an untruncated reference (at 6/5 it is cos 1.000000000, i.e. exact). Correctness costs +11%
    # here (103.5 -> 115.0 ms), which still fits the original budget, so the threshold stands.
    # Do NOT "fix" a regression here by shrinking the window: that trades a quantitative bias in
    # y / the bridge target / the CG solution for a speed number.
    check("estimator iteration budget", te_sf < 0.150, f"{te_sf*1e3:.1f} ms < 150 ms")

    print("\n" + "=" * 70)
    print(f"FAILED: {', '.join(_fails)}" if _fails else "all gates passed")
    print("=" * 70)
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
