"""Gate the ANALYTIC bridge tangent (2026-07-18): d/ds FDK(y, P(s*theta)) in one fused pass.

The claim chain, each link gated separately so a failure names its culprit:

T1  bridge_P_and_dP (rigid_motion): dP/ds = P_nom @ [[skew(w)R, t],[0,0]]
      vs a float64 central difference of params_to_Pmot(s*theta):        rel max < 1e-6
T2  view_angular_weights_dot (geometry_3d): the Voronoi weight and its s-derivative
      value vs view_angular_weights:                                     rel max < 1e-9
      derivative vs float64 central difference:                          rel max < 1e-6
T3  the WHOLE analytic chain (dP -> dw -> tangent backprojection), float64, against an
    INDEPENDENT derivative: torch.autograd.functional.jvp through a functional rebuild of
    P(s), w(s) and the torch tangent reference's VALUE path. Autograd shares the frozen-cell
    a.e. semantics (floor/argsort carry no gradient), so agreement here certifies the
    hand-rolled tangent math to float64 precision:                       rel max < 1e-9
T3b coarse central difference (delta=0.02, the old bridge_pair step) vs analytic, float64:
    confirms the a.e. convention matches the true function -- the bilinear interpolant is C0,
    so the FD carries O(delta) kink error at cell-crossing voxels; this is the ~1e-3
    disagreement the old velocity target LIVED WITH:                     rel rms < 2e-2, report
T4  value-path factoring: fdk_conebeam_3d_tangent's x output vs fdk_conebeam_3d_batched
    (weights folded into the sinogram BEFORE the filter). This is the "positive per-view
    scalar commutes with cosine/Wang/ohnesorge-clamp/ramp" claim:        rel max < 5e-4
T5  the fp32 Triton fused kernel vs the fp32 torch reference, same filtered sinogram:
      value rel max < 1e-4;  tangent rel rms < 1e-3 AND outlier fraction (rel > 1e-3) < 1e-3.
    The tangent's MAX is deliberately NOT gated: the bilinear derivative g_u is DISCONTINUOUS
    at detector-cell edges, so a voxel whose ju lands within an ulp of an integer can be put
    on opposite sides by the two implementations' fp32 op orders (einsum vs in-register dot),
    and its derivative then legitimately jumps by O(1) of the local sinogram gradient.
    Measured: 54 of 262144 voxels (0.02%) above 1e-4, everything else at rounding level --
    the a.e.-derivative's measure-zero set made visible by fp32. (The VALUE is C0, which is
    why ITS max is tight.) The old FD target carried 1.8e-2 rel rms error at EVERY voxel.
T6  full CQ500/Thies scale (360 views, 500x700, 256^3): analytic vs Triton-FD(0.02)
    consistency (rel rms < 5e-2, report) and TIMING: old bridge_pair cost (3 torch FDKs)
    vs 3 Triton FDKs vs ONE fused tangent pass.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import (ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d,
                              view_angular_weights, view_angular_weights_dot)
from fm3d.projector_3d import (_backproject_tangent_torch, fdk_conebeam_3d_batched,
                               fdk_conebeam_3d_tangent)
from fm3d.rigid_motion import bridge_P_and_dP, params_to_Pmot, random_motion, skew, so3_exp

dev = "cuda"
FAIL = []


def gate(name, ok, detail):
    print(f"[gate] {name}: {detail} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        FAIL.append(name)


def rel(a, b):
    d = (a - b).abs()
    ref = b.abs().amax().clamp_min(1e-30)
    return float(d.amax() / ref), float(d.pow(2).mean().sqrt() / ref)


def smooth_sino(B, V, nv, nu, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.rand((B, V, nv, nu), generator=g).to(dev)
    s = F.avg_pool2d(s.view(B * V, 1, nv, nu), 9, stride=1, padding=4)
    s = F.avg_pool2d(s, 9, stride=1, padding=4).view(B, V, nv, nu)
    return s.to(dtype).contiguous()


# ---- functional (autograd-clean) rebuilds for T3 --------------------------------------------
def P_of_s(s, theta, P_nom):
    """params_to_Pmot(s*theta, P_nom) built without in-place ops, differentiable in s."""
    t = theta[..., :3]
    w = theta[..., 3:]
    R = so3_exp(s * w)                                           # (V, 3, 3)
    top = torch.cat([R, (s * t)[..., None]], dim=-1)             # (V, 3, 4)
    bot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=theta.device,
                       dtype=theta.dtype).expand(theta.shape[0], 1, 4)
    return P_nom @ torch.cat([top, bot], dim=-2)


def w_of_P(P):
    """view_angular_weights, rebuilt with out-of-place scatter (differentiable in P)."""
    import math
    M = P[..., :3, :3]
    p4 = P[..., :3, 3]
    S = -(torch.linalg.inv(M) @ p4[..., None])[..., 0]
    beta = torch.atan2(S[..., 1], S[..., 0])
    order = torch.argsort(beta, dim=-1)
    bs = torch.gather(beta, -1, order)
    gap = torch.cat([bs[..., 1:] - bs[..., :-1],
                     (bs[..., 0] + 2.0 * math.pi - bs[..., -1])[..., None]], dim=-1)
    share = 0.5 * (gap + torch.roll(gap, 1, dims=-1))
    return torch.zeros_like(share).scatter(-1, order, share)


def main():
    torch.manual_seed(0)
    f64 = dict(device=dev, dtype=torch.float64)

    # ---- small config for the math gates ----------------------------------------------------
    cfg = ConeBeam3DConfig.thies(n_views=30)
    P_nom = build_conebeam_orbit(cfg, device=dev, dtype=torch.float64)
    theta = random_motion(cfg.n_views, trans_mm=10.0, rot_deg=10.0, device=dev,
                          dtype=torch.float64)
    s0 = 0.37

    # T1: dP/ds vs float64 central difference
    P, dP = bridge_P_and_dP(theta, P_nom, s0)
    d = 1e-4
    dP_fd = (params_to_Pmot((s0 + d) * theta, P_nom)
             - params_to_Pmot((s0 - d) * theta, P_nom)) / (2 * d)
    m, _ = rel(dP, dP_fd)
    gate("T1 dP/ds", m < 1e-6, f"rel max {m:.3e}")

    # T2: Voronoi weight + derivative
    w0, dw0 = view_angular_weights_dot(P[None], dP[None])
    m_w, _ = rel(w0[0], view_angular_weights(P[None])[0])
    d = 1e-5
    w_p = view_angular_weights(params_to_Pmot((s0 + d) * theta, P_nom)[None])[0]
    w_m = view_angular_weights(params_to_Pmot((s0 - d) * theta, P_nom)[None])[0]
    m_d, _ = rel(dw0[0], (w_p - w_m) / (2 * d))
    gate("T2 weights", m_w < 1e-9 and m_d < 1e-6,
         f"value rel {m_w:.3e} | deriv rel {m_d:.3e}")

    # ---- T3: the whole chain vs torch.autograd jvp, float64 --------------------------------
    D = H = W = 64
    vox = dict(D=D, H=H, W=W, dx=3.0, dy=3.0, dz=3.0)
    bp_geo = dict(du=float(cfg.du), dv=float(cfg.dv), u0=float(cfg.det_offset_u_mm),
                  v_off=0.0, half_u=0.5 * cfg.nu * float(cfg.du),
                  half_v=0.5 * cfg.nv * float(cfg.dv), eps=1e-8)
    g64 = smooth_sino(1, cfg.n_views, cfg.nv, cfg.nu, seed=3, dtype=torch.float64)

    def value_of_s(s):
        Ps = P_of_s(s, theta, P_nom)[None]
        ws = w_of_P(Ps)
        out, _ = _backproject_tangent_torch(
            g64, Ps, torch.zeros_like(Ps), ws, torch.zeros_like(ws), **vox, **bp_geo)
        return out

    s_t = torch.tensor(s0, **f64)
    _, jvp_ref = torch.autograd.functional.jvp(value_of_s, s_t, torch.ones((), **f64))
    _, dx_ana = _backproject_tangent_torch(
        g64, P[None], dP[None], w0, dw0, **vox, **bp_geo)
    m, r = rel(dx_ana, jvp_ref)
    gate("T3 chain-vs-jvp", m < 1e-9, f"rel max {m:.3e} rms {r:.3e}")

    # T3b: coarse FD (the OLD velocity target) vs analytic -- reports the error the finite
    # difference lived with; the bilinear kinks make this O(delta), not O(delta^2).
    d = 0.02
    fd = (value_of_s(torch.tensor(s0 + d, **f64))
          - value_of_s(torch.tensor(s0 - d, **f64))) / (2 * d)
    m, r = rel(fd, dx_ana)
    gate("T3b fd(0.02)-vs-analytic", r < 2e-2, f"rel max {m:.3e} rms {r:.3e}")

    # ---- T4/T5: the fp32 wrapper on the same small config ----------------------------------
    u, v = detector_coords_3d(cfg, device=dev)
    P32 = P[None].float()
    dP32 = dP[None].float()
    sino32 = g64.float()
    w32, dw32 = w0.float(), dw0.float()

    os.environ["FM3D_FDK_TRITON"] = "1"
    x_tan, dx_tan = fdk_conebeam_3d_tangent(sino32, P32, dP32, u, v, cfg, **vox,
                                            view_weight=w32, view_weight_dot=dw32)
    x_plain = fdk_conebeam_3d_batched(sino32, P32, u, v, cfg, **vox, view_weight=w32)
    m, r = rel(x_tan, x_plain)
    gate("T4 value-factoring", m < 5e-4, f"rel max {m:.3e} rms {r:.3e}")

    os.environ["FM3D_FDK_TRITON"] = "0"
    x_ref, dx_ref = fdk_conebeam_3d_tangent(sino32, P32, dP32, u, v, cfg, **vox,
                                            view_weight=w32, view_weight_dot=dw32)
    os.environ.pop("FM3D_FDK_TRITON", None)
    m_x, _ = rel(x_tan, x_ref)
    m_d, r_d = rel(dx_tan, dx_ref)
    # max NOT gated for the tangent -- cell-edge kink voxels, see the module docstring
    reld = (dx_tan - dx_ref).abs() / dx_ref.abs().amax().clamp_min(1e-30)
    frac = float((reld > 1e-3).float().mean())
    gate("T5 triton-vs-torch", m_x < 1e-4 and r_d < 1e-3 and frac < 1e-3,
         f"value rel {m_x:.3e} | tangent rms {r_d:.3e}, max {m_d:.3e} (not gated), "
         f"frac>1e-3 {frac:.2e}")

    # ---- T6: full scale -- consistency + the timing that motivated all of this -------------
    cfgF = ConeBeam3DConfig.thies(n_views=360)
    P_nomF = build_conebeam_orbit(cfgF, device=dev)
    uF, vF = detector_coords_3d(cfgF, device=dev)
    thF = random_motion(cfgF.n_views, trans_mm=10.0, rot_deg=10.0, device=dev)
    sF = smooth_sino(1, cfgF.n_views, cfgF.nv, cfgF.nu, seed=4)
    voxF = dict(D=256, H=256, W=256, dx=1.0, dy=1.0, dz=1.0)
    s_mid = 0.5

    def fdk_at(s, triton=True):
        os.environ["FM3D_FDK_TRITON"] = "1" if triton else "0"
        try:
            Ps = params_to_Pmot(s * thF, P_nomF)[None]
            return fdk_conebeam_3d_batched(sF, Ps, uF, vF, cfgF, **voxF,
                                           view_weight=view_angular_weights(Ps))
        finally:
            os.environ.pop("FM3D_FDK_TRITON", None)

    def tangent_at(s):
        PF, dPF = bridge_P_and_dP(thF, P_nomF, s)
        wF, dwF = view_angular_weights_dot(PF[None], dPF[None])
        return fdk_conebeam_3d_tangent(sF, PF[None], dPF[None], uF, vF, cfgF, **voxF,
                                       view_weight=wF, view_weight_dot=dwF)

    x_a, dx_a = tangent_at(s_mid)
    d = 0.02
    dx_fd = (fdk_at(s_mid + d) - fdk_at(s_mid - d)) / (2 * d)
    m, r = rel(dx_a, dx_fd)
    gate("T6 full-scale fd-consistency", r < 5e-2, f"rel max {m:.3e} rms {r:.3e}")

    def clock(fn, n=3):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t_old = clock(lambda: [fdk_at(s_mid - d, False), fdk_at(s_mid, False),
                           fdk_at(s_mid + d, False)], n=1)       # the OLD bridge_pair
    t_3tri = clock(lambda: [fdk_at(s_mid - d), fdk_at(s_mid), fdk_at(s_mid + d)])
    t_fuse = clock(lambda: tangent_at(s_mid))
    gate("T6 timing", True,
         f"bridge tangent draw: 3x torch FDK {t_old:.2f}s | 3x triton FDK {t_3tri:.2f}s | "
         f"fused analytic {t_fuse:.2f}s ({t_old / t_fuse:.1f}x vs old)")

    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
