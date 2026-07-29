"""Machine-precision parity rig for `triton_leap_grad`: kernel adjoints vs torch autograd.

The FD-of-the-LEAP-loss gate (`gate_leap_projector` T3) proves the gradient is a descent
direction, but its 1e-2-level secant noise cannot tell an exact gradient from the retired
2.9e-3 surrogate. This rig can: both LEAP models (SF, JOSEPH) are re-implemented here in
PURE TORCH -- same maths, same a.e. conventions (trunc-toward-zero indices, clamped
overlaps, border-zero taps) -- on a small problem, and torch.autograd differentiates them
w.r.t. the LEAP-form arrays (src, moduleCenter, rowVec, colVec). The Triton kernels must
match BOTH the value and all four gradient arrays to fp32 round-off.

The P-chain half of `leap_grad_P` (the vjp through `modular_arrays_torch`) is plain torch
autograd and needs no rig. The VALUE-vs-real-LEAP half lives in the gate (tex quantization
bars); this file is about the hand-written reverse-mode chains.

    CUDA_VISIBLE_DEVICES=1 python scripts/dev_leap_ref_autograd.py     # ~30 s

The mirror of `scripts/dev_sf_ref_autograd.py`, which played this role for the retired SF
kernel (and whose gauge-FD trap ledger applies here too: autograd, not FD, is the referee).

THE KINK TRAP (measured 2026-07-29, one afternoon of bisection -- do not re-diagnose):
this rig's volume MUST BE SMOOTH. The models' gradients are a.e.-subgradients with kinks
wherever a sample coordinate crosses an integer (a bilinear tap pair swaps) or a footprint
edge crosses a pixel edge. The fp32 kernel and the fp64 reference can land on OPPOSITE SIDES
of such a kink (seen: cb = 18.99999 in fp64, >= 19.0 in fp32) -- the VALUE is continuous
there, so it stays 8-digit exact, while the two gradients are both valid subgradients that
differ by the tap's local jump. On a white-noise volume that jump is O(1) at single unlucky
pixels (a whole view read 28% off from ONE pixel); it means NOTHING for descent. The jump is
proportional to the volume's local SECOND difference, so a blurred volume suppresses it
below fp32 round-off -- the same reason the FD gate uses a C1 phantom.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fm3d.geometry_3d import ConeBeam3DConfig, build_conebeam_orbit, detector_coords_3d
from fm3d.rigid_motion import params_to_Pmot
from fm3d.triton_leap_grad import leap_grad_geom, leap_forward_model, modular_arrays_torch

DEV = "cuda"
_FAILED = []


def check(name, ok, detail=""):
    print(f"     [{'PASS' if ok else 'FAIL'}] {name}   {detail}", flush=True)
    if not ok:
        _FAILED.append(name)


def _taps(vol, iz, icross, j, ydom, NP, D, H, W):
    """Border-zero gather at (z=iz, cross=icross, dominant=j), axis-mapped like the kernel."""
    okz = (iz >= 0) & (iz < D)
    okc = (icross >= 0) & (icross < NP)
    izc = iz.clamp(0, D - 1)
    icc = icross.clamp(0, NP - 1)
    flat = izc * (H * W) + torch.where(ydom, j * W + icc, icc * W + j)
    return vol.reshape(-1)[flat] * (okz & okc).to(vol.dtype)


def sf_model_ref(vol, src, mod, rowv, colv, *, nv, nu, du, dv, dx, dz, D, H, W):
    """modularBeamProjectorKernel_SF in pure differentiable torch. One view."""
    NP = W
    u0g, v0g = -0.5 * (nu - 1) * du, -0.5 * (nv - 1) * dv
    b0, z0 = -0.5 * (NP - 1) * dx, -0.5 * (D - 1) * dz
    m = torch.arange(nv, device=vol.device, dtype=vol.dtype)
    n = torch.arange(nu, device=vol.device, dtype=vol.dtype)
    t_r = (m * dv + v0g)[:, None, None]                       # (nv,1,1)
    s = (n * du + u0g)[None, :, None]                         # (1,nu,1)
    det = mod[None, None] + s * colv[None, None] + t_r * rowv[None, None]   # (nv,nu,3)
    r = det - src[None, None]
    Dd = r.norm(dim=-1)
    pmc = src - mod
    pmc_u, pmc_v = (pmc * colv).sum(), (pmc * rowv).sum()
    rho = torch.sqrt(colv[0] ** 2 + colv[1] ** 2)
    ufx, ufy = colv[0] / rho, colv[1] / rho
    ydom = r[..., 1].abs() > r[..., 0].abs()
    r_a = torch.where(ydom, r[..., 1], r[..., 0])
    r_b = torch.where(ydom, r[..., 0], r[..., 1])
    p_a = torch.where(ydom, src[1], src[0])
    p_b = torch.where(ydom, src[0], src[1])
    uf_sel = torch.where(ydom, ufx, ufy)
    kh = 0.5 * dx * uf_sel.abs() / du
    kv = 0.5 * dz / dv
    alpha = r_a ** 2 + r_b ** 2
    beta = r_a ** 2 + r[..., 2] ** 2
    L = dx * torch.sqrt(alpha * beta) / r_a ** 2
    mf = m[:, None, None]
    nf = n[None, :, None]
    j = torch.arange(NP, device=vol.device)
    w_s = j.to(vol.dtype) * dx + b0                            # (NP,)
    dist_a = w_s - p_a[..., None]                              # (nv,nu,NP)
    bco = p_b[..., None] + dist_a * (r_b / r_a)[..., None]
    zpt = src[2] + dist_a * (r[..., 2] / r_a)[..., None]
    ib = torch.trunc(0.5 + (bco - b0) / dx).long()
    iz = torch.trunc(0.5 + (zpt - z0) / dz).long()
    b_c = ib.to(vol.dtype) * dx + b0
    z_c = iz.to(vol.dtype) * dz + z0
    dist = torch.sqrt((p_b[..., None] - bco) ** 2 + dist_a ** 2
                      + (src[2] - zpt) ** 2)
    t_mag = Dd[..., None] / dist
    dlx = torch.where(ydom[..., None], b_c - src[0], w_s - src[0])
    dly = torch.where(ydom[..., None], w_s - src[1], b_c - src[1])
    dlz = z_c - src[2]
    U = dlx * colv[0] + dly * colv[1] + dlz * colv[2]
    Vv = dlx * rowv[0] + dly * rowv[1] + dlz * rowv[2]
    iu_c = (pmc_u + t_mag * U - u0g) / du
    iv_c = (pmc_v + t_mag * Vv - v0g) / dv
    hfw = kh[..., None] * t_mag
    vfw = kv * t_mag
    n_pos, n_neg = nf + 0.5, nf - 0.5
    m_pos, m_neg = mf + 0.5, mf - 0.5
    hW1 = (torch.minimum(n_pos, iu_c + hfw) - torch.maximum(n_neg, iu_c - hfw)).clamp(min=0)
    upos = (uf_sel > 0)[..., None]
    A0 = torch.where(upos, torch.minimum(n_pos, iu_c - hfw),
                     torch.minimum(n_pos, iu_c + 2 * hfw))
    B0 = torch.where(upos, torch.maximum(n_neg, iu_c - 2 * hfw),
                     torch.maximum(n_neg, iu_c + hfw))
    hW0 = (A0 - B0).clamp(min=0)
    hW2 = (1 - hW1 - hW0).clamp(min=0)
    vW1 = (torch.minimum(m_pos, iv_c + vfw) - torch.maximum(m_neg, iv_c - vfw)).clamp(min=0)
    vW0 = (torch.minimum(m_pos, iv_c - vfw)
           - torch.maximum(m_neg, iv_c - 2 * vfw)).clamp(min=0)
    vW2 = (1 - vW1 - vW0).clamp(min=0)
    jj = j[None, None, :].expand_as(ib)
    yd = ydom[..., None].expand_as(ib)
    G = torch.zeros_like(bco)
    for a, hw in ((-1, hW0), (0, hW1), (1, hW2)):
        for c, vw in ((-1, vW0), (0, vW1), (1, vW2)):
            G = G + hw * vw * _taps(vol, iz + c, ib + a, jj, yd, NP, D, H, W)
    return L * G.sum(dim=-1)


def joseph_model_ref(vol, src, mod, rowv, colv, *, nv, nu, du, dv, dx, dz, D, H, W):
    """modularBeamJosephProjectorKernel + lineIntegral_Joseph_ZYX in pure torch. One view."""
    NP = W
    u0g, v0g = -0.5 * (nu - 1) * du, -0.5 * (nv - 1) * dv
    b0, z0 = -0.5 * (NP - 1) * dx, -0.5 * (D - 1) * dz
    m = torch.arange(nv, device=vol.device, dtype=vol.dtype)
    n = torch.arange(nu, device=vol.device, dtype=vol.dtype)
    t_r = (m * dv + v0g)[:, None, None]
    s = (n * du + u0g)[None, :, None]
    det = mod[None, None] + s * colv[None, None] + t_r * rowv[None, None]
    r = det - src[None, None]
    ydom = r[..., 1].abs() > r[..., 0].abs()
    r_a = torch.where(ydom, r[..., 1], r[..., 0])
    r_b = torch.where(ydom, r[..., 0], r[..., 1])
    p_a = torch.where(ydom, src[1], src[0])
    p_b = torch.where(ydom, src[0], src[1])
    L = dx * r.norm(dim=-1) / r_a.abs()
    j0 = torch.where(r_a > 0, 0, NP - 1)
    j = torch.arange(NP, device=vol.device)
    w_j = j.to(vol.dtype) * dx + b0
    lam = (w_j - p_a[..., None]) / r_a[..., None]
    cb = (p_b[..., None] + lam * r_b[..., None] - b0) / dx
    cz = (src[2] + lam * r[..., 2][..., None] - z0) / dz
    ibf = torch.floor(cb)
    izf = torch.floor(cz)
    wb = cb - ibf
    wz = cz - izf
    ib = ibf.long()
    iz = izf.long()
    jj = j[None, None, :].expand_as(ib)
    yd = ydom[..., None].expand_as(ib)
    S = (1 - wb) * (1 - wz) * _taps(vol, iz, ib, jj, yd, NP, D, H, W) \
        + wb * (1 - wz) * _taps(vol, iz, ib + 1, jj, yd, NP, D, H, W) \
        + (1 - wb) * wz * _taps(vol, iz + 1, ib, jj, yd, NP, D, H, W) \
        + wb * wz * _taps(vol, iz + 1, ib + 1, jj, yd, NP, D, H, W)
    wgt = torch.where(jj == j0[..., None], 0.5, 1.0)
    return L * (wgt * S).sum(dim=-1)


def main():
    torch.manual_seed(0)
    # Small but not degenerate: cube volume, panel wider than tall, moved geometry.
    D = H = W = 48
    dx = dy = dz = 2.0
    nv, nu = 40, 56
    cfg = ConeBeam3DConfig.thies(n_views=6)
    P_nom = build_conebeam_orbit(cfg, device=DEV)
    th = torch.randn(6, 6, device=DEV) * torch.tensor([3., 3., 3., .05, .05, .05],
                                                      device=DEV)
    th[2, 3] += 0.14                    # ~8 deg: exercise a tilted panel in both models
    P = params_to_Pmot(th, P_nom)
    u, v = detector_coords_3d(cfg, device=DEV)
    du = float(u[1] - u[0]) * 10.0      # coarse panel so footprints span cells
    dv = float(v[1] - v[0]) * 10.0
    vol = torch.rand(D, H, W, device=DEV) * 0.02
    vol[10:38, 12:36, 8:40] += 0.02     # a block so borders and interior both matter
    # SMOOTH, or the rig lies -- see the kink-trap note in the module docstring.
    k = torch.exp(-0.5 * (torch.arange(-6, 7, device=DEV, dtype=torch.float32) / 2.0) ** 2)
    k = (k / k.sum())
    v5 = vol[None, None]
    for d in range(3):
        sh = [1, 1, 1, 1, 1]; sh[2 + d] = k.numel()
        pad = tuple(k.numel() // 2 if i == d else 0 for i in range(3))
        v5 = torch.nn.functional.conv3d(v5, k.view(sh), padding=pad)
    vol = v5[0, 0].contiguous()
    gout = torch.randn(6, nv, nu, device=DEV)

    with torch.no_grad():
        arr32 = tuple(a.to(torch.float32).contiguous()
                      for a in modular_arrays_torch(P.to(torch.float64), 0.0, 0.0))
    kwm = dict(nv=nv, nu=nu, du=du, dv=dv, dx=dx, dz=dz, D=D, H=H, W=W)

    for kind, ref in (("SF", sf_model_ref), ("JOSEPH", joseph_model_ref)):
        # ---- torch-autograd reference (fp64) over the arrays
        arrs = tuple(a.to(torch.float64).clone().requires_grad_(True) for a in arr32)
        g_ref = torch.stack([ref(vol.to(torch.float64), arrs[0][i], arrs[1][i],
                                 arrs[2][i], arrs[3][i], **kwm) for i in range(6)])
        (g_ref * gout.to(torch.float64)).sum().backward()
        ref_grads = [a.grad.clone() for a in arrs]

        # ---- kernel value
        g_ker = leap_forward_model(vol[None], P[None], nv=nv, nu=nu, dx=dx, dy=dy, dz=dz,
                                   du=du, dv=dv, kind=kind)[0]
        rv = float((g_ker - g_ref.float()).norm() / g_ref.norm())
        check(f"[{kind}] value: kernel vs torch reference", rv < 5e-5, f"rel = {rv:.2e}")

        # ---- the fp32 NOISE FLOOR: the same autograd reference run in fp32. The kernel is
        # fp32 throughout; what it can be held to is this floor, not fp64 truth. (World
        # coordinates are O(600 mm) and the adjoint chains cancel against each other, so the
        # conditioning costs ~3 digits -- measured, see the printout.)
        arrs32g = tuple(a.to(torch.float32).clone().requires_grad_(True) for a in arr32)
        g32 = torch.stack([ref(vol.to(torch.float32), arrs32g[0][i], arrs32g[1][i],
                               arrs32g[2][i], arrs32g[3][i], **kwm) for i in range(6)])
        (g32 * gout).sum().backward()
        floor = max(float((a.grad - rg.float()).norm() / (rg.norm() + 1e-30))
                    for a, rg in zip(arrs32g, ref_grads))
        print(f"     [....] [{kind}] fp32 autograd noise floor vs fp64: {floor:.2e}")

        # ---- kernel gradient: must sit at/below the fp32 floor (x3 slack for different
        # summation orders), not at fp64 truth
        ker_grads = leap_grad_geom(vol, gout, P, nv=nv, nu=nu, dx=dx, dz=dz, du=du, dv=dv,
                                   kind=kind)
        for nm, kg, rg in zip(("src", "mod", "rowv", "colv"), ker_grads, ref_grads):
            rel = float((kg - rg).norm() / (rg.norm() + 1e-30))
            check(f"[{kind}] d/d{nm}: kernel vs autograd", rel < max(3 * floor, 2e-4),
                  f"rel = {rel:.2e}  (fp32 floor {floor:.1e})")

    # ---- the VD BACKPROJECTION TANGENT (the bridge's velocity target, 2026-07-30) ---------
    # `leap_fdk_backproject_tangent` claims to be the EXACT s-derivative of the deployed FDK
    # backprojection. Referee: a pure-torch fp64 replica of the VD model, jvp'd through the
    # modular decomposition AND the per-view weights. Smooth sinogram -- the kink trap above
    # applies here too (bilinear cells in (ju, jv) instead of volume voxels).
    from fm3d.leap_projector import leap_fdk_backproject_tangent

    def vd_ref(g2, src, mod, rowv, colv, wgt, *, D, H, W, dx, dz, du, dv):
        V, nv_, nu_ = g2.shape
        dt = g2.dtype
        ii = torch.arange(W, device=g2.device, dtype=dt)
        jj = torch.arange(H, device=g2.device, dtype=dt)
        kk = torch.arange(D, device=g2.device, dtype=dt)
        zz, yy, xx = torch.meshgrid((kk - (D - 1) / 2) * dz, (jj - (H - 1) / 2) * dx,
                                    (ii - (W - 1) / 2) * dx, indexing="ij")
        out = torch.zeros((D, H, W), device=g2.device, dtype=dt)
        for i in range(V):
            u_, v_, p_, c_ = colv[i], rowv[i], src[i], mod[i]
            n_ = torch.linalg.cross(u_, v_)
            pmc = p_ - c_
            pmcn, pmcu, pmcv = (pmc * n_).sum(), (pmc * u_).sum(), (pmc * v_).sum()
            rx, ry, rz = xx - p_[0], yy - p_[1], zz - p_[2]
            rdn = rx * n_[0] + ry * n_[1] + rz * n_[2]
            Dm = -pmcn / rdn
            ru = rx * u_[0] + ry * u_[1] + rz * u_[2]
            rv = rx * v_[0] + ry * v_[1] + rz * v_[2]
            ju = (pmcu + Dm * ru) / du + (nu_ - 1) / 2
            jv = (pmcv + Dm * rv) / dv + (nv_ - 1) / 2
            Wt = pmcn * torch.sqrt(Dm ** 2 * (ru ** 2 + rv ** 2) + pmcn ** 2) / rdn ** 2
            x0 = torch.floor(ju)
            y0 = torch.floor(jv)
            fu, fv = ju - x0, jv - y0
            iu, iv = x0.long(), y0.long()
            gi = g2[i]

            def tap(a, b):
                okm = (a >= 0) & (a < nv_) & (b >= 0) & (b < nu_)
                return gi[a.clamp(0, nv_ - 1), b.clamp(0, nu_ - 1)] * okm.to(dt)

            G = (1 - fu) * (1 - fv) * tap(iv, iu) + fu * (1 - fv) * tap(iv, iu + 1) \
                + (1 - fu) * fv * tap(iv + 1, iu) + fu * fv * tap(iv + 1, iu + 1)
            out = out + wgt[i] * Wt * G
        return out

    from fm3d.rigid_motion import bridge_P_and_dP
    th_t = torch.randn(6, 6, device=DEV, dtype=torch.float64) * torch.tensor(
        [3., 3., 3., .05, .05, .05], device=DEV, dtype=torch.float64)
    th_t[2, 3] += 0.14
    P_nom64 = build_conebeam_orbit(cfg, device=DEV, dtype=torch.float64)
    P_t, dP_t = bridge_P_and_dP(th_t, P_nom64, 0.37)
    nvt, nut = 64, 88
    dut, dvt = du * 1.2, dv * 1.2
    g_t = torch.rand(1, 6, nvt, nut, device=DEV)
    g_t = torch.nn.functional.avg_pool2d(g_t.view(6, 1, nvt, nut), 9, 1, 4)
    g_t = torch.nn.functional.avg_pool2d(g_t, 9, 1, 4).view(1, 6, nvt, nut).contiguous()
    wgt_t = torch.rand(1, 6, device=DEV) + 0.5
    dwgt_t = torch.randn(1, 6, device=DEV) * 0.1
    x_k, dx_k = leap_fdk_backproject_tangent(
        g_t, P_t[None].float(), dP_t[None].float(), wgt_t, dwgt_t,
        D=D, H=H, W=W, dx=dx, dy=dx, dz=dz, du=dut, dv=dvt)
    sddt = float(P_t[:, 0, :3].norm(dim=-1).mean())
    uu_t = (torch.arange(nut, device=DEV) - (nut - 1) / 2) * dut
    vv_t = (torch.arange(nvt, device=DEV) - (nvt - 1) / 2) * dvt
    dist_t = torch.sqrt(torch.tensor(sddt ** 2, device=DEV)
                        + uu_t[None, :] ** 2 + vv_t[:, None] ** 2)
    g2_t = (g_t[0] / (sddt * dist_t)[None]).to(torch.float64)

    def tang_ref(P64, w64):
        a = modular_arrays_torch(P64, 0.0, 0.0)
        return vd_ref(g2_t, a[0], a[1], a[2], a[3], w64,
                      D=D, H=H, W=W, dx=dx, dz=dz, du=dut, dv=dvt)

    x_r, dx_r = torch.autograd.functional.jvp(
        tang_ref, (P_t, wgt_t[0].to(torch.float64)),
        (dP_t, dwgt_t[0].to(torch.float64)))
    rv_ = float((x_k[0] - x_r.float()).norm() / x_r.norm())
    rt_ = float((dx_k[0] - dx_r.float()).norm() / dx_r.norm())
    check("[VD-TANGENT] value: kernel vs torch reference", rv_ < 5e-6, f"rel = {rv_:.2e}")
    check("[VD-TANGENT] ds-derivative: kernel vs fp64 jvp", rt_ < 5e-4, f"rel = {rt_:.2e}")

    print("\n" + ("ALL CHECKS PASS" if not _FAILED else f"FAILED: {_FAILED}"))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
