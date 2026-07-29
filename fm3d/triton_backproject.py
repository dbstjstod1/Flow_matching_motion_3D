"""Fused Triton backprojector for the static (disp=None) FDK, plus its ANALYTIC s-tangent.

WHY. Ported from Flowmatching-4DCT's `fdct/triton_backproject.py` (2026-07-18), where profiling
put the torch FDK at 14.5 s/call and 70% of a bridge draw -- and INSENSITIVE to view_chunk, so
it is not launch-bound. The torch path materializes, per (view_chunk x vox_chunk) block, the
einsum coordinate tensor plus ~a dozen elementwise intermediates (u, v, w, ju, jv, the
normalized grid, the mask ...), each a full read+write of a (vc, Np) tensor: the same
coordinate-bandwidth disease the retired Triton ray-march projector cured in the forward
(module deleted 2026-07-28 with the SF switch; there, 92% of the block). This kernel fuses the whole chain -- P @ x, perspective divide, detector index,
bilinear gather, 1/w^2 mask-accumulate -- so per (voxel, view) the only memory traffic is the
4 detector taps; coordinates never leave registers. Measured there: 24x on the backprojection.

SCOPE. Unlike the 4DCT sibling, this covers EVERY FDK in this project: our motion is rigid and
enters through the per-view Pmat (`P_nom @ T(theta_v)`, never a `disp` field), and the kernel
takes arbitrary per-view (B, V, 3, 4) matrices. The `disp` MC path keeps the torch loop.

THE TANGENT KERNEL (`_bp_tangent_kernel`) is this project's replacement for 4DCT's
`warp_bspline_dt`: their analytic FM tangent differentiates a B-spline warp of the VOLUME,
but our bridge's t-dependence lives in the projection matrices, x(s) = FDK(y, P(s*theta)).
The filtered sinogram is s-independent (filtering is per-view linear, see
`fdk_conebeam_3d_tangent`), so d/ds acts on the backprojection alone:

    per (voxel, view):  u_h, v_h, w = P X ;  du_h, dv_h, dw = Pdot X
                        udot = (du_h - u*dw)/w,  vdot = (dv_h - v*dw)/w
                        d[val/w^2] = (g_u*judot + g_v*jvdot)/w^2 - 2*val*dw/w^3

where (g_u, g_v) is the bilinear interpolant's own spatial derivative, read from the SAME
four taps as the value (the trick the retired ray-march kernel's NEED_RAY adjoint used in 3D).
Per-view scalars (wgt, dwgt) fold in the Voronoi angular weight and ITS s-derivative
(`geometry_3d.view_angular_weights_dot`). The mask and the interpolation cell are FROZEN at
the evaluation point -- their moving edges are measure-zero, the same a.e.-Jacobian argument
4DCT used to linearize its truncation clamp (`lin_ref`).

DETERMINISM. No atomics -- each program owns its voxel tile exclusively and LOOPS OVER VIEWS
in a fixed sequential order, then stores once. Two runs are bit-identical.

vs THE TORCH PATH it is NOT bit-identical, for two reasons, both float-reassociation-sized:
  * the view sum is associated ((v0+v1)+v2)+... per voxel here, vs per-view_chunk partial sums;
  * `grid_sample` recovers the pixel coordinate from the normalized grid as
    ((x_norm+1)*nu-1)/2, a round-trip through [-1,1] this kernel skips (it uses ju directly).
Measured agreement is gated by scripts/gate_fdk_fast.py; gates that compare FDK outputs must
run BOTH operands through the same backend.

CONVENTION. Bilinear sampling reproduces `F.grid_sample(mode='bilinear',
padding_mode='zeros', align_corners=False)`: index 0.0 is the first detector element's
CENTRE, and any of the 4 taps falling outside [0, nu-1] x [0, nv-1] contributes 0. The mask
`(w > 0) & (|u-u0| <= half_u) & (|v-v_off| <= half_v)` is applied on top, identical to the
torch path.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                             # pragma: no cover
    HAVE_TRITON = False


def enabled() -> bool:
    """Kill switch: FM3D_FDK_TRITON=0 forces the torch path.

    Independent of the FORWARD projector (LEAP, `leap_projector`) and of the gates
    reference `projector_3d.reference_project_3d_batched`; neither
    implies the other."""
    return HAVE_TRITON and os.environ.get("FM3D_FDK_TRITON", "1") != "0"


if HAVE_TRITON:

    @triton.jit
    def _bilinear2d(g_ptr, base, ju, jv, nu, nv, mask):
        """grid_sample(align_corners=False, padding_mode='zeros') at continuous element index.
        `base` is an int64 SCALAR offset of this (b, v) projection inside g."""
        x0 = tl.floor(ju)
        y0 = tl.floor(jv)
        fx = ju - x0
        fy = jv - y0
        ix = x0.to(tl.int32)
        iy = y0.to(tl.int32)
        acc = tl.zeros(ju.shape, dtype=tl.float32)
        for cy in tl.static_range(2):
            yc = iy + cy
            wy = fy if cy == 1 else 1.0 - fy
            oky = (yc >= 0) & (yc < nv)
            for cx in tl.static_range(2):
                xc = ix + cx
                wx = fx if cx == 1 else 1.0 - fx
                ok = mask & oky & (xc >= 0) & (xc < nu)
                v = tl.load(g_ptr + base + yc * nu + xc, mask=ok, other=0.0)
                acc += v * (wx * wy)
        return acc

    @triton.jit
    def _bp_kernel(g_ptr, P_ptr, out_ptr,
                   V, nv, nu, Npix, W, H, D,
                   dx, dy, dz, du, dv, u0, v_off,
                   half_u, half_v, eps,
                   BLOCK: tl.constexpr, W2: tl.constexpr):
        # W2: apply FDK's 1/w^2 distance weight (the Feldkamp backprojection). W2=False is the
        # plain voxel-driven accumulation the open-source iterative solvers use as their
        # (unmatched) A^T -- RTK's BackProjectionImageFilter and TIGRE's Atb('matched') both
        # backproject WITHOUT the FDK weight. See projector_3d.backproject_3d_batched.
        pid_b = tl.program_id(1)
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = p < Npix
        # voxel linear index -> world mm, matching the torch path's meshgrid exactly:
        # x/y/z centred at the isocentre, index order (z, y, x) row-major.
        iz = p // (H * W)
        rem = p - iz * (H * W)
        iy = rem // W
        ix = rem - iy * W
        x = (ix.to(tl.float32) - (W - 1) * 0.5) * dx
        y = (iy.to(tl.float32) - (H - 1) * 0.5) * dy
        z = (iz.to(tl.float32) - (D - 1) * 0.5) * dz
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for view in range(0, V):
            pb = P_ptr + (pid_b * V + view) * 12
            p00 = tl.load(pb + 0)
            p01 = tl.load(pb + 1)
            p02 = tl.load(pb + 2)
            p03 = tl.load(pb + 3)
            p10 = tl.load(pb + 4)
            p11 = tl.load(pb + 5)
            p12 = tl.load(pb + 6)
            p13 = tl.load(pb + 7)
            p20 = tl.load(pb + 8)
            p21 = tl.load(pb + 9)
            p22 = tl.load(pb + 10)
            p23 = tl.load(pb + 11)
            u_h = p00 * x + p01 * y + p02 * z + p03
            v_h = p10 * x + p11 * y + p12 * z + p13
            w = p20 * x + p21 * y + p22 * z + p23
            w_safe = tl.where(tl.abs(w) < eps, eps, w)
            u = u_h / w_safe
            v = v_h / w_safe
            ju = (u - u0) / du + (nu - 1) * 0.5
            jv = (v - v_off) / dv + (nv - 1) * 0.5
            base = ((pid_b * V + view) * nv).to(tl.int64) * nu
            val = _bilinear2d(g_ptr, base, ju, jv, nu, nv, mask)
            ok = mask & (w > 0) & (tl.abs(u - u0) <= half_u) & (tl.abs(v - v_off) <= half_v)
            if W2:
                val = val / (w_safe * w_safe)
            acc += tl.where(ok, val, 0.0)
        tl.store(out_ptr + pid_b.to(tl.int64) * Npix + p, acc, mask=mask)

    @triton.jit
    def _bp_tangent_kernel(g_ptr, P_ptr, dP_ptr, wgt_ptr, dwgt_ptr, out_ptr, dout_ptr,
                           V, nv, nu, Npix, W, H, D,
                           dx, dy, dz, du, dv, u0, v_off,
                           half_u, half_v, eps,
                           BLOCK: tl.constexpr):
        """Value AND directional derivative of the weighted backprojection.

            out  = sum_v wgt[v]  * mask_v * val_v / w_v^2
            dout = sum_v wgt[v]  * mask_v * d/ds[val_v / w_v^2]
                 + sum_v dwgt[v] * mask_v * val_v / w_v^2

        The value/derivative pair reads the SAME four detector taps: the bilinear value uses
        corner weights (wx*wy) and its ju/jv-derivative uses (sx*wy)/(wx*sy) with sx,sy = +-1
        -- the retired ray-march backward kernel's trilinear trick, one dimension down.
        """
        pid_b = tl.program_id(1)
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = p < Npix
        iz = p // (H * W)
        rem = p - iz * (H * W)
        iy = rem // W
        ix = rem - iy * W
        x = (ix.to(tl.float32) - (W - 1) * 0.5) * dx
        y = (iy.to(tl.float32) - (H - 1) * 0.5) * dy
        z = (iz.to(tl.float32) - (D - 1) * 0.5) * dz
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        dacc = tl.zeros((BLOCK,), dtype=tl.float32)
        for view in range(0, V):
            pb = P_ptr + (pid_b * V + view) * 12
            p00 = tl.load(pb + 0)
            p01 = tl.load(pb + 1)
            p02 = tl.load(pb + 2)
            p03 = tl.load(pb + 3)
            p10 = tl.load(pb + 4)
            p11 = tl.load(pb + 5)
            p12 = tl.load(pb + 6)
            p13 = tl.load(pb + 7)
            p20 = tl.load(pb + 8)
            p21 = tl.load(pb + 9)
            p22 = tl.load(pb + 10)
            p23 = tl.load(pb + 11)
            db = dP_ptr + (pid_b * V + view) * 12
            q00 = tl.load(db + 0)
            q01 = tl.load(db + 1)
            q02 = tl.load(db + 2)
            q03 = tl.load(db + 3)
            q10 = tl.load(db + 4)
            q11 = tl.load(db + 5)
            q12 = tl.load(db + 6)
            q13 = tl.load(db + 7)
            q20 = tl.load(db + 8)
            q21 = tl.load(db + 9)
            q22 = tl.load(db + 10)
            q23 = tl.load(db + 11)
            wg = tl.load(wgt_ptr + pid_b * V + view)
            dwg = tl.load(dwgt_ptr + pid_b * V + view)

            u_h = p00 * x + p01 * y + p02 * z + p03
            v_h = p10 * x + p11 * y + p12 * z + p13
            w = p20 * x + p21 * y + p22 * z + p23
            du_h = q00 * x + q01 * y + q02 * z + q03
            dv_h = q10 * x + q11 * y + q12 * z + q13
            dw = q20 * x + q21 * y + q22 * z + q23
            w_safe = tl.where(tl.abs(w) < eps, eps, w)
            u = u_h / w_safe
            v = v_h / w_safe
            udot = (du_h - u * dw) / w_safe
            vdot = (dv_h - v * dw) / w_safe
            ju = (u - u0) / du + (nu - 1) * 0.5
            jv = (v - v_off) / dv + (nv - 1) * 0.5

            # fused 4-tap value + spatial derivative (cell frozen: d(floor)=0 a.e.)
            x0 = tl.floor(ju)
            y0 = tl.floor(jv)
            fx = ju - x0
            fy = jv - y0
            jx = x0.to(tl.int32)
            jy = y0.to(tl.int32)
            base = ((pid_b * V + view) * nv).to(tl.int64) * nu
            val = tl.zeros((BLOCK,), dtype=tl.float32)
            gu = tl.zeros((BLOCK,), dtype=tl.float32)
            gv = tl.zeros((BLOCK,), dtype=tl.float32)
            for cy in tl.static_range(2):
                yc = jy + cy
                wy = fy if cy == 1 else 1.0 - fy
                sy = 1.0 if cy == 1 else -1.0
                oky = (yc >= 0) & (yc < nv)
                for cx in tl.static_range(2):
                    xc = jx + cx
                    wx = fx if cx == 1 else 1.0 - fx
                    sx = 1.0 if cx == 1 else -1.0
                    ok4 = mask & oky & (xc >= 0) & (xc < nu)
                    gval = tl.load(g_ptr + base + yc * nu + xc, mask=ok4, other=0.0)
                    val += gval * (wx * wy)
                    gu += gval * (sx * wy)
                    gv += gval * (wx * sy)

            ok = mask & (w > 0) & (tl.abs(u - u0) <= half_u) & (tl.abs(v - v_off) <= half_v)
            inv_w2 = 1.0 / (w_safe * w_safe)
            contrib = val * inv_w2
            dcontrib = (gu * (udot / du) + gv * (vdot / dv)) * inv_w2 \
                - 2.0 * contrib * (dw / w_safe)
            acc += tl.where(ok, wg * contrib, 0.0)
            dacc += tl.where(ok, wg * dcontrib + dwg * contrib, 0.0)
        tl.store(out_ptr + pid_b.to(tl.int64) * Npix + p, acc, mask=mask)
        tl.store(dout_ptr + pid_b.to(tl.int64) * Npix + p, dacc, mask=mask)


def backproject_static(g: torch.Tensor, Pmat: torch.Tensor, D: int, H: int, W: int, *,
                       dx: float, dy: float, dz: float, du: float, dv: float,
                       u0: float, v_off: float, half_u: float, half_v: float,
                       eps: float, block: int = 256, w2: bool = True) -> torch.Tensor:
    """Static cone-beam voxel-driven backprojection. `w2=True` (default) applies FDK's 1/w^2
    distance weight (a FILTERED sinogram makes this the Feldkamp step); `w2=False` is the plain
    accumulation the open-source iterative solvers use as their unmatched A^T.

    g (B,V,nv,nu) fp32, Pmat (B,V,3,4) fp32 -> (B,D,H,W). The caller applies the
    angle_span/V weight and `scale` afterwards, exactly as the torch path does."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available; use the torch backprojection")
    B, V, nv, nu = g.shape
    g = g.contiguous()
    P = Pmat.reshape(B, V, 12).contiguous().to(torch.float32)
    Npix = D * H * W
    out = torch.empty((B, Npix), device=g.device, dtype=torch.float32)
    grid = (triton.cdiv(Npix, block), B)
    _bp_kernel[grid](g, P, out, V, nv, nu, Npix, W, H, D,
                     float(dx), float(dy), float(dz), float(du), float(dv),
                     float(u0), float(v_off), float(half_u), float(half_v), float(eps),
                     BLOCK=block, W2=bool(w2))
    return out.view(B, D, H, W)


def backproject_tangent(g: torch.Tensor, Pmat: torch.Tensor, Pdot: torch.Tensor,
                        wgt: torch.Tensor, dwgt: torch.Tensor,
                        D: int, H: int, W: int, *,
                        dx: float, dy: float, dz: float, du: float, dv: float,
                        u0: float, v_off: float, half_u: float, half_v: float,
                        eps: float, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted backprojection AND its exact directional derivative along (Pdot, dwgt).

    g (B,V,nv,nu) fp32 FILTERED sinogram (weights NOT folded in); Pmat/Pdot (B,V,3,4);
    wgt/dwgt (B,V) per-view angular weights [rad] and their s-derivative.
    Returns (x, dx/ds), each (B,D,H,W); the caller applies `scale` to both.
    BLOCK=128 (not 256): the tangent kernel holds ~2x the live registers of the plain one."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available; use the torch tangent backprojection")
    B, V, nv, nu = g.shape
    g = g.contiguous()
    P = Pmat.reshape(B, V, 12).contiguous().to(torch.float32)
    dP = Pdot.reshape(B, V, 12).contiguous().to(torch.float32)
    wg = wgt.reshape(B, V).contiguous().to(torch.float32)
    dwg = dwgt.reshape(B, V).contiguous().to(torch.float32)
    Npix = D * H * W
    out = torch.empty((B, Npix), device=g.device, dtype=torch.float32)
    dout = torch.empty((B, Npix), device=g.device, dtype=torch.float32)
    grid = (triton.cdiv(Npix, block), B)
    _bp_tangent_kernel[grid](g, P, dP, wg, dwg, out, dout, V, nv, nu, Npix, W, H, D,
                             float(dx), float(dy), float(dz), float(du), float(dv),
                             float(u0), float(v_off), float(half_u), float(half_v),
                             float(eps), BLOCK=block)
    return out.view(B, D, H, W), dout.view(B, D, H, W)
