"""Fused Triton ray-march sampler for the 3D cone-beam forward projector.

WHY. Profiling the torch-native projector on the real config (94x256x256, V=660, det_bin=2,
n_samples=384) showed that `F.grid_sample` -- the actual trilinear interpolation -- was only
8% of a block. 92% went into MATERIALIZING the (rays, n_samples, 3) sample-coordinate tensor
and reading it back. Fusing the world->normalized map into one `addcmul` cut that 22x, but the
coordinate tensor (432 MiB per block) is still written, read by `grid_sample`, and -- worse --
RETAINED by autograd for the backward pass.

Every production GPU projector (ASTRA `cone_fp.cu`, TIGRE `ray_interpolated_projection.cu`,
torch-radon `texture.cu`) avoids this by generating each sample coordinate in-register inside a
fused march loop, sampling the volume through a `cudaTextureObject_t`. This kernel does the
same, minus the texture: coordinates never leave registers, and no coordinate tensor exists.

WHY NOT TEXTURES. Two reasons, in order:
  1. Hardware trilinear on NVIDIA computes the interpolation weights in 9-bit fixed point, i.e.
     it quantizes to ~1/256 of a voxel. This project's load-bearing gate is
     `phi=Id MC-FDK == static FDK` to 5.7e-07; texture filtering would break it.
  2. `cudaTextureObject_t` is reachable only from CUDA C++. This machine has nvcc 11.2 while
     `flow_matching`'s torch 2.8.0 is built against CUDA 12.8, so `cpp_extension` cannot build
     (major-version mismatch). Triton ships its own ptxas and needs no system toolkit.
So: fp32 manual trilinear. We keep exactness and give up the texture unit's free interpolation.

CONVENTION. Sampling is in CONTINUOUS VOXEL INDEX coordinates, reproducing
`F.grid_sample(..., mode='bilinear', padding_mode='zeros', align_corners=False)` exactly:
index 0.0 is the first voxel's CENTRE, and any of the 8 corners falling outside
[0, W-1] x [0, H-1] x [0, D-1] contributes 0 (that is what `padding_mode='zeros'` means, and it
is the physically right choice here -- a ray leaving the FOV must accumulate no attenuation).

ADJOINT. The backward kernel scatters `grad_out * step * w_corner` into the volume gradient with
`tl.atomic_add`. That is the EXACT transpose of the forward, so the DC step keeps a matched
projector pair and avoids the biased fixed point an unmatched pair converges to
(Zeng & Gullberg, IEEE TMI 19(5):548, 2000). Atomics make it nondeterministic in summation
ORDER only, not in value beyond float reassociation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                             # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _trilinear(vol_ptr, b, px, py, pz, W, H, D, mask):
        """grid_sample(align_corners=False, padding_mode='zeros') at continuous voxel index."""
        x0 = tl.floor(px)
        y0 = tl.floor(py)
        z0 = tl.floor(pz)
        fx = px - x0
        fy = py - y0
        fz = pz - z0
        ix = x0.to(tl.int32)
        iy = y0.to(tl.int32)
        iz = z0.to(tl.int32)
        acc = tl.zeros(px.shape, dtype=tl.float32)
        for cz in tl.static_range(2):
            zc = iz + cz
            wz = fz if cz == 1 else 1.0 - fz
            okz = (zc >= 0) & (zc < D)
            for cy in tl.static_range(2):
                yc = iy + cy
                wy = fy if cy == 1 else 1.0 - fy
                oky = okz & (yc >= 0) & (yc < H)
                for cx in tl.static_range(2):
                    xc = ix + cx
                    wx = fx if cx == 1 else 1.0 - fx
                    ok = mask & oky & (xc >= 0) & (xc < W)
                    off = ((b * D + zc) * H + yc) * W + xc
                    v = tl.load(vol_ptr + off, mask=ok, other=0.0)
                    acc += v * (wx * wy * wz)
        return acc

    @triton.jit
    def _fwd_kernel(vol_ptr, Ax, Ay, Az, Bx, By, Bz, stp, out_ptr,
                    R, rays_per_batch, W, H, D,
                    NS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        b = (r // rays_per_batch).to(tl.int32)
        ax = tl.load(Ax + r, mask=mask, other=0.0)
        ay = tl.load(Ay + r, mask=mask, other=0.0)
        az = tl.load(Az + r, mask=mask, other=0.0)
        bx = tl.load(Bx + r, mask=mask, other=0.0)
        by = tl.load(By + r, mask=mask, other=0.0)
        bz = tl.load(Bz + r, mask=mask, other=0.0)
        s = tl.load(stp + r, mask=mask, other=0.0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(NS):
            t = k + 0.5
            acc += _trilinear(vol_ptr, b, ax + bx * t, ay + by * t, az + bz * t,
                              W, H, D, mask)
        tl.store(out_ptr + r, acc * s, mask=mask)

    @triton.jit
    def _bwd_kernel(vol_ptr, gvol_ptr, Ax, Ay, Az, Bx, By, Bz, stp, gout_ptr,
                    gAx, gAy, gAz, gBx, gBy, gBz, gstp,
                    R, rays_per_batch, W, H, D,
                    NS: tl.constexpr, BLOCK: tl.constexpr,
                    NEED_VOL: tl.constexpr, NEED_RAY: tl.constexpr):
        """Adjoint in the volume AND in the ray constants (A, B, step).

        The ray-constant half is what makes motion estimation possible on this kernel: A, B and
        step are the only route by which the projection matrices enter, so without their gradient
        d(loss)/d(theta) is zero (see rigid_motion.py). It needs the SPATIAL derivative of the
        trilinear interpolant, which the corner weights hand over for free:

            f      = sum_corners  v * wx * wy * wz          wx = fx (cx=1) or 1-fx (cx=0)
            df/dpx = sum_corners  v * (+1 if cx else -1) * wy * wz

        and then, since p_k = A + B*(k+1/2),

            d out/d step = sum_k f(p_k)                  (the bare line integral)
            d out/d A    = step * sum_k  df/dp (p_k)
            d out/d B    = step * sum_k  df/dp (p_k) * (k + 1/2)

        Out-of-bounds corners contribute v = 0 to BOTH sums, so this reproduces
        `padding_mode='zeros'` in the gradient exactly as the forward does in the value -- no
        boundary delta term, which is also what grid_sample's own backward omits.

        NEED_VOL / NEED_RAY are compile-time flags. The ray half must LOAD the eight corner
        values (the volume half only scatters into them), so computing it when nobody asked
        would double this kernel's memory traffic for nothing -- and the FM training path, which
        differentiates the volume and holds P fixed, asks for exactly the other half.
        """
        pid = tl.program_id(0)
        r = pid * BLOCK + tl.arange(0, BLOCK)
        mask = r < R
        b = (r // rays_per_batch).to(tl.int32)
        ax = tl.load(Ax + r, mask=mask, other=0.0)
        ay = tl.load(Ay + r, mask=mask, other=0.0)
        az = tl.load(Az + r, mask=mask, other=0.0)
        bx = tl.load(Bx + r, mask=mask, other=0.0)
        by = tl.load(By + r, mask=mask, other=0.0)
        bz = tl.load(Bz + r, mask=mask, other=0.0)
        s = tl.load(stp + r, mask=mask, other=0.0)
        go = tl.load(gout_ptr + r, mask=mask, other=0.0)
        g = go * s

        acc = tl.zeros((BLOCK,), dtype=tl.float32)      # sum_k f          -> d/d step
        dax = tl.zeros((BLOCK,), dtype=tl.float32)      # sum_k df/dpx     -> d/d A
        day = tl.zeros((BLOCK,), dtype=tl.float32)
        daz = tl.zeros((BLOCK,), dtype=tl.float32)
        dbx = tl.zeros((BLOCK,), dtype=tl.float32)      # sum_k df/dpx * t -> d/d B
        dby = tl.zeros((BLOCK,), dtype=tl.float32)
        dbz = tl.zeros((BLOCK,), dtype=tl.float32)

        for k in range(NS):
            t = k + 0.5
            px = ax + bx * t
            py = ay + by * t
            pz = az + bz * t
            x0 = tl.floor(px)
            y0 = tl.floor(py)
            z0 = tl.floor(pz)
            fx = px - x0
            fy = py - y0
            fz = pz - z0
            ix = x0.to(tl.int32)
            iy = y0.to(tl.int32)
            iz = z0.to(tl.int32)
            fk = tl.zeros((BLOCK,), dtype=tl.float32)
            gx = tl.zeros((BLOCK,), dtype=tl.float32)
            gy = tl.zeros((BLOCK,), dtype=tl.float32)
            gz = tl.zeros((BLOCK,), dtype=tl.float32)
            for cz in tl.static_range(2):
                zc = iz + cz
                wz = fz if cz == 1 else 1.0 - fz
                sz = 1.0 if cz == 1 else -1.0
                okz = (zc >= 0) & (zc < D)
                for cy in tl.static_range(2):
                    yc = iy + cy
                    wy = fy if cy == 1 else 1.0 - fy
                    sy = 1.0 if cy == 1 else -1.0
                    oky = okz & (yc >= 0) & (yc < H)
                    for cx in tl.static_range(2):
                        xc = ix + cx
                        wx = fx if cx == 1 else 1.0 - fx
                        sx = 1.0 if cx == 1 else -1.0
                        ok = mask & oky & (xc >= 0) & (xc < W)
                        off = ((b * D + zc) * H + yc) * W + xc
                        if NEED_VOL:
                            tl.atomic_add(gvol_ptr + off, g * (wx * wy * wz), mask=ok)
                        if NEED_RAY:
                            v = tl.load(vol_ptr + off, mask=ok, other=0.0)
                            fk += v * (wx * wy * wz)
                            gx += v * (sx * wy * wz)
                            gy += v * (wx * sy * wz)
                            gz += v * (wx * wy * sz)
            if NEED_RAY:
                acc += fk
                dax += gx
                day += gy
                daz += gz
                dbx += gx * t
                dby += gy * t
                dbz += gz * t

        if NEED_RAY:
            gs = g                                       # = gout * step
            tl.store(gstp + r, go * acc, mask=mask)
            tl.store(gAx + r, gs * dax, mask=mask)
            tl.store(gAy + r, gs * day, mask=mask)
            tl.store(gAz + r, gs * daz, mask=mask)
            tl.store(gBx + r, gs * dbx, mask=mask)
            tl.store(gBy + r, gs * dby, mask=mask)
            tl.store(gBz + r, gs * dbz, mask=mask)


class _RayMarch(torch.autograd.Function):
    """out[r] = step[r] * sum_k vol[b(r)] @ (A[r] + B[r]*(k+0.5)).

    Differentiable in ALL FOUR tensor inputs. The volume gradient is an atomic scatter (the exact
    transpose of the forward gather); the A / B / step gradients come from the trilinear
    interpolant's spatial derivative -- see `_bwd_kernel`. Only the halves that are actually asked
    for are computed, which matters: the FM training path differentiates the volume with P fixed,
    while motion estimation differentiates P with the volume fixed, and neither should pay for
    the other.
    """

    @staticmethod
    def forward(ctx, vol, A, B, step, rays_per_batch, n_samples, block):
        # vol (Bv,D,H,W) contiguous; A,B (3,R); step (R,)
        Bv, D, H, W = vol.shape
        R = step.numel()
        out = torch.empty(R, device=vol.device, dtype=torch.float32)
        grid = (triton.cdiv(R, block),)
        _fwd_kernel[grid](vol, A[0], A[1], A[2], B[0], B[1], B[2], step, out,
                          R, rays_per_batch, W, H, D, NS=n_samples, BLOCK=block)
        ctx.save_for_backward(vol, A, B, step)
        ctx.meta = (Bv, D, H, W, R, rays_per_batch, n_samples, block)
        return out

    @staticmethod
    def backward(ctx, gout):
        vol, A, B, step = ctx.saved_tensors
        Bv, D, H, W, R, rays_per_batch, n_samples, block = ctx.meta
        need_vol = ctx.needs_input_grad[0]
        need_ray = any(ctx.needs_input_grad[1:4])

        dev = gout.device
        gvol = torch.zeros((Bv, D, H, W), device=dev, dtype=torch.float32) if need_vol else None
        gA = torch.zeros((3, R), device=dev, dtype=torch.float32) if need_ray else None
        gB = torch.zeros((3, R), device=dev, dtype=torch.float32) if need_ray else None
        gs = torch.zeros((R,), device=dev, dtype=torch.float32) if need_ray else None

        if need_vol or need_ray:
            # Triton needs real pointers even for the half it will not write, so the unused
            # outputs alias a 1-element scratch buffer rather than being None.
            z1 = torch.zeros(1, device=dev, dtype=torch.float32)
            _gvol = gvol if need_vol else z1
            _gA, _gB, _gs = (gA, gB, gs) if need_ray else (
                z1.expand(3, 1), z1.expand(3, 1), z1)
            grid = (triton.cdiv(R, block),)
            _bwd_kernel[grid](
                vol, _gvol,
                A[0], A[1], A[2], B[0], B[1], B[2], step, gout.contiguous(),
                _gA[0], _gA[1], _gA[2], _gB[0], _gB[1], _gB[2], _gs,
                R, rays_per_batch, W, H, D,
                NS=n_samples, BLOCK=block, NEED_VOL=need_vol, NEED_RAY=need_ray)

        return gvol, gA, gB, gs, None, None, None


def raymarch(vol, A, B, step, rays_per_batch, n_samples, block=256):
    """vol (Bv,D,H,W) -> line integrals (R,). A,B are (3,R) in VOXEL-INDEX coords, step (R,) mm.

    `A`/`B` are the per-ray offset/slope of the sample path: p_k = A + B*(k+0.5). They are tiny
    (R*3 floats), so autograd saves them instead of the (R, n_samples, 3) coordinate tensor."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available; use the grid_sample projector")
    return _RayMarch.apply(vol.contiguous(), A.contiguous(), B.contiguous(),
                           step.contiguous(), int(rays_per_batch), int(n_samples), int(block))
