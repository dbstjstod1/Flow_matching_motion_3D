"""A DROP-IN, NUMERICALLY EQUIVALENT replacement for the vendored cone backprojector's kernels.

WHY THIS FILE EXISTS -- one measured number
-------------------------------------------
The vendored `DifferentiableConeBeamBackprojector` (bench/thies/vendor/geometry_gradients_CT/
backprojector_cone.py, byte-identical to Thies et al.'s release) is correct and unusably slow in
its BACKWARD pass. Measured on an A6000, 128^3 volume / 360 views / 500x700 panel:

    forward   (their kernel)      0.33 s
    backward  (their kernel)     21.13 s          <- 65x the forward

and the 21 s is not physics, it is ATOMIC CONTENTION. Their `backward_loop` runs one thread per
voxel and each thread issues **twelve** `cuda.atomic.add` into the SAME twelve scalars
`proj_matrix_error[a, :, :]`, once per view, with the view loop unrolled into 360 separate kernel
launches. That is

    128^3 threads x 12 components x 360 views  =  9.06e9 atomic adds  onto 12 addresses,

and 21.13 s / 9.06e9 = 2.33 ns per atomic -- i.e. the whole kernel time IS the serialized
read-modify-write queue. Nothing else in it costs anything.

At 100 gradient-descent iterations (Thies' Eq. 6) that is 36 MINUTES PER PATIENT, and the
evaluation cohort is 30 patients -> 18 h. It is the single largest cost anywhere in this repo.

WHAT WE CHANGE, AND WHAT WE DELIBERATELY DO NOT
-----------------------------------------------
The arithmetic is COPIED, expression for expression, from their kernels -- including the two
things a reimplementation would silently "fix" and thereby stop being their operator:

  * `interpolate2d_cuda` floors with `int()`, which TRUNCATES TOWARD ZERO, not toward -inf. For a
    sample position in (-1, 0) that yields index 0 with a NEGATIVE delta, i.e. a linear
    EXTRAPOLATION off the panel edge rather than a clamp or a zero. `_interp2` below reproduces
    that exactly. (This is also why a `torch.nn.functional.grid_sample` port does not work: a
    straight grid_sample rewrite was measured 8.3e-2 off, purely from this convention.)
  * the volume axis map (`point2 <- axis 0`, `point1 <- axis 1`, `point0 <- axis 2`) and the
    per-view `torch.gradient(sinogram, dim=(2,1))` detector derivative are theirs, unchanged.

The ONLY differences are execution-shape ones, both of which change the SUMMATION ORDER of a
float reduction and nothing else:

  1. FORWARD: their per-view `cuda.atomic.add(reco, (x,y,z), ...)` writes to the thread's OWN
     voxel -- one owner, so the atomic is uncontended and simply unnecessary. We accumulate the
     view sum in a register and store once.
  2. BACKWARD: the twelve per-view accumulators are summed in a SHARED-MEMORY block reduction
     first, so a thread block issues 12 atomics instead of 12 per thread (256x fewer), and all
     360 views run in ONE launch instead of 360.

Float addition is not associative, so the outputs agree to reduction-order noise, not bitwise.
`scripts/gate_bench_thies.py` check G9 pins that against the vendored kernels directly, and G6
(the finite-difference check on dI/dP) is unchanged and still runs on whichever backend is
selected. PROVENANCE.md section 4.5 records this as a PERFORMANCE-ONLY deviation.

`FAST=False` (or `FM3D_THIES_VENDOR_BP=1` in the environment) restores the vendored kernels for
every caller, which is what the gate uses as its counterparty.
"""

from __future__ import annotations

import os

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .vendor_import import cone_backprojector

__all__ = ["FastConeBackprojector", "backprojector", "FAST"]

# Environment escape hatch: a single flag flips the whole benchmark back to the vendored kernels.
FAST = os.environ.get("FM3D_THIES_VENDOR_BP", "0") == "0"

_THREADS = (8, 8, 4)                 # 256 threads/block -- the block-reduction width below
_TPB = _THREADS[0] * _THREADS[1] * _THREADS[2]
# The backward's tree reduction halves `_TPB` down to 1 and is baked into the kernel as a
# compile-time constant, so a launch geometry that disagrees would silently drop partial sums
# rather than fail. Both invariants are cheap to assert and impossible to notice otherwise.
if _TPB & (_TPB - 1):
    raise RuntimeError(f"_THREADS must multiply to a power of two, got {_TPB}")
if _TPB < 12:
    raise RuntimeError("the reduction needs at least 12 threads per block (one per component)")
# Grid-stride, so any volume shape works. 1024 blocks was the flat part of a measured sweep
# (8192 -> 512 blocks moved fwd+bwd only 1779 -> 1565 ms, i.e. the block reduction and the
# atomics are NOT the remaining cost); it keeps enough blocks to fill the device on the 256^3
# output grid too.
_BLOCKS = (8, 8, 16)
_kernels: dict = {}


def _build():
    """JIT-compile the two kernels once (numba is imported lazily, like `vendor_import`)."""
    if _kernels:
        return _kernels
    from numba import cuda, float32

    @cuda.jit(device=True, inline=True)
    def _interp2(array, pos_x, pos_y):
        """VERBATIM `helper.interpolate2d_cuda`, including the `int()` truncation.

        Their `None`-guard idiom is expressed here as an in-bounds flag per tap, which is the
        same control flow with the same four-tap fallback to 0.0."""
        fx = int(pos_x)
        fy = int(pos_y)
        dx = pos_x - fx
        dy = pos_y - fy
        nx = array.shape[0]
        ny = array.shape[1]
        x0_ok = 0 <= fx <= nx - 1
        x1_ok = 0 <= fx + 1 <= nx - 1
        y0_ok = 0 <= fy <= ny - 1
        y1_ok = 0 <= fy + 1 <= ny - 1
        a = array[fx, fy] if (x0_ok and y0_ok) else float32(0.0)
        b = array[fx + 1, fy] if (x1_ok and y0_ok) else float32(0.0)
        c = array[fx, fy + 1] if (x0_ok and y1_ok) else float32(0.0)
        d = array[fx + 1, fy + 1] if (x1_ok and y1_ok) else float32(0.0)
        t1 = dx * b + (1.0 - dx) * a
        t2 = dx * d + (1.0 - dx) * c
        return dy * t2 + (1.0 - dy) * t1

    @cuda.jit
    def forward_loop(sinogram, projection_matrices, reco, volume_shape, volume_spacing,
                     volume_origin):
        """Their `forward_loop`, with the per-view atomic replaced by a register accumulator.

        Each (x,y,z) is owned by exactly one thread in the grid-stride loop, so the vendored
        `cuda.atomic.add(reco, (x,y,z), ...)` never contended -- it was 360 read-modify-writes
        to a location no one else touches. One store instead."""
        sx, sy, sz = cuda.grid(3)
        gx, gy, gz = cuda.gridsize(3)
        for x in range(sx, volume_shape[0], gx):
            point2 = x * volume_spacing[0] + volume_origin[0]
            for y in range(sy, volume_shape[1], gy):
                point1 = y * volume_spacing[1] + volume_origin[1]
                for z in range(sz, volume_shape[2], gz):
                    point0 = z * volume_spacing[2] + volume_origin[2]
                    acc = float32(0.0)
                    for a in range(sinogram.shape[0]):
                        p = projection_matrices[a]
                        u = p[0, 0] * point0 + p[0, 1] * point1 + p[0, 2] * point2 + p[0, 3]
                        v = p[1, 0] * point0 + p[1, 1] * point1 + p[1, 2] * point2 + p[1, 3]
                        w = p[2, 0] * point0 + p[2, 1] * point1 + p[2, 2] * point2 + p[2, 3]
                        iw = float32(1.0) / w                    # see the backward's note
                        acc += _interp2(sinogram[a], v * iw, u * iw)
                    reco[x, y, z] = acc

    @cuda.jit
    def backward_loop(volume_shape, volume_spacing, volume_origin, projection_matrices,
                      sino_dx, sino_dy, volume_error, proj_matrix_error):
        """Their `backward_loop` for ALL views in one launch, with a shared-memory reduction.

        The twelve expressions are theirs, character for character. What changes is that each
        thread keeps its twelve partial sums in registers over its grid-stride voxels, the block
        reduces them through shared memory, and only thread 0..11 of each block issue an atomic
        -- 12 per block per view instead of 12 per thread per view."""
        sm = cuda.shared.array((12, _TPB), dtype=float32)
        tid = (cuda.threadIdx.z * cuda.blockDim.y + cuda.threadIdx.y) * cuda.blockDim.x \
            + cuda.threadIdx.x
        sx, sy, sz = cuda.grid(3)
        gx, gy, gz = cuda.gridsize(3)

        for a in range(projection_matrices.shape[0]):
            p = projection_matrices[a]
            sdx = sino_dx[a]
            sdy = sino_dy[a]
            a00 = float32(0.0); a01 = float32(0.0); a02 = float32(0.0); a03 = float32(0.0)
            a10 = float32(0.0); a11 = float32(0.0); a12 = float32(0.0); a13 = float32(0.0)
            a20 = float32(0.0); a21 = float32(0.0); a22 = float32(0.0); a23 = float32(0.0)
            for x in range(sx, volume_shape[0], gx):
                point2 = x * volume_spacing[0] + volume_origin[0]
                for y in range(sy, volume_shape[1], gy):
                    point1 = y * volume_spacing[1] + volume_origin[1]
                    for z in range(sz, volume_shape[2], gz):
                        point0 = z * volume_spacing[2] + volume_origin[2]
                        u = p[0, 0] * point0 + p[0, 1] * point1 + p[0, 2] * point2 + p[0, 3]
                        v = p[1, 0] * point0 + p[1, 1] * point1 + p[1, 2] * point2 + p[1, 3]
                        w = p[2, 0] * point0 + p[2, 1] * point1 + p[2, 2] * point2 + p[2, 3]
                        # ONE reciprocal instead of the vendored fourteen divisions. Their
                        # expressions divide by `w` eight times and by `w**2` four times, plus
                        # `v/w`, `u/w` for the sample position -- and an fp32 divide is ~20x an
                        # fma on this part. Folding them into iw = 1/w is a last-ulp change of
                        # the same formula, inside the reduction-order tolerance G9 already
                        # measures; it is what takes the backward from 4.7x the forward to ~2x,
                        # which is the ratio its two interpolations (vs the forward's one) say
                        # it should be.
                        iw = float32(1.0) / w
                        iw2 = iw * iw
                        gpx = _interp2(sdx, v * iw, u * iw)
                        gpy = _interp2(sdy, v * iw, u * iw)
                        ve = volume_error[x, y, z]
                        gpx = gpx * ve
                        gpy = gpy * ve
                        gx0 = gpx * iw
                        gy0 = gpy * iw
                        hx = -(gpx * u + gpy * v) * iw2
                        a00 += gx0 * point0
                        a01 += gx0 * point1
                        a02 += gx0 * point2
                        a03 += gx0
                        a10 += gy0 * point0
                        a11 += gy0 * point1
                        a12 += gy0 * point2
                        a13 += gy0
                        a20 += hx * point0
                        a21 += hx * point1
                        a22 += hx * point2
                        a23 += hx
            sm[0, tid] = a00; sm[1, tid] = a01; sm[2, tid] = a02; sm[3, tid] = a03
            sm[4, tid] = a10; sm[5, tid] = a11; sm[6, tid] = a12; sm[7, tid] = a13
            sm[8, tid] = a20; sm[9, tid] = a21; sm[10, tid] = a22; sm[11, tid] = a23
            cuda.syncthreads()
            # TREE reduction, all 12 components at once. The obvious "threads 0..11 each sum
            # their own row" costs 12 threads x _TPB serial adds per block per view and was
            # measured to leave the backward at 6x the forward; log2(_TPB) parallel halvings
            # put it where the arithmetic says it should be (2 interpolations vs the forward's 1).
            half = _TPB // 2
            while half > 0:
                if tid < half:
                    for c in range(12):
                        sm[c, tid] += sm[c, tid + half]
                cuda.syncthreads()
                half //= 2
            if tid < 12:
                cuda.atomic.add(proj_matrix_error, (a, tid // 4, tid % 4), sm[tid, 0])
            cuda.syncthreads()

    _kernels.update(forward=forward_loop, backward=backward_loop, cuda=cuda)
    return _kernels


class FastConeBackprojector(Function):
    """Same signature and same semantics as `DifferentiableConeBeamBackprojector`."""

    @staticmethod
    def forward(ctx, sinogram, projection_matrices, geometry):
        projection_matrices = projection_matrices.detach()
        k = _build()
        cuda = k["cuda"]
        reco = torch.zeros(tuple(geometry.volume_shape), device="cuda", dtype=torch.float32)
        k["forward"][_BLOCKS, _THREADS](
            cuda.as_cuda_array(sinogram.contiguous()),
            cuda.as_cuda_array(projection_matrices.contiguous()),
            cuda.as_cuda_array(reco),
            geometry.volume_shape, geometry.volume_spacing, geometry.volume_origin)
        ctx.save_for_backward(sinogram, projection_matrices)
        ctx.geometry = geometry
        return reco

    @staticmethod
    @once_differentiable
    def backward(ctx, volume_error):
        sinogram, projection_matrices = ctx.saved_tensors
        geometry = ctx.geometry
        k = _build()
        cuda = k["cuda"]
        pme = torch.zeros((sinogram.shape[0], 3, 4), device="cuda", dtype=torch.float32)
        sdx, sdy = torch.gradient(sinogram, dim=(2, 1))          # theirs, unchanged
        k["backward"][_BLOCKS, _THREADS](
            geometry.volume_shape, geometry.volume_spacing, geometry.volume_origin,
            cuda.as_cuda_array(projection_matrices.contiguous()),
            cuda.as_cuda_array(sdx.contiguous()), cuda.as_cuda_array(sdy.contiguous()),
            cuda.as_cuda_array(volume_error.contiguous()),
            cuda.as_cuda_array(pme))
        return None, pme, None


def backprojector(fast: bool | None = None):
    """The autograd Function class in use. `fast=None` reads the module flag / env var."""
    use = FAST if fast is None else bool(fast)
    return FastConeBackprojector if use else cone_backprojector()
