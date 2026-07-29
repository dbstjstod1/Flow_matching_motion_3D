"""The FDK, and the two entry points that route the projector pair to LEAP.

THE PROJECTOR PAIR IS LEAP'S (user decision 2026-07-29). `forward_project_3d_batched` and
`adjoint_project_3d_batched` are thin wrappers over `fm3d/leap_projector.py` -- read that
module for the model, the measured agreement, the VD-vs-SF adjoint choice and the LEAP API
traps. There is no backend switch and no environment variable: one operator, one path.

WHAT STAYS OURS, AND WHY:
  * THE FDK **ALGORITHM**, below -- cosine/Wang/Ohnesorge pre-weights, our ramp windows, the
    Voronoi per-view angular weight read out of `Pmat`, per-view rigid-motion geometry. LEAP's
    own `fbp` has none of those and REFUSES tilted modular panels ("FBP only implemented for
    modular geometries whose rowVectors are aligned with the z-axis"), which is precisely the
    motion-compensated case. Since 2026-07-30 the backprojection OPERATOR under the algorithm
    is LEAP's modular VD backprojector (`leap_projector.leap_fdk_backproject`, which folds
    LEAP's geometric ray weight back to our exact 1/w^2 convention; parity vs the retired
    fused kernel 3.1e-3 / ls-scale 1.000000, and faster). FM3D_FDK_LEAP=0 forces the torch
    reference loop. The bridge's analytic s-TANGENT stays on our fused kernel
    (`triton_backproject.backproject_tangent`) -- LEAP has no tangent -- while the tangent
    path's VALUE also comes from LEAP, keeping x(1) consistent with the static anchor.
  * `d(loss)/dP`, in `triton_leap_grad.leap_grad_P`. LEAP has no geometry derivative at all
    (`leaptorch`'s backward returns the volume gradient and `None` for everything else), so
    the motion estimator could never run on it. `LEAPProject` takes its value and its volume
    gradient from LEAP and its geometry gradient from our kernel -- the EXACT gradient of the
    Joseph forward LEAP is pinned to (`leap_projector.FORCE_JOSEPH`), so one operator and one
    gradient serve every geometry.
  * The ray-march `grid_sample` REFERENCE, `reference_project_3d_batched` below. It is not a
    backend any more -- nothing in production can reach it -- but it is the gates' independent
    oracle: exact autograd by construction, in a third codebase, which is what lets a gate
    catch LEAP and our SF kernel being wrong in the same way.

FDK (Feldkamp) reconstruction, kept geometrically consistent with the forward model:
  1. cosine pre-weight   g~ = g * SDD / sqrt(SDD^2 + u^2 + v^2)
  2. ramp filter along the detector u-axis (FFT, optional Hann apodization)
  3. distance-weighted backprojection  += g_filt(u*, v*) / w^2
using the SAME P matrices (u = u_h/w, v = v_h/w, w = along-axis source
distance), no convention drift.
The FDK is SELF-NORMALIZED by the geometry constant SOD*SDD/2 (`_fdk_physical_norm`),
so it returns absolute mu [1/mm]; no constant is ever fitted (2026-07-28).

Memory note (prompt 6, problem #1 of 2): the FDK is view-chunked (`view_chunk`) and
voxel-chunked (`vox_chunk`); this is the "view chunking" half of the 3D memory strategy. The
FM prior's volume-size problem is the OTHER half and is solved by 3D patches (prior_patch.py).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from . import triton_backproject
from .geometry_3d import ConeBeam3DConfig
from .filters import DEFAULT_RAMP_WINDOW, ramp_filter, calibrate_scale  # ramp; calibrate_scale is gate-only now
from .leap_projector import leap_backproject_3d_batched, leap_project_3d_batched

__all__ = [
    "forward_project_3d_batched",
    "adjoint_project_3d_batched",
    "reference_project_3d_batched",
    "reference_adjoint_3d_batched",
    "_fdk_physical_norm",
    "fdk_conebeam_3d_batched",
    "fdk_conebeam_3d_tangent",
    "wang_weight",
    "calibrate_scale",
]


def _ray_box_intersect_3d(o, d, box_min, box_max, eps_dir: float = 1e-8):
    """Slab-method ray/box intersection in 3D. o, d: (R, 3); box_*: (3,)."""
    d_safe = torch.where(d.abs() < eps_dir, torch.full_like(d, eps_dir), d)
    inv_d = 1.0 / d_safe
    t0 = (box_min[None, :] - o) * inv_d
    t1 = (box_max[None, :] - o) * inv_d
    t_small = torch.minimum(t0, t1)
    t_big = torch.maximum(t0, t1)
    tmin = torch.max(t_small, dim=-1).values
    tmax = torch.min(t_big, dim=-1).values
    hit = tmax > tmin
    return tmin, tmax, hit


# ---------------------------------------------------------------------------------------
_RAMP_WINDOW_NOTE = """WHY THE FDK RAMP IS APODIZED (`filters.DEFAULT_RAMP_WINDOW =
"shepphann"`), CHANGED FROM "ramlak" 2026-07-29.

The forward operator's VOXEL BASIS decides how much of the ramp is safe to use, and ours changed
when the projector did. The retired ray-march/gridsample forward integrated a TRILINEAR-TENT
object (grid_sample bilinear), which is band-limited near the voxel Nyquist -- so an unapodized
ramlak ramp had nothing spurious left to amplify, and "noiseless simulated data, so use the
sharpest filter" was a sound argument. Its successor, our SF pair, integrated the CUBE
(piecewise-constant) basis that separable-footprint is defined for -- and so does LEAP's own SF
kernel, which is why this section was written. (Since the 2026-07-30 Joseph pin the deployed
forward is a ray-driven TENT-basis model again, so the cube-basis texture below is history, not
a live constraint: measured, the 1 mm ram-lak texture fell 18-21 -> 3.5-5.0 HU. The apodization
argument is kept because it is what the ramp default rests on.)
A piecewise-constant object carries real spectral energy
ABOVE the sampling Nyquist -- the voxel faces -- and our detector resolves it (0.64 mm pitch is
0.42 mm at isocenter against 1 mm voxels), so ramlak reconstructs the voxel grid itself as a
crosshatch texture.

MEASURED on CQ500 val p216, static FDK, excess sd over the GT in a homogeneous brain ROI plus
bone-edge gradient magnitude as the sharpness proxy (scripts/diag_static_fdk.py):

    forward / filter                     excess sd     bone-edge sharpness   LEAP equivalent
    SF   1mm   + ramlak                    13.2 HU          99% of GT         ord12
    SF   1mm   + shepp                     10.3 HU          97%               ord2 (LEAP DEFAULT)
    SF   1mm   + hann                       1.4 HU          90%               ord12 + lowpass 2.0
    SF   1mm   + shepphann  (DEPLOYED)      0.0 HU          90%               ord2  + lowpass 2.0
    SF   1mm   + hann @ cutoff 0.419        0.0 HU          75%               (voxel-Nyquist cut)
    SF   0.5mm upsampled + ramlak           0.0 HU          82%   (4x cost)
    tent(ray)  + ramlak                     3.4 HU          90%               <- the old regime

`shepphann` DOMINATES every alternative examined: 0 texture at the same 90% sharpness the retired
trilinear-basis operator gave, for free. Three routes were measured and rejected:
  * LEAP's own default (`shepp` / ord2) alone leaves 10.3 HU -- it does not fix this. LEAP does not
    enable its low-pass by default because at ITS native voxel (du*SOD/SDD, here 0.4187 mm) the
    detector does not over-resolve the grid; our 1 mm grid is 2.39x coarser, which is what creates
    the problem.
  * cutting exactly at the voxel Nyquist (cutoff 0.419) costs 15 more points of sharpness and buys
    nothing -- the texture lives in a narrow near-Nyquist band, not uniformly above it.
  * FORWARD-PROJECTING A FINER GRID (upsample to 0.5 mm, project, reconstruct at 1 mm) -- the
    "match the voxel to the data sampling" route -- is 4x the cost AND blurrier (82%), because our
    object is only KNOWN at 1 mm: upsampling adds no information and trilinear interpolation is a
    broad-band low-pass (sinc^2 per axis), so it attenuates mid frequencies a designed filter keeps.

CAVEAT ON THE SHARPNESS COLUMN: the denominator is the GT volume, and that GT is itself aliased --
CQ500 is resampled 0.53 -> 1 mm with `sitkLinear` and no anti-alias filter, leaving 7-13x excess
energy in the top decile of its own band (measured against a Gaussian-prefiltered downsample). So
"99% of GT" for ramlak partly rewards REPRODUCING GT aliasing. The GT is taken as given (user's
call); quote this column with the caveat attached.

`ramlak` remains available and is still the right choice if the forward is ever put back on a
band-limited basis.

REFUTED alternative, do not retry: widening the SF footprint to the tent's SUPPORT. Non-integer
widths beat against the voxel pitch (443 HU), and the exact double width only reaches 9.2 HU
while costing 30% of the edge sharpness -- it is the tent's piecewise-CUBIC shape that matters,
not its support."""


def _world_to_grid_norm_3d(pts, *, W, H, D, dx, dy, dz, X0, Y0, Z0,
                           align_corners: bool):
    """World (x, y, z) -> 5D grid_sample normalized coords in [-1, 1].

    Returns (..., 3) ordered (x_norm, y_norm, z_norm) as grid_sample expects
    (x indexes W, y indexes H, z indexes D).

    REFERENCE implementation. The forward projector does NOT call this on its hot path --
    it uses `_grid_affine_3d` + one fused `addcmul` instead (22x faster, see below). Kept
    because it is the readable statement of the convention that `warp.py` and
    `forward_projector_2d.py` must agree with, and the fused path is checked against it."""
    ix = (pts[..., 0] - X0) / dx - 0.5
    iy = (pts[..., 1] - Y0) / dy - 0.5
    iz = (pts[..., 2] - Z0) / dz - 0.5
    if align_corners:
        x_norm = 2.0 * ix / max(W - 1, 1) - 1.0
        y_norm = 2.0 * iy / max(H - 1, 1) - 1.0
        z_norm = 2.0 * iz / max(D - 1, 1) - 1.0
    else:
        x_norm = (2.0 * ix + 1.0) / float(W) - 1.0
        y_norm = (2.0 * iy + 1.0) / float(H) - 1.0
        z_norm = (2.0 * iz + 1.0) / float(D) - 1.0
    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def _grid_affine_3d(*, W, H, D, dx, dy, dz, X0, Y0, Z0, align_corners: bool, device):
    """(scale, shift) with  grid_norm = world * scale - shift,  per axis (x, y, z).

    `_world_to_grid_norm_3d` is AFFINE in the world point, so it collapses. With
    align_corners=False the -0.5 (voxel-centre) and the +1 (of (2i+1)/W) cancel exactly:

        x_norm = (2*((px - X0)/dx - 0.5) + 1)/W - 1 = px * 2/(dx*W) - (2*X0/(dx*W) + 1)

    Composing with the ray `p(t) = o + d*t` and `t = tmin + step*k` makes the whole sample
    grid one `addcmul` over per-RAY constants -- no (rays, n_samples, 3) `pts` tensor, no
    per-axis slicing, no `stack`. Measured on (2 views x 96 rows x 512 cols x 384 samples):
    grid construction 16.62 ms -> 0.76 ms (22x), peak 3053 -> 1616 MiB, and the whole block
    18.30 -> 2.44 ms (7.5x). The grid agrees with the reference to 2.1e-6 in [-1,1] units
    (= 2.7e-4 voxel) and the sampled line integrals to 2.1e-5 relative -- both far below the
    n_samples=384 quadrature error (5.1e-5), i.e. this is float reassociation, not a model
    change.

    WHY THIS IS THE BOTTLENECK AT ALL: profiling says `grid_sample` (the actual trilinear
    interpolation) is only 8% of a block; 92% went into materializing and re-reading the
    coordinate tensor. A torch-native ray marcher is memory-bandwidth-bound on coordinates,
    not compute-bound on interpolation."""
    if align_corners:
        sx, sy, sz = 2.0 / (dx * max(W - 1, 1)), 2.0 / (dy * max(H - 1, 1)), 2.0 / (dz * max(D - 1, 1))
        cx = 2.0 * (X0 / dx + 0.5) / max(W - 1, 1) + 1.0
        cy = 2.0 * (Y0 / dy + 0.5) / max(H - 1, 1) + 1.0
        cz = 2.0 * (Z0 / dz + 0.5) / max(D - 1, 1) + 1.0
    else:
        sx, sy, sz = 2.0 / (dx * W), 2.0 / (dy * H), 2.0 / (dz * D)
        cx, cy, cz = 2.0 * X0 / (dx * W) + 1.0, 2.0 * Y0 / (dy * H) + 1.0, 2.0 * Z0 / (dz * D) + 1.0
    scale = torch.tensor([sx, sy, sz], device=device, dtype=torch.float32)
    shift = torch.tensor([cx, cy, cz], device=device, dtype=torch.float32)
    return scale, shift


def forward_project_3d_batched(
    volumes: torch.Tensor,    # (B, 1, D, H, W)
    Pmat: torch.Tensor,       # (B, V, 3, 4)  per-volume, per-view projection matrices
    u_coords: torch.Tensor,   # (nu,) physical detector coords [mm], lateral
    v_coords: torch.Tensor,   # (nv,) physical detector coords [mm], axial (z)
    *,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """THE forward operator: LEAP modular-beam. (B,V,nv,nu) [mm * mu].

    Differentiable in the volume (LEAP's backprojection) and in `Pmat` (the exact gradient
    of LEAP's pinned Joseph kernel); `fm3d/leap_projector.py` is where both decisions are
    argued and gated. There is
    no backend argument: the volume box is the centred grid the whole repo uses, the tensors
    must be on a GPU, and that is the only configuration that exists.
    """
    if volumes.ndim != 5:
        raise ValueError("volumes must be (B, 1, D, H, W)")
    if not volumes.is_cuda:
        raise ValueError("the projector pair is LEAP's and runs on the GPU only; "
                         "`reference_project_3d_batched` serves CPU tensors, for gates")
    return leap_project_3d_batched(volumes, Pmat, u_coords, v_coords, dx=dx, dy=dy, dz=dz)


def reference_project_3d_batched(
    volumes: torch.Tensor,    # (B, 1, D, H, W)
    Pmat: torch.Tensor,       # (B, V, 3, 4)
    u_coords: torch.Tensor,
    v_coords: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dz: float,
    X0: float | None = None,
    Y0: float | None = None,
    Z0: float | None = None,
    n_samples: int = 256,
    view_chunk: int = 4,
    row_chunk: int | None = None,
    align_corners: bool = False,
    reg: float = 1e-8,
) -> torch.Tensor:
    """GATE-ONLY reference: torch `grid_sample` ray march, trilinear basis, exact autograd.

    NOT a backend and NOT reachable from any production path (that was removed 2026-07-29 with
    the switch to LEAP). Its whole job is to be a THIRD, independent implementation so a gate
    can tell "LEAP and our SF kernel agree" apart from "LEAP and our SF kernel are wrong the
    same way". It also serves CPU tensors, `align_corners=True` and non-centred boxes, none of
    which the deployed operator covers. `n_samples`/`view_chunk`/`row_chunk` parameterize this
    path and nothing else.
    """
    if volumes.ndim != 5:
        raise ValueError("volumes must be (B, 1, D, H, W)")
    vol = volumes.to(torch.float32)
    device = vol.device
    B, _, D, H, W = vol.shape

    if X0 is None:
        X0 = -0.5 * W * dx
    if Y0 is None:
        Y0 = -0.5 * H * dy
    if Z0 is None:
        Z0 = -0.5 * D * dz

    Pmat = Pmat.to(device=device, dtype=torch.float32)
    u_coords = u_coords.to(device=device, dtype=torch.float32)
    v_coords = v_coords.to(device=device, dtype=torch.float32)
    V = Pmat.shape[1]
    nu = u_coords.shape[0]
    nv = v_coords.shape[0]

    box_min = torch.tensor([X0, Y0, Z0], device=device, dtype=torch.float32)
    box_max = torch.tensor([X0 + dx * W, Y0 + dy * H, Z0 + dz * D],
                           device=device, dtype=torch.float32)
    I3 = torch.eye(3, device=device, dtype=torch.float32)
    ks = torch.arange(n_samples, device=device, dtype=torch.float32) + 0.5

    # Homogeneous detector directions [u; v; 1] for every element (nv, nu, 3);
    # sinogram layout is (nv, nu): row = v (axial), column = u (lateral).
    uu = u_coords[None, :].expand(nv, nu)
    vv = v_coords[:, None].expand(nv, nu)
    uh = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)          # (nv, nu, 3)

    rc = nv if row_chunk is None else int(row_chunk)
    outs_v = []
    g_scale, g_shift = _grid_affine_3d(W=W, H=H, D=D, dx=dx, dy=dy, dz=dz,
                                       X0=X0, Y0=Y0, Z0=Z0,
                                       align_corners=align_corners, device=device)

    for v0 in range(0, V, view_chunk):
        v1 = min(v0 + view_chunk, V)
        vc = v1 - v0
        P = Pmat[:, v0:v1]                        # (B, vc, 3, 4)
        A = P[..., :3] + reg * I3                 # (B, vc, 3, 3)
        b = P[..., 3]                             # (B, vc, 3)
        A_inv = torch.linalg.inv(A)               # (B, vc, 3, 3)
        src = -(A_inv @ b.unsqueeze(-1)).squeeze(-1)                 # (B, vc, 3)

        rows = []
        for r0 in range(0, nv, rc):
            r1 = min(r0 + rc, nv)
            nr = r1 - r0
            # Ray directions must stay fp32 even under an enclosing torch.autocast: einsum is
            # on autocast's fp16 cast list (linalg.inv is not), and fp16 directions cost ~1e-3
            # relative = up to ~0.25 voxel of sample position over a head-sized path, silently.
            # `enabled=False` is an exact no-op when no autocast is active.
            with torch.autocast(device_type=device.type, enabled=False):
                dirs = torch.einsum("bvij,mnj->bvmni", A_inv, uh[r0:r1])  # (B,vc,nr,nu,3)
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)

            o = src[:, :, None, None, :].expand(B, vc, nr, nu, 3)
            tmin, tmax, hit = _ray_box_intersect_3d(
                o.reshape(-1, 3), dirs.reshape(-1, 3), box_min, box_max)
            tmin = tmin.view(B, vc, nr, nu)
            tmax = tmax.view(B, vc, nr, nu)
            hit = hit.view(B, vc, nr, nu)

            step = (tmax - tmin).clamp_min(0.0) / float(n_samples)

            # Fused sample grid: grid_norm(k) = A + Bk*k, both per-RAY (see _grid_affine_3d).
            # A = (o + d*tmin) * scale - shift ;  Bk = (d * scale) * step
            Bd = dirs * g_scale                                       # (B,vc,nr,nu,3)
            A = torch.addcmul(o * g_scale - g_shift, Bd, tmin[..., None])
            Bk = Bd * step[..., None]
            grid = torch.addcmul(A.unsqueeze(-2), Bk.unsqueeze(-2), ks[:, None])
            grid = grid.reshape(B, vc * nr, nu, n_samples, 3)

            samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=align_corners)
            samp = samp[:, 0].view(B, vc, nr, nu, n_samples)

            integral = samp.sum(dim=-1) * step                       # (B, vc, nr, nu)
            rows.append(torch.where(hit, integral, torch.zeros_like(integral)))
        outs_v.append(torch.cat(rows, dim=2))                        # (B, vc, nv, nu)

    return torch.cat(outs_v, dim=1)                                  # (B, V, nv, nu)


def adjoint_project_3d_batched(
    sino: torch.Tensor,       # (B, V, nv, nu)
    Pmat: torch.Tensor,       # (B, V, 3, 4)
    u_coords: torch.Tensor,
    v_coords: torch.Tensor,
    *,
    D: int,
    H: int,
    W: int,
    dx: float,
    dy: float,
    dz: float,
) -> torch.Tensor:
    """A^T sino: LEAP's backprojection. (B,1,D,H,W).

    NOT the exact transpose of `forward_project_3d_batched` any more, and that is a deliberate
    trade the whole repo now lives with: `leap_projector.ADJOINT_MODE` selects LEAP's
    voxel-driven backprojector (self-adjointness 1.9e-4; 'SF' would buy 1.9e-7 at ~2.8x the
    cost, the retired matched pair had 0.0). Every `A^T` in the loop -- the CG data step
    included -- goes through here.
    """
    if sino.ndim != 4:
        raise ValueError("sino must be (B, V, nv, nu)")
    if not sino.is_cuda:
        raise ValueError("the projector pair is LEAP's and runs on the GPU only; "
                         "`reference_adjoint_3d_batched` serves CPU tensors, for gates")
    return leap_backproject_3d_batched(sino, Pmat, u_coords, v_coords,
                                       D=D, H=H, W=W, dx=dx, dy=dy, dz=dz)


def reference_adjoint_3d_batched(
    sino: torch.Tensor,
    Pmat: torch.Tensor,
    u_coords: torch.Tensor,
    v_coords: torch.Tensor,
    *,
    D: int,
    H: int,
    W: int,
    dx: float,
    dy: float,
    dz: float,
    X0: float | None = None,
    Y0: float | None = None,
    Z0: float | None = None,
    n_samples: int = 256,
    view_chunk: int = 4,
    row_chunk: int | None = None,
    reg: float = 1e-8,
) -> torch.Tensor:
    """GATE-ONLY: the EXACT transpose of `reference_project_3d_batched`, by autograd
    (differentiate `<A(x), s>` at `x = 0`). Exact by construction, which is the point."""
    s = sino.to(torch.float32)
    B = s.shape[0]
    x = torch.zeros((B, 1, D, H, W), device=s.device, dtype=torch.float32,
                    requires_grad=True)
    out = reference_project_3d_batched(
        x, Pmat, u_coords, v_coords, dx=dx, dy=dy, dz=dz, X0=X0, Y0=Y0, Z0=Z0,
        n_samples=n_samples, view_chunk=view_chunk, row_chunk=row_chunk, reg=reg)
    (out * s).sum().backward()
    return x.grad


def wang_weight(u_coords: torch.Tensor, cfg) -> torch.Tensor | None:
    """Displaced-detector redundancy weight (Wang, Med Phys 29(7):1634, 2002). (nu,) or None.

    In half-fan the panel is pushed laterally so each view images somewhat more than half the
    object; the opposite view supplies the rest. Over a FULL 360 deg orbit the rays with
    |u| < d (d = the panel's overhang past the central ray) are therefore acquired TWICE -- once
    at beta, once at its conjugate near beta+pi -- while everything beyond d is acquired once.
    Backprojecting all of it with a uniform 2*pi/V weight would double-count the centre.

        w(u) = 0                              t <= -1
             = sin^2( pi/4 * (1 + t) )        -1 < t < 1        t = sign(u0) * u / d
             = 1                              t >= 1

    `w(u) + w(-u) = 1` identically, so a conjugate pair sums to exactly one contribution, and
    `w` falls SMOOTHLY to 0 at the panel's SHORT edge.

    That last property used to be read as "half-fan therefore needs no truncation extrapolation".
    It does not follow, and it was WRONG on real data. Wang tapers the short edge; nothing tapers
    the LONG one, where a real patient (on a couch, against a projection that never decays to
    zero) is still cut off. MEASURED on SPARE-MC against their own RTK reconstruction of the same
    projections: turning the pad on moves rel-rmse 0.1030 -> 0.0947. See `ohnesorge_pad`.

    Returns None for a centred panel (full-fan), where the weight would be a no-op only if
    applied as identity -- so we skip it rather than multiply by a bump function.
    """
    if not getattr(cfg, "is_half_fan", False):
        return None
    d = cfg.overlap_half_width_mm()
    s = 1.0 if cfg.det_offset_u_mm > 0 else -1.0
    t = (s * u_coords / d).clamp(-1.0, 1.0)
    return torch.sin(0.25 * torch.pi * (1.0 + t)) ** 2


def ohnesorge_pad(g: torch.Tensor, npad: int, left: bool = True,
                  right: bool = True) -> torch.Tensor:
    """Lateral truncation extrapolation, TRANSCRIBED FROM RTK (`rtkFFTProjectionsConvolution
    ImageFilter.hxx`, `PadInputImageRegion` + `UpdateTruncationMirrorWeights`), which cites
    Ohnesorge et al., Med Phys 27(1):39, 2000, eq. 3a/3b. `g` is (..., nu); returns (..., nu+2*npad).

    The ramp is NON-LOCAL, so a projection that does not decay to zero at the panel edge presents
    a STEP to the |f| filter, and the ramp differentiates that step into a low-frequency
    cupping/halo spread across the whole reconstruction. Extrapolating the projection outward
    removes the step.

    THE EXTENSION IS A POINT REFLECTION THROUGH THE BORDER SAMPLE, not a copy of it:

        right:  g[nu-1 + d] = w[d] * ( 2*S_E - g[nu-1 - d] ),   S_E = g[nu-1]
        left:   g[   0 - d] = w[d] * ( 2*S_A - g[     0 + d] ), S_A = g[1]
        w[d]  = sin( (npad - d) * pi / (2*npad - 2) ) ** 0.75           d = 0..npad

    so the extension CONTINUES THE SLOPE of the data (2*S_E - g[edge-d] rises if g was falling
    into the edge) and then decays to zero under `w`. Our previous version copied the edge VALUE
    under a raised-cosine taper, which flat-lines the slope and leaves a first-order kink -- and
    it was disabled entirely for half-fan.

    `S_A = g[1]`, not `g[0]`, is RTK's own indexing (`iidx[0] = leftRegion.GetIndex(0) +
    leftRegion.GetSize(0) + 1`). It looks like an off-by-one against the right-hand branch, but it
    is what RTK ships and what SPARE's reference reconstructions were made with, so it is
    reproduced rather than "fixed". It moves one column of 512.

    `npad` is RTK's truncation extent in COLUMNS = `floor(TruncationCorrection * nu)`, capped by
    the available zero-pad (RTK's `m_ZeroPadFactors = 2` leaves nu/2 on each side).

    ONLY EXTRAPOLATE A BORDER THAT IS A REAL PANEL EDGE. `left`/`right` exist because a point
    reflection through a border whose value is an ARTIFICIAL ZERO evaluates to `-g[edge-d]`: it
    injects a negative mirror of the projection. That is what the half-fan symmetric enlargement
    leaves on one side, and padding it blindly made SPARE-MC monotonically WORSE (rel-rmse 0.1030
    -> 0.1249 at npad=256). The caller knows which side was enlarged; it must say so.

    `clamp_min(0)` IS WHAT MAKES THIS SAFE ON UNTRUNCATED DATA, and it is not a hack: `g` is a
    (weighted) LINE INTEGRAL, which cannot be negative. Ohnesorge's reflection presupposes
    truncation -- where the border sample S is large because the object was cut through. Where the
    object FITS the panel, S ~ 0 and the raw formula returns `-g[edge-d]`, a negative mirror, i.e.
    pure damage; that cost `gate_shepp_halffan` 10/10 -> 2/10. Clamping sends exactly that case to
    zero, so an untruncated projection is zero-extended (a no-op) and a truncated one is
    extrapolated. The alternative -- deciding per call from `g.abs().amax()` -- looks equivalent
    and is NOT: `moco._continuous_backproject_3d` filters the sinogram in VIEW CHUNKS, so a
    batch-dependent decision flips between chunks and breaks the `phi=Id MC-FDK == static FDK`
    invariant (measured: 2.3e-7 -> 2.3e-1). The extension must be an ELEMENTWISE function of the
    projection, and this one is.
    """
    if npad <= 0 or not (left or right):
        return g
    nu = g.shape[-1]
    dev, dt = g.device, g.dtype
    d = torch.arange(0, npad + 1, device=dev, dtype=torch.float64)
    denom = max(2 * npad - 2, 1)
    w = torch.sin((npad - d) * torch.pi / denom).clamp_min(0.0) ** 0.75
    w = w.to(dt)

    dd = torch.arange(1, npad + 1, device=dev)                     # distance from the border
    out = [g]
    if right:
        S_E = g[..., nu - 1:nu]                                    # the right border sample
        out.append(w[dd] * (2.0 * S_E - g[..., (nu - 1) - dd]).clamp_min(0.0))
    if left:
        S_A = g[..., 1:2]                                          # RTK's left reference (index 1)
        # d = 1..npad -> columns -1..-npad, so flip to store them inward-out
        out.insert(0, (w[dd] * (2.0 * S_A - g[..., dd]).clamp_min(0.0)).flip(-1))
    return torch.cat(out, dim=-1)


def _fdk_physical_norm(cfg) -> float:
    """The FDK unit constant. SELF-NORMALIZING: it is fixed by the GEOMETRY, which is given.

        SOD * SDD * (1.0 if half-fan else 0.5)

    With the cosine pre-weight, a ramp on the PHYSICAL detector frequency axis
    (rfftfreq(N, d=du)) and the projective 1/w^2 weight, this is the textbook Feldkamp
    constant that turns the angle-weighted view sum into absolute mu [1/mm]. SOD^2/2 is the
    (SOD/w)^2 magnification; the extra SDD/SOD is because the ramp is convolved on the
    detector axis (pitch du at SDD) rather than on the virtual iso-plane detector, where
    frequencies are M = SDD/SOD times larger. Full-fan 2pi measures every line twice -> 1/2;
    the half-fan Wang weight already de-duplicates conjugate rays, so it takes none.

    NO LEAST-SQUARES SCALE ANYWHERE (user, 2026-07-28, following the 4DCT sibling's
    2026-07-21 deletion of the same thing -- `fdct/projector_3d._fdk_physical_norm`). SOD and
    SDD are GIVEN, so there is nothing to fit, and fitting was actively harmful: a scalar
    regressed on FDK(A(v)) vs v silently absorbs any operator or physics-level error (units,
    voxel size, detector pitch, and -- in the sibling's case -- scatter/bowtie/air constants),
    leaving the whole pipeline self-consistent and wrong. `calibrate_scale` survives only as a
    gate diagnostic, where it must now come out ~1.0.

    VERIFIED on this geometry (uniform water cylinder, no fitting): mu = 0.01996 vs 0.02."""
    return float(cfg.SOD) * float(cfg.SDD) * (1.0 if getattr(cfg, "is_half_fan", False) else 0.5)


def fdk_conebeam_3d_batched(
    sino: torch.Tensor,       # (B, V, nv, nu)
    Pmat: torch.Tensor,       # (B, V, 3, 4) nominal geometry used for recon
    u_coords: torch.Tensor,   # (nu,)
    v_coords: torch.Tensor,   # (nv,)
    cfg: ConeBeam3DConfig,
    *,
    D: int,
    H: int,
    W: int,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    window: str = DEFAULT_RAMP_WINDOW,   # filters.py is the ONE source of truth
    cutoff: float = 1.0,
    scale: float | None = None,
    view_chunk: int = 8,
    vox_chunk: int = 2_000_000,
    eps: float = 1e-8,
    disp: torch.Tensor | None = None,
    trunc_pad: int | None = None,
    view_weight: torch.Tensor | None = None,
    _return_filtered: bool = False,
) -> torch.Tensor:
    """Batched FDK (Feldkamp) cone-beam reconstruction. Returns (B, D, H, W).

    `window`/`cutoff` control the ramp apodization (see `recon_2d.ramp_filter`).
    Default `ramlak` (plain ramp) -- `hann` costs substantial resolution and was the
    reason the early 3D reconstructions looked blurred.

    IMPORTANT (V_axis trick contract, same as `fbp_fanbeam_2d_batched`): the
    angular weight is `angle_span / sino.shape[1]`. Calling this with the view
    axis = 1 (each view on the batch axis) yields angle_span * SVBP_v, so a
    caller that averages the batch elements over the true V reproduces the
    standard FDK weight — this is what the continuous 3D MC-FBP relies on.

    `disp` (B, V, 3, D, H, W) VOXEL displacements = MOTION-COMPENSATED backprojection.
    Each reference voxel q is backprojected from where its material actually WAS at that
    view's acquisition time:

        r(q) = q - phi(q, tau_v)        (1st-order inverse of the pull map r -> r + phi(r))

    i.e. the DVF moves the backprojection COORDINATES, and the ray geometry (u, v, and the
    1/w^2 distance weight) is evaluated at the moved point r, not at q. This is the correct
    MC-FDK and it costs NO extra interpolation: the only resampling is the detector
    `grid_sample` that plain FDK already performs.

    Do NOT go back to `backproject-then-warp-the-volume` (FDK on the reference grid followed
    by `warp_volume(svbp, -phi)`). It applies the SAME first-order map but pays one trilinear
    VOLUME resample per view, and at |phi| ~ 1 voxel a trilinear resample destroys ~40% of the
    Laplacian energy -- measured. The blur then survives the view average.

    `disp=None` (or all-zero) reduces EXACTLY to the uncorrected static FDK: the moved coords
    are the voxel centers bit-for-bit.

    `view_weight` (V,) or (B,V) REPLACES the uniform `angle_span / V` angular weight with a
    per-view one, in radians. Use `geometry_3d.view_angular_weights(Pmat)` to get the angular
    share each view actually covers -- under rigid motion about the gantry axis the views stop
    being equiangular and the uniform weight is simply the wrong Riemann sum (worth -2.01 dB at
    5 deg; see that function). It is applied by PRE-SCALING the sinogram, which is exact: every
    step between here and the backprojection sum (cosine, Wang, Ohnesorge pad, ramp) is linear
    and acts view-by-view, so scaling view v by c multiplies its backprojected contribution by
    c. `None` keeps the old behaviour bit-for-bit, and on the nominal orbit the weights come
    back uniform, so passing them there is a no-op too.
    """
    device = sino.device
    dtype = torch.float32
    sino = sino.to(device=device, dtype=dtype)
    Pmat = Pmat.to(device=device, dtype=dtype)

    if view_weight is not None:
        w = view_weight.to(device=device, dtype=dtype)
        if w.ndim == 1:
            w = w[None]
        if w.shape[-1] != sino.shape[1]:
            raise ValueError(f"view_weight has {w.shape[-1]} views, sinogram has "
                             f"{sino.shape[1]}")
        # net weight per view = view_weight; the uniform factor below then cancels the divisor.
        uni = float(cfg.angle_span) / float(sino.shape[1])
        sino = sino * (w / uni)[..., None, None]
    u_coords = u_coords.to(device=device, dtype=dtype)
    v_coords = v_coords.to(device=device, dtype=dtype)
    B, V, nv, nu = sino.shape
    du = float(cfg.du)
    dv = float(cfg.dv)
    u0 = float(cfg.det_offset_u_mm)      # lateral panel offset (element coords are shifted)
    # NOT `v0`: that name is the view-chunk loop index further down, and shadowing it silently
    # replaces the detector's axial offset with a VIEW NUMBER inside the backprojection -- which
    # shifts the panel by tens of millimetres per chunk and smears the reconstruction into an
    # arc. It reads as a geometry bug and it cost an afternoon. (Caught by gate_shepp_halffan
    # dropping 10/10 -> 2/10 with an offset that was supposed to be a no-op at 0.)
    v_off = float(getattr(cfg, "det_offset_v_mm", 0.0))   # AXIAL panel offset (SPARE-MC: -2 mm)

    # 0) HALF-FAN: widen the projection to the SYMMETRIC range [-u_max, +u_max] with zeros.
    #
    # This is not cosmetic and it is not the same as `trunc_pad`. The ramp filter is NON-LOCAL,
    # so `ramp(cos * w * g)` has non-zero TAILS beyond the panel's short edge -- on the side the
    # displaced panel never covered. Those tails are real filtered signal and they backproject
    # into exactly the doubly-sampled core r < r_ov. Clipping them (which is what the
    # `|u - u0| <= half_u` mask does on the un-padded array) starves the core.
    # MEASURED on a uniform water cylinder, interior value relative to the periphery, Halcyon:
    #     overlap d (mm)      175     135      95      40
    #     un-padded         0.913   0.854   0.806   0.707     <- deficit grows as d shrinks
    #     symmetric pad     1.001   ...     ...     1.001     <- flat
    # This is what RTK's `rtkDisplacedDetectorImageFilter` does when it enlarges the projection.
    if cfg.is_half_fan:
        u_lo, u_hi = float(u_coords[0]), float(u_coords[-1])
        umax = max(abs(u_lo), abs(u_hi))
        nu_p = 2 * int(math.ceil(umax / du))
        u_p = (torch.arange(nu_p, device=device, dtype=dtype) - (nu_p - 1) / 2.0) * du
        j0 = int(round((u_lo - float(u_p[0])) / du))
        if j0 < 0 or j0 + nu > nu_p:
            raise RuntimeError(f"half-fan symmetric pad misaligned: j0={j0}, nu={nu}, "
                               f"nu_p={nu_p}")
        pad_l, pad_r = j0, nu_p - nu - j0
        sino = F.pad(sino, (pad_l, pad_r))
        u_coords, nu, u0 = u_p, nu_p, 0.0
        # Which borders of the WIDENED array are still REAL panel edges? A side that got zeros is
        # not one, and must never be extrapolated (see `ohnesorge_pad`). On SPARE-MC the panel is
        # offset so hard that j0 = 0: the LEFT border IS the panel's long edge, at full Wang
        # weight, and the enlargement lands entirely on the right.
        real_l, real_r = (pad_l == 0), (pad_r == 0)
    else:
        real_l = real_r = True                       # full-fan: both borders are the panel

    # 1) cosine pre-weight (flat panel): SDD / sqrt(SDD^2 + u^2 + v^2).
    cos_w = cfg.SDD / torch.sqrt(
        cfg.SDD ** 2 + u_coords[None, :] ** 2 + v_coords[:, None] ** 2 + eps
    )                                                                # (nv, nu)
    g = sino * cos_w[None, None]

    # 1a) HALF-FAN redundancy weight (Wang 2002), between the cosine weight and the ramp. It is
    # a projection-domain weight, so it must precede the filter.
    w_wang = wang_weight(u_coords, cfg)
    if w_wang is not None:
        g = g * w_wang[None, None, None, :]

    # 1b) LATERAL TRUNCATION EXTRAPOLATION, before the ramp (Ohnesorge et al., Med Phys
    # 27(1):39, 2000; Hsieh's water-cylinder variant). `ramp_filter` zero-pads to >= 2*nu, so a
    # projection that does not decay to zero at the panel edge presents a step to the |f|
    # filter: the ramp differentiates it into a DC/cupping error that is spread over the whole
    # reconstruction. Our full-fan FOV is 262.6 mm and DIR-Lab case2's thorax is 297 mm wide,
    # so its rows genuinely run off the panel. Extending each row by its edge value under a
    # raised-cosine taper removes the step.
    # Where the object DOES fit, the edge value is 0 and the appended columns are 0 -- but this
    # is NOT bit-for-bit a no-op, because `ramp_filter` sizes its FFT as the next power of two
    # above 2*nu: widening nu changes N (e.g. 256 -> 512) and hence the ramp's discretization
    # and its circular wraparound. Measured on a non-truncated Shepp-Logan: 6.5e-4 relative.
    # The longer transform is the more accurate one. It cannot perturb the
    # `phi=Id MC-FDK == static FDK` gate (verified 2.3e-7): the same filtered sinogram feeds
    # both paths and `disp` never enters here.
    # Half-fan needs NO extrapolation: the Wang weight already tapers the short edge to zero,
    # and the object fits inside the (much larger) offset FOV so the long edge sees nothing.
    # Extrapolating a half-fan projection would invent data the conjugate view already supplies.
    # LATERAL TRUNCATION EXTRAPOLATION -- and it applies to HALF-FAN TOO. The old code disabled it
    # whenever the Wang weight was on, reasoning that "Wang tapers the short edge to zero and the
    # object fits inside the offset FOV". The first half is true; the second is not. Wang only
    # touches the SHORT edge. On the LONG edge -- which after the symmetric enlargement is the
    # array's own border, at Wang weight 1.0 -- the patient and the COUCH are still cut off.
    # MEASURED on SPARE-MC (`Proj/`, cos x Wang applied, mean over views and rows):
    #     col   0 (u=-346.5, the long edge)  0.1124     <- should be 0 for an untruncated view
    #     col   1                            0.1189     <- and still climbing inward
    #     peak                               3.7715
    # 3% of the peak, and the ramp is NON-LOCAL, so that step becomes a low-frequency cupping /
    # halo across the whole reconstruction -- which is exactly what we measured against SPARE's own
    # reconstruction (see docs and data/diag/spare_operator_*.png).
    #
    # Extrapolate every REAL panel edge (`real_l`/`real_r`); `ohnesorge_pad`'s clamp makes it a
    # NO-OP wherever the projection was not actually truncated, so this needs no data-dependent
    # branch -- and it must not have one: `moco._continuous_backproject_3d` filters in view chunks,
    # so any decision taken from `g.amax()` flips between chunks and breaks the
    # `phi=Id MC-FDK == static FDK` invariant.
    npad = (nu // 4) if trunc_pad is None else int(trunc_pad)
    pad_l = 0                       # how many columns were prepended -- the crop offset below
    if npad > 0:
        g = ohnesorge_pad(g, npad, left=real_l, right=real_r)
        pad_l = npad if real_l else 0          # the sides are padded INDEPENDENTLY
    nu_pad = g.shape[-1]

    # 2) ramp filter along u (row-wise; FDK filters only the lateral axis).
    # CHUNKED OVER VIEWS. `ramp_filter` runs an rfft of length >= 2*nu_pad in complex64, so a
    # whole (B,V,nv,nu_pad) transform is ~4x the sinogram. Half-fan's symmetric padding nearly
    # doubles nu, which pushed det_bin=1 (nu 1280 -> 2320, N=8192) to a 27 GiB allocation.
    # Filtering is row-independent, so chunking is exact.
    out = torch.empty((B, V, nv, nu), device=device, dtype=dtype)
    fchunk = max(1, min(V, int(view_chunk)))
    for v0 in range(0, V, fchunk):
        v1 = min(v0 + fchunk, V)
        blk = ramp_filter(g[:, v0:v1].reshape(-1, nu_pad), du=du, window=window,
                          cutoff=cutoff).view(B, v1 - v0, nv, nu_pad)
        out[:, v0:v1] = blk[..., pad_l:pad_l + nu]      # pad_l, NOT npad: the sides are separate
    g = out

    if _return_filtered:
        # `fdk_conebeam_3d_tangent`'s single-source front-end: the fully filtered sinogram in
        # the FINAL element coordinate system (half-fan may have widened nu and zeroed u0).
        return g, u0

    # detector extent, in the SHIFTED element coordinate system (u0 = panel offset)
    half_u = 0.5 * nu * du
    half_v = 0.5 * nv * dv
    Npix = D * H * W

    # 3) distance-weighted backprojection -- THROUGH LEAP's modular VD backprojector
    # (2026-07-30, user decision: the FDK ALGORITHM -- cosine/Wang/Ohnesorge/ramp/Voronoi,
    # per-view motion Pmat -- is ours; the OPERATOR under it is LEAP's, like every other
    # A/A^T in the repo). `leap_fdk_backproject` folds LEAP's geometric ray weight back to
    # our exact 1/w^2 convention (its docstring has the algebra; parity vs the retired
    # fused kernel 3.1e-3 rel / corr 0.999995 / ls-scale 1.000000, and it is FASTER).
    # NOTE: LEAP's own `fbp` cannot do this job -- it refuses modular geometries whose
    # panels tilt past axial alignment, which is precisely the motion-compensated case.
    needs_grad = torch.is_grad_enabled() and (g.requires_grad or Pmat.requires_grad)
    if disp is None and not needs_grad and g.is_cuda \
            and os.environ.get("FM3D_FDK_LEAP", "1") != "0":
        from .leap_projector import leap_fdk_backproject
        recon = leap_fdk_backproject(
            g, Pmat, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
            u0=u0, v_off=v_off).view(B, Npix)
    else:
        # torch loop: the differentiable / disp / CPU / FM3D_FDK_LEAP=0 reference path.
        recon = _backproject_static_torch(
            g, Pmat, disp, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
            u0=u0, v_off=v_off, half_u=half_u, half_v=half_v, eps=eps,
            view_chunk=view_chunk, vox_chunk=vox_chunk)

    recon = recon * (float(cfg.angle_span) / float(V))
    recon = recon.view(B, D, H, W)
    # scale=None => the deterministic geometry constant (self-normalized, absolute mu).
    # An explicit float overrides it; scale=1.0 gets the RAW sum, which is what
    # `calibrate_scale` diagnostics want.
    return recon * (_fdk_physical_norm(cfg) if scale is None else scale)


def _backproject_static_torch(g, Pmat, disp, *, D, H, W, dx, dy, dz, du, dv,
                              u0, v_off, half_u, half_v, eps, view_chunk, vox_chunk,
                              w2=True):
    """The original torch backprojection loop, chunked over views AND voxels. (B, Npix).

    Kept verbatim as (a) the `disp` MC path, which the Triton kernel does not cover, and
    (b) the FM3D_FDK_LEAP=0 fallback / gate reference for the deployed LEAP-backed path."""
    device, dtype = g.device, g.dtype
    B, V, nv, nu = g.shape

    # Voxel homogeneous world coords (centered at origin, matching projector).
    xs = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0) * dx
    ys = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0) * dy
    zs = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0) * dz
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")           # (D, H, W)
    Xh = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1),
         torch.ones(D * H * W, device=device, dtype=dtype)], dim=0)  # (4, Npix)
    Npix = D * H * W
    if disp is not None:
        if disp.shape[-3:] != (D, H, W) or disp.shape[-4] != 3:
            raise ValueError(f"disp must be (B,V,3,{D},{H},{W}), got {tuple(disp.shape)}")
        # (B,V,3,Npix) voxels -> mm, per axis. Channel order is [dx, dy, dz] (warp.py).
        disp = disp.to(device=device, dtype=dtype).reshape(*disp.shape[:2], 3, Npix)
        spac = torch.tensor([dx, dy, dz], device=device, dtype=dtype)[None, None, :, None]
        disp = disp * spac                                           # (B,V,3,Npix) mm

    recon = torch.zeros((B, Npix), device=device, dtype=dtype)
    for v0 in range(0, V, view_chunk):
        v1 = min(v0 + view_chunk, V)
        vc = v1 - v0
        P = Pmat[:, v0:v1]                                           # (B, vc, 3, 4)
        src = g[:, v0:v1].reshape(B * vc, 1, nv, nu)
        for p0 in range(0, Npix, vox_chunk):
            p1 = min(p0 + vox_chunk, Npix)
            if disp is None:
                uvw = torch.einsum("bvij,jp->bvip", P, Xh[:, p0:p1])  # (B, vc, 3, Np)
            else:
                # MOTION COMPENSATION: backproject reference voxel q from r = q - phi(q).
                d = disp[:, v0:v1, :, p0:p1]                          # (B, vc, 3, Np) mm
                Xc = Xh[None, None, :, p0:p1] - F.pad(d, (0, 0, 0, 1))
                uvw = torch.einsum("bvij,bvjp->bvip", P, Xc)          # (B, vc, 3, Np)
            u_h, v_h, w = uvw[:, :, 0], uvw[:, :, 1], uvw[:, :, 2]
            w_safe = torch.where(w.abs() < eps, torch.full_like(w, eps), w)
            u = u_h / w_safe
            v = v_h / w_safe

            # detector element index of the landing point (undo BOTH panel offsets)
            ju = (u - u0) / du + (nu - 1) / 2.0
            jv = (v - v_off) / dv + (nv - 1) / 2.0
            x_norm = (2.0 * ju + 1.0) / float(nu) - 1.0
            y_norm = (2.0 * jv + 1.0) / float(nv) - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1)             # (B, vc, Np, 2)

            val = F.grid_sample(
                src, grid.reshape(B * vc, 1, p1 - p0, 2), mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )[:, 0, 0]                                               # (B*vc, Np)
            val = val.view(B, vc, p1 - p0)

            mask = (w > 0) & ((u - u0).abs() <= half_u) & ((v - v_off).abs() <= half_v)
            wv = val / (w_safe ** 2) if w2 else val   # w2=False: plain (unmatched-A^T) accumulate
            contrib = torch.where(mask, wv, torch.zeros_like(val))
            recon[:, p0:p1] += contrib.sum(dim=1)
    return recon


def _backproject_tangent_torch(g, Pmat, Pdot, wgt, dwgt, *, D, H, W, dx, dy, dz, du, dv,
                               u0, v_off, half_u, half_v, eps,
                               view_chunk=8, vox_chunk=2_000_000):
    """Torch reference for `triton_backproject.backproject_tangent`, dtype-generic.

    Same math as the fused kernel, in whatever dtype `g` carries -- float64 is the point:
    scripts/gate_fdk_tangent.py uses this at float64 both to certify the tangent MATH against
    a float64 central difference of `_backproject_static_torch` (isolating the a.e.-Jacobian
    claim from fp32 noise) and as the reference the fp32 Triton kernel is compared to. Also
    the FM3D_FDK_TRITON=0 / no-triton fallback for `fdk_conebeam_3d_tangent`.

    Manual bilinear taps (no grid_sample): the tangent needs the interpolant's own ju/jv
    derivative, and grid_sampler ships no forward-AD rule -- the same reason the 2D sibling's
    `fbp_fanbeam_2d` JVP variant hand-rolls its detector interpolation."""
    device, dtype = g.device, g.dtype
    B, V, nv, nu = g.shape
    Pmat = Pmat.to(device=device, dtype=dtype)
    Pdot = Pdot.to(device=device, dtype=dtype)
    wgt = wgt.to(device=device, dtype=dtype).reshape(B, V)
    dwgt = dwgt.to(device=device, dtype=dtype).reshape(B, V)

    xs = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2.0) * dx
    ys = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2.0) * dy
    zs = (torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0) * dz
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    Xh = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), zz.reshape(-1),
         torch.ones(D * H * W, device=device, dtype=dtype)], dim=0)  # (4, Npix)
    Npix = D * H * W

    out = torch.zeros((B, Npix), device=device, dtype=dtype)
    dout = torch.zeros((B, Npix), device=device, dtype=dtype)
    for v0 in range(0, V, view_chunk):
        v1 = min(v0 + view_chunk, V)
        vc = v1 - v0
        P = Pmat[:, v0:v1]
        dP = Pdot[:, v0:v1]
        src = g[:, v0:v1].reshape(B * vc, nv * nu)
        wg = wgt[:, v0:v1, None]                                     # (B, vc, 1)
        dwg = dwgt[:, v0:v1, None]
        for p0 in range(0, Npix, vox_chunk):
            p1 = min(p0 + vox_chunk, Npix)
            uvw = torch.einsum("bvij,jp->bvip", P, Xh[:, p0:p1])     # (B, vc, 3, Np)
            duvw = torch.einsum("bvij,jp->bvip", dP, Xh[:, p0:p1])
            u_h, v_h, w = uvw[:, :, 0], uvw[:, :, 1], uvw[:, :, 2]
            du_h, dv_h, dw = duvw[:, :, 0], duvw[:, :, 1], duvw[:, :, 2]
            w_safe = torch.where(w.abs() < eps, torch.full_like(w, eps), w)
            u = u_h / w_safe
            v = v_h / w_safe
            udot = (du_h - u * dw) / w_safe
            vdot = (dv_h - v * dw) / w_safe
            ju = (u - u0) / du + (nu - 1) / 2.0
            jv = (v - v_off) / dv + (nv - 1) / 2.0

            # manual 4-tap bilinear: value + ju/jv derivative (cell frozen, a.e. exact)
            j0 = torch.floor(ju)
            i0 = torch.floor(jv)
            fx = (ju - j0).view(B * vc, -1)
            fy = (jv - i0).view(B * vc, -1)
            j0 = j0.long().view(B * vc, -1)
            i0 = i0.long().view(B * vc, -1)
            val = torch.zeros_like(fx)
            gu = torch.zeros_like(fx)
            gv = torch.zeros_like(fx)
            for cy in (0, 1):
                yc = i0 + cy
                wy = fy if cy == 1 else 1.0 - fy
                sy = 1.0 if cy == 1 else -1.0
                oky = (yc >= 0) & (yc < nv)
                for cx in (0, 1):
                    xc = j0 + cx
                    wx = fx if cx == 1 else 1.0 - fx
                    sx = 1.0 if cx == 1 else -1.0
                    ok = oky & (xc >= 0) & (xc < nu)
                    idx = yc.clamp(0, nv - 1) * nu + xc.clamp(0, nu - 1)
                    tap = torch.gather(src, -1, idx)
                    tap = torch.where(ok, tap, torch.zeros_like(tap))
                    val = val + tap * (wx * wy)
                    gu = gu + tap * (sx * wy)
                    gv = gv + tap * (wx * sy)
            val = val.view(B, vc, -1)
            gu = gu.view(B, vc, -1)
            gv = gv.view(B, vc, -1)

            mask = (w > 0) & ((u - u0).abs() <= half_u) & ((v - v_off).abs() <= half_v)
            inv_w2 = 1.0 / (w_safe ** 2)
            contrib = val * inv_w2
            dcontrib = (gu * (udot / du) + gv * (vdot / dv)) * inv_w2 \
                - 2.0 * contrib * (dw / w_safe)
            zero = torch.zeros_like(contrib)
            out[:, p0:p1] += torch.where(mask, wg * contrib, zero).sum(dim=1)
            dout[:, p0:p1] += torch.where(mask, wg * dcontrib + dwg * contrib, zero).sum(dim=1)
    return out, dout


def fdk_conebeam_3d_tangent(
    sino: torch.Tensor,       # (B, V, nv, nu)
    Pmat: torch.Tensor,       # (B, V, 3, 4) the s-dependent geometry P(s)
    Pdot: torch.Tensor,       # (B, V, 3, 4) its derivative dP/ds (rigid_motion.bridge_P_and_dP)
    u_coords: torch.Tensor,
    v_coords: torch.Tensor,
    cfg: ConeBeam3DConfig,
    *,
    D: int,
    H: int,
    W: int,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    window: str = DEFAULT_RAMP_WINDOW,   # filters.py is the ONE source of truth
    cutoff: float = 1.0,
    scale: float | None = None,
    view_weight: torch.Tensor | None = None,       # (B,V) or (V,) [rad], w(s)
    view_weight_dot: torch.Tensor | None = None,   # its s-derivative; required with view_weight
    view_chunk: int = 8,
    vox_chunk: int = 2_000_000,
    eps: float = 1e-8,
    trunc_pad: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FDK(y, P(s)) AND its exact s-derivative, in ONE reconstruction pass.

    The analytic replacement for `train_fm3d.bridge_pair`'s three-FDK central difference,
    and this project's answer to Flowmatching-4DCT's analytic FM tangent (theirs
    differentiates a volume-domain B-spline warp; our bridge's s lives in the projection
    matrices, so the derivative is a JVP through the BACKPROJECTION).

    WHY ONE PASS IS ENOUGH. The sinogram y is s-independent, and every filtering stage is
    per-view and commutes with a positive per-view scalar: cosine and Wang are elementwise
    multiplies, the ramp is linear, and `ohnesorge_pad`'s clamp_min(0) satisfies
    (c*x).clamp_min(0) = c*x.clamp_min(0) for c > 0 -- so the s-dependent Voronoi view weight
    w_v(s) (a positive scalar per view) factors THROUGH the filter instead of being folded
    into the sinogram up front as `fdk_conebeam_3d_batched` does. Filter once, then

        x(s)     = sum_v  w_v(s) * BP_v(P_v(s)) [g_v]
        dx/ds(s) = sum_v  w_v(s) * d/ds BP_v(P_v(s)) [g_v]  +  wdot_v(s) * BP_v(P_v(s)) [g_v]

    both accumulated by the fused kernel from the same detector taps. The value output
    therefore matches `fdk_conebeam_3d_batched(sino, Pmat, view_weight=w)` up to float
    reassociation only -- gated (scripts/gate_fdk_tangent.py) together with the derivative
    (vs float64 central differences).

    `view_weight=None` means the uniform angle_span/V weight (then its derivative is 0 and
    `view_weight_dot` must be None too). `scale` multiplies BOTH outputs. `disp` is not
    supported here -- the bridge's motion is rigid and enters through Pmat.
    Returns (x, dx/ds), each (B, D, H, W), fp32.
    """
    device = sino.device
    dtype = torch.float32
    sino = sino.to(device=device, dtype=dtype)
    Pmat = Pmat.to(device=device, dtype=dtype)
    Pdot = Pdot.to(device=device, dtype=dtype)
    B, V, nv, nu_in = sino.shape

    uni = float(cfg.angle_span) / float(V)
    if view_weight is None:
        if view_weight_dot is not None:
            raise ValueError("view_weight_dot without view_weight")
        wgt = torch.full((B, V), uni, device=device, dtype=dtype)
        dwgt = torch.zeros((B, V), device=device, dtype=dtype)
    else:
        if view_weight_dot is None:
            raise ValueError("view_weight without view_weight_dot (use "
                             "geometry_3d.view_angular_weights_dot)")
        wgt = view_weight.to(device=device, dtype=dtype).expand(B, V).clone()
        dwgt = view_weight_dot.to(device=device, dtype=dtype).expand(B, V).clone()

    # single-source filtering front-end (UNWEIGHTED -- the weights ride on the BP instead)
    g, u0 = fdk_conebeam_3d_batched(
        sino, Pmat, u_coords, v_coords, cfg, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz,
        window=window, cutoff=cutoff, view_chunk=view_chunk, eps=eps,
        trunc_pad=trunc_pad, _return_filtered=True)
    nu = g.shape[-1]                                   # half-fan may have widened the panel
    du = float(cfg.du)
    dv = float(cfg.dv)
    v_off = float(getattr(cfg, "det_offset_v_mm", 0.0))
    half_u = 0.5 * nu * du
    half_v = 0.5 * nv * dv

    # Same grad-safety gate as the static path above: `backproject_tangent` is a raw kernel
    # with no autograd Function, so taking it while a graph is live would return graph-free
    # tensors and d(loss)/d(theta, sino) would come out silently ZERO -- the exact bug class
    # `_use_triton`'s docstring memorializes. The torch fallback IS differentiable, so route
    # there whenever anyone is asking for gradients.
    needs_grad = torch.is_grad_enabled() and (
        g.requires_grad or Pmat.requires_grad or Pdot.requires_grad
        or wgt.requires_grad or dwgt.requires_grad)
    if not needs_grad and triton_backproject.enabled() and g.is_cuda:
        # dx/ds comes from OUR fused tangent kernel (the analytic s-derivative of the
        # bilinear/1/w^2 FDK model -- LEAP has no tangent); the VALUE x(s) comes from the
        # SAME LEAP backprojection as `fdk_conebeam_3d_batched`, so the bridge endpoint
        # x(1) stays bit-consistent with the static anchor. The tangent is therefore the
        # derivative of a 3e-3-close SIBLING model of the value path -- the same
        # cross-model contract the theta gradient lives under, gated by
        # `gate_fdk_tangent.py` (FD of THIS x(s)).
        _, dxds = triton_backproject.backproject_tangent(
            g, Pmat, Pdot, wgt, dwgt, D, H, W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
            u0=u0, v_off=v_off, half_u=half_u, half_v=half_v, eps=eps)
        from .leap_projector import leap_fdk_backproject
        x = leap_fdk_backproject(
            g * wgt.to(g.dtype).view(B, V, 1, 1), Pmat, D=D, H=H, W=W,
            dx=dx, dy=dy, dz=dz, du=du, dv=dv, u0=u0, v_off=v_off)
    else:
        x, dxds = _backproject_tangent_torch(
            g, Pmat, Pdot, wgt, dwgt, D=D, H=H, W=W, dx=dx, dy=dy, dz=dz, du=du, dv=dv,
            u0=u0, v_off=v_off, half_u=half_u, half_v=half_v, eps=eps,
            view_chunk=view_chunk, vox_chunk=vox_chunk)
        x = x.view(B, D, H, W)
        dxds = dxds.view(B, D, H, W)

    s = _fdk_physical_norm(cfg) if scale is None else scale
    x = x * s
    dxds = dxds * s          # the tangent carries the same constant (it is s-independent)
    return x, dxds
