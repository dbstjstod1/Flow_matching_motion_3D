"""Thies' reconstruction operator: cosine pre-weight + ramp, then their UNWEIGHTED backprojection.

This is the `r : R^M -> R^K` of TMI section II-B.2, wired to OUR geometry so the benchmark and
`scripts/run_posterior3d.py` consume the identical sinogram.

WHAT THEIR RECONSTRUCTION IS, EXACTLY
-------------------------------------
TMI L303-312 and Eq. 3:

    "the backprojection depends on an appropriately filtered version of the projection images.
     Since we consider projection data acquired on a full circular trajectory, we apply a
     classical shift-invariant ramp filter. During backprojection, the reconstructed value I at
     position p_world is computed as the SUM over the values of the filtered projection data at
     the forward projected positions in each projection image"

        I(p) = sum_j  d_j( g( s_j(p) ) )                                              (Eq. 3)

and TMI L476 for the pre-weight: "A ramp filter and a cosine filter are applied to the computed
projection images."

So three steps, and only three:

  1. cosine pre-weight  g~ = g * SDD / sqrt(SDD^2 + u^2 + v^2)   [flat panel]
  2. ramp along the detector u-axis, plain |f| -- "classical shift-invariant", no apodization
  3. BACKPROJECT AS AN UNWEIGHTED SUM

Step 3 is the load-bearing difference from our own FDK, and it is not our reading of an ambiguous
sentence: their released CUDA kernel does literally

    cuda.atomic.add(reco, (x, y, z), interpolate2d_cuda(sino, v / w, u / w))
        -- vendor/geometry_gradients_CT/backprojector_cone.py:71

with NO `1/w^2` distance weight and NO angular weight. Ours
(`fm3d.projector_3d.fdk_conebeam_3d_batched`) has both, and the angular one -- a Voronoi
partition read out of the moved Pmat -- is worth +1.14 dB under motion
(`fm3d.geometry_3d.view_angular_weights`). `distance_weight=True` restores the `1/w^2` term for
an A/B, but the DEFAULT IS FALSE = theirs. Do not report a `distance_weight=True` number as
"Thies".

FILTER ONCE, BACKPROJECT MANY
-----------------------------
The paper is explicit that only the backprojection sees the updated matrices ("the backprojection
r uses the updated projection matrices to analytically backproject the filtered projection data.
Note that, for simplicity, we omit the filtered projection data as an input to r", L207-211). The
cosine weight and the ramp are functions of the DETECTOR alone, so the filtered sinogram is
computed once per scan and reused across all 100 gradient-descent iterations. Same structure as
our `fm3d.projector_3d.fdk_backproject_filtered`, and it is what makes their loop cheap.

COORDINATE CONVENTIONS (the part that silently corrupts a benchmark if wrong)
----------------------------------------------------------------------------
Their kernel, read off the source:

  * volume array axis 0 <-> world z, axis 1 <-> world y, axis 2 <-> world x
    (`point2` comes from loop index `x` = axis 0 and multiplies P's column 2 = our z; and so on).
    That is exactly our (D, H, W) layout, so `volume_shape=(D,H,W)`,
    `volume_spacing=(dz,dy,dx)`, `volume_origin=(z0,y0,x0)`, world = index*spacing + origin.
  * the sinogram is indexed `sino[v_index, u_index]` -- our (nv, nu). Good.
  * **the projection matrices must map world mm -> DETECTOR PIXEL INDEX**, not to mm. Two
    independent confirmations: `interpolate2d_cuda` indexes the sinogram array directly with
    `v/w, u/w`, and the backward pass differentiates the sinogram with `torch.gradient(...)` at
    unit spacing (`backprojector_cone.py:83`), which is a per-index derivative. Our
    `geometry_3d.build_conebeam_orbit` emits mm (K = [[SDD,0,u0],[0,SDD,v0],[0,0,1]]), so
    `to_pixel_matrices` applies the detector index map. Getting this wrong does not raise -- it
    silently reconstructs a differently-scaled object and the whole benchmark is void.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fm3d.filters import ramp_filter
from fm3d.geometry_3d import ConeBeam3DConfig, detector_coords_3d

from .fast_backprojector import backprojector
from .vendor_import import vendored_geometry

__all__ = ["to_pixel_matrices", "ThiesConeRecon", "VolumeGrid",
           "MU_LO", "MU_HI", "HU_WINDOW", "to_unit", "from_unit"]


# ---------------------------------------------------------------------------------------------
# Fixed, SAMPLE-INDEPENDENT intensity mapping.
#
# TMI L493-496: "The intensities of the reconstructed volumes are normalized with fixed,
# sample-independent offset and slope to lie approximately in the interval [0, 1] before feeding
# them to the network." The paper does not print the constants. We use the window its own Fig. 6
# is displayed in (-1200 .. +1500 HU, caption at L606), converted to mu at our mu_water -- so the
# net's input range and the figures' display range are the same interval, and both are constants
# of the protocol rather than of the sample.
# ---------------------------------------------------------------------------------------------
HU_WINDOW = (-1200.0, 1500.0)
MU_WATER = 0.02                      # 1/mm, the value fm3d.dataset_cq500 uses for HU <-> mu
MU_LO = (HU_WINDOW[0] / 1000.0 + 1.0) * MU_WATER     # 0.0 (air-ish)
MU_HI = (HU_WINDOW[1] / 1000.0 + 1.0) * MU_WATER     # 0.05


def to_unit(mu: torch.Tensor) -> torch.Tensor:
    """mu [1/mm] -> the net's ~[0,1] input. Affine with FIXED constants (see above)."""
    return (mu - MU_LO) / (MU_HI - MU_LO)


def from_unit(x: torch.Tensor) -> torch.Tensor:
    return x * (MU_HI - MU_LO) + MU_LO


_GEOM_CACHE: dict = {}


@dataclass(frozen=True)
class VolumeGrid:
    """The reconstruction grid, in the vendored kernel's own terms.

    Thies use TWO of these and it matters (L505-508): motion is estimated on
    `VolumeGrid.centred(128, 2.0)` and the reported images are reconstructed on
    `VolumeGrid.centred(256, 1.0)` with the estimate held fixed.
    """
    shape: tuple[int, int, int]          # (D, H, W) = (z, y, x)
    spacing: tuple[float, float, float]  # (dz, dy, dx) [mm]
    origin: tuple[float, float, float]   # world coord of index 0, per axis [mm]

    @classmethod
    def centred(cls, n: int | tuple[int, int, int], voxel_mm: float) -> "VolumeGrid":
        """Voxel CENTRES at (i - (N-1)/2) * d -- the convention of `fm3d.geometry_3d`'s header,
        used by every projector, FDK and warp in this repo. The isocentre therefore sits at the
        grid centre, matching `fm3d.dataset_cq500`'s volumes voxel-for-voxel."""
        shp = (n, n, n) if isinstance(n, int) else tuple(int(v) for v in n)
        org = tuple(-(s - 1) / 2.0 * voxel_mm for s in shp)
        return cls(shp, (voxel_mm,) * 3, org)

    def vendored(self):
        """Their `Geometry` collector, memoized.

        `volume_spacing`/`volume_origin` are pushed to the DEVICE once. Handing numpy arrays to
        their `@cuda.jit` kernels works, but numba then copies them host->device on EVERY launch
        -- and their backward launches one kernel PER VIEW (`call_backward_kernel` loops over
        angles in Python), so a 360-view scan would pay 360 tiny synchronous copies per gradient
        step, plus a NumbaPerformanceWarning each time. Their code is untouched; only what we
        hand it changes.

        detector_origin / detector_spacing are unused by the cone kernel -- the whole detector
        mapping lives in the pixel-domain projection matrices. Passed as None so any future use
        raises instead of silently reading a wrong number.
        """
        key = (self.shape, self.spacing, self.origin)
        if key not in _GEOM_CACHE:
            from numba import cuda as _nbcuda
            Geometry = vendored_geometry()
            _GEOM_CACHE[key] = Geometry(
                volume_shape=self.shape,
                volume_origin=_nbcuda.to_device(np.asarray(self.origin, dtype=np.float32)),
                volume_spacing=_nbcuda.to_device(np.asarray(self.spacing, dtype=np.float32)),
                detector_origin=None, detector_spacing=None)
        return _GEOM_CACHE[key]


def to_pixel_matrices(P_mm: torch.Tensor, cfg: ConeBeam3DConfig) -> torch.Tensor:
    """(V,3,4) our mm-domain P  ->  (V,3,4) P in DETECTOR PIXEL INDEX units. Differentiable.

    `geometry_3d.detector_coords_3d` places element `ju` at
    `u_mm = (ju - (nu-1)/2)*du + det_offset_u_mm`, so the inverse index map is

        ju = (u_mm - off_u)/du + (nu-1)/2

    which is the left-multiplication below. Because it is a CONSTANT matrix, autograd carries the
    gradient the vendored kernel returns for `P_pix` straight back to `P_mm` (and hence to the
    spline parameters) with no extra code.
    """
    S = torch.zeros(3, 3, device=P_mm.device, dtype=P_mm.dtype)
    S[0, 0] = 1.0 / cfg.du
    S[1, 1] = 1.0 / cfg.dv
    S[0, 2] = (cfg.nu - 1) / 2.0 - cfg.det_offset_u_mm / cfg.du
    S[1, 2] = (cfg.nv - 1) / 2.0 - cfg.det_offset_v_mm / cfg.dv
    S[2, 2] = 1.0
    return torch.matmul(S, P_mm)


class ThiesConeRecon:
    """`r(P)` of TMI II-B.2. Call `.filter(y)` once per scan, then `.backproject(g, P, grid)`.

    `__call__(y, P_mm, grid)` does both, for convenience in the one-shot paths (gates, final
    256^3 reconstruction). Inside the 100-iteration loop use the two-step form.
    """

    def __init__(self, cfg: ConeBeam3DConfig, *, distance_weight: bool = False,
                 mu_scale: bool = True, ramp_window: str = "ramlak", ramp_cutoff: float = 1.0,
                 fast: bool | None = None):
        if abs(cfg.angular_range_deg - 360.0) > 1e-6:
            raise ValueError(
                f"angular_range_deg={cfg.angular_range_deg}: the plain ramp is only justified on "
                "the FULL circular trajectory the paper simulates (L303-307). Their SHORT-scan "
                "path (the 200 deg clinical scans, L727-729) additionally needs Parker weights "
                "and truncation extrapolation, which are deliberately not implemented here.")
        if cfg.is_half_fan:
            raise ValueError("half-fan is not part of Thies' protocol; the Wang weight it needs "
                             "lives in our FDK, not theirs.")
        self.cfg = cfg
        self.distance_weight = bool(distance_weight)
        self.mu_scale = bool(mu_scale)
        self.ramp_window = ramp_window
        self.ramp_cutoff = float(ramp_cutoff)
        # THE KERNELS, not the maths. `fast=None` (the default) takes `fast_backprojector`'s
        # numerically-equivalent rewrite of the vendored kernels -- same expressions, same
        # `int()`-truncating interpolation, only the float REDUCTION ORDER differs (register
        # accumulation in the forward, a shared-memory block reduction in the backward). It
        # exists because the vendored backward spends 21.1 of its 21.5 s in serialized atomics;
        # see that module's docstring for the measurement. `fast=False` (or FM3D_THIES_VENDOR_BP=1)
        # restores the vendored kernels, which is what gate check G9 compares against.
        self.fast = fast
        self._bp = backprojector(fast).apply

    # -- constants -----------------------------------------------------------------------
    @property
    def scale(self) -> float:
        """OUR addition (PROVENANCE.md section 4.2), not theirs: put the unweighted sum into
        mu [1/mm] so it is directly comparable to the GT volume and to our FDK.

            ours   = sum_v  dbeta_v * g_filt / w^2 * (SOD*SDD/2)
            theirs = sum_v            g_filt

        so with a uniform dbeta = span/V and the on-axis value w = SOD,

            scale = dbeta * (SOD*SDD/2) / SOD^2 = dbeta * SDD / (2*SOD).

        The residual `1/w^2` shape they omit is a smooth radial cupping, NOT a streak -- it is
        why this is a scalar and not a calibration file. Gate G3 pins the central-region
        agreement against our static FDK. `mu_scale=False` returns their raw sum.
        """
        if not self.mu_scale:
            return 1.0
        cfg = self.cfg
        d_beta = cfg.angle_span / cfg.n_views
        return d_beta * cfg.SDD / (2.0 * cfg.SOD)

    # -- step 1+2: the filter, once per scan --------------------------------------------
    @torch.no_grad()
    def filter(self, y: torch.Tensor) -> torch.Tensor:
        """(V,nv,nu) raw line integrals -> (V,nv,nu) cosine-weighted, ramp-filtered.

        No autograd: `y` is data and the filter never sees the motion parameters, which is the
        entire reason it can be hoisted out of the optimizer loop.
        """
        cfg = self.cfg
        if y.dim() == 4:
            if y.shape[0] != 1:
                raise ValueError(f"the vendored kernel is single-scan; got batch {y.shape[0]}")
            y = y[0]
        if y.dim() != 3:
            raise ValueError(f"y must be (V,nv,nu) or (1,V,nv,nu); got {tuple(y.shape)}")
        V, nv, nu = y.shape
        if (nv, nu) != (cfg.nv, cfg.nu):
            raise ValueError(f"detector {(nv, nu)} != cfg {(cfg.nv, cfg.nu)}")

        u, v = detector_coords_3d(cfg, device=y.device, dtype=y.dtype)
        # 1) cosine pre-weight for a FLAT panel.
        w_cos = cfg.SDD / torch.sqrt(cfg.SDD ** 2 + u[None, :] ** 2 + v[:, None] ** 2)
        g = y * w_cos[None]
        # 2) plain ramp along u. "classical shift-invariant ramp filter" (L307) = ramlak; we do
        #    NOT inherit our own `shepphann` default, which exists to tame the cube-voxel basis of
        #    our forward operator and is not part of their method.
        g = ramp_filter(g.reshape(-1, nu), du=cfg.du,
                        window=self.ramp_window, cutoff=self.ramp_cutoff).reshape(V, nv, nu)
        return g.contiguous().float()

    # -- step 3: the vendored backprojection --------------------------------------------
    def backproject(self, g_filt: torch.Tensor, P_mm: torch.Tensor,
                    grid: VolumeGrid) -> torch.Tensor:
        """(V,nv,nu) filtered sinogram + (V,3,4) mm-domain P -> (D,H,W) volume [mu].

        Differentiable in `P_mm` through the vendored analytic Jacobian; `g_filt` gets no
        gradient (their `backward` returns None for it, which is correct here -- the sinogram is
        a measurement, not a variable).
        """
        if P_mm.shape[-2:] != (3, 4) or P_mm.dim() != 3:
            raise ValueError(f"P_mm must be (V,3,4); got {tuple(P_mm.shape)}")
        if P_mm.shape[0] != g_filt.shape[0]:
            raise ValueError(f"{P_mm.shape[0]} matrices vs {g_filt.shape[0]} views")
        P_pix = to_pixel_matrices(P_mm, self.cfg).contiguous().float()
        # `.clone()`: the vendored forward hands back `torch.as_tensor(<numba DeviceNDArray>)`
        # over memory it allocated itself. Cloning immediately gives us a plainly-owned torch
        # tensor and costs 67 MB at 256^3 -- cheap insurance against a dangling device pointer,
        # and it is differentiable so the analytic Jacobian is untouched.
        vol = self._bp(g_filt, P_pix, grid.vendored()).clone()
        if self.distance_weight:
            vol = vol * self._inv_w2(P_mm, grid)
        return vol * self.scale

    def __call__(self, y: torch.Tensor, P_mm: torch.Tensor, grid: VolumeGrid) -> torch.Tensor:
        return self.backproject(self.filter(y), P_mm, grid)

    # -- the A/B knob --------------------------------------------------------------------
    def _inv_w2(self, P_mm: torch.Tensor, grid: VolumeGrid) -> torch.Tensor:
        """sum_v 1/w_v(p)^2 is NOT separable from the backprojection, so `distance_weight=True`
        cannot be exact without editing their kernel -- which we will not do. This applies the
        VIEW-AVERAGED weight instead, which is the honest cheap approximation and is enough to
        answer "does the missing distance weight matter for the artifact pattern?".

        Deliberately not the default, and any number produced with it must be labelled as a
        MODIFIED Thies, per PROVENANCE.md section 4.1.
        """
        D, H, W = grid.shape
        dz, dy, dx = grid.spacing
        z0, y0, x0 = grid.origin
        dev, dt = P_mm.device, P_mm.dtype
        z = torch.arange(D, device=dev, dtype=dt) * dz + z0
        yy = torch.arange(H, device=dev, dtype=dt) * dy + y0
        x = torch.arange(W, device=dev, dtype=dt) * dx + x0
        pts = torch.stack(torch.meshgrid(z, yy, x, indexing="ij"), -1)          # (D,H,W,3) z,y,x
        pts = pts.flip(-1)                                                      # -> x,y,z
        w = torch.einsum("vc,dhwc->vdhw", P_mm[:, 2, :3], pts) + P_mm[:, 2, 3][:, None, None, None]
        return (1.0 / w.clamp_min(1e-6) ** 2).mean(0) * self.cfg.SOD ** 2
