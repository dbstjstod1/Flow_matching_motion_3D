"""Spatially resolved 3D VIF, and the regression target `VIF* = 1 - K * VIF`.

TMI II-B.3, L370-334:

    "As the regression target, we choose the visual information fidelity (VIF) [34]. ... To
     obtain a spatially resolved volumetric map of the VIF, we follow the approach by Shao et
     al. [35]. Given a paired data set of motion-free and motion-affected head CT volumes, we
     compute their corresponding VIF map by setting the motion-free volume as reference image
     and the motion-affected volume as distorted image. The VIF map is further scaled by the
     number of voxels K such that an average operation yields values in the range [0, 1]. As the
     final regression target for the network we use VIF*(I_dist, I_ref) = 1 - K * VIF(...)"

  [34] Sheikh & Bovik, "Image information and visual quality", IEEE TIP 15(2):430-444, 2006.
  [35] Y. Shao, F. Sun, H. Li, Y. Liu, "A novel approach for computing quality map of visual
       information fidelity index", Proc. ISKE vol. 213, Springer 2014, pp. 163-173.

WHAT WE IMPLEMENT, AND WHY IT SATISFIES THE PAPER'S OWN CONSTRAINT
------------------------------------------------------------------
We do not have Shao et al. What the paper actually *requires* of the map is stated in its own
sentence: it must be "scaled by the number of voxels K such that an average operation yields
values in the range [0,1]" -- i.e.

        mean_i (K * VIFmap_i) = sum_i VIFmap_i  ==  the scalar VIF.

So the map must be a DECOMPOSITION OF THE SCALAR, and the scalar is Sheikh & Bovik's. That fixes
the construction up to how the per-scale numerator is localized:

    VIF = sum_scales sum_i num_s(i)  /  sum_scales sum_i den_s(i)          (the scalar, verbatim)
    VIFmap(i) = sum_scales up_s(num_s)(i)  /  sum_scales sum_i den_s(i)    (ours)

with `up_s` the SUM-PRESERVING upsample from scale s back to full resolution (nearest-neighbour
replication divided by the replication factor). Then `sum_i VIFmap(i) == VIF` exactly, which gate
check G4 asserts to 1e-5. The denominator stays global because it is the reference image's total
information content -- a normalizer, not a local quantity.

We use the PIXEL-DOMAIN VIF (VIF-P, Sheikh's own simplified variant: local Gaussian statistics
at 4 scales instead of a steerable pyramid), extended to 3D by making every filter and every
decimation separable in 3 dimensions. `sigma_nsq = 2` is Sheikh's calibrated HVS noise floor for
0-255 imagery, so inputs are internally rescaled to 0-255 before the statistics are formed --
otherwise that constant would silently dominate and every VIF would collapse to ~0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["vif_map_3d", "vif_scalar_3d", "vif_star_map_3d", "SIGMA_NSQ"]

SIGMA_NSQ = 2.0          # Sheikh & Bovik's HVS noise floor, calibrated on 0-255 images
_EPS = 1e-10


def _gauss1d(n: int, sd: float, device, dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0
    g = torch.exp(-(k ** 2) / (2.0 * sd ** 2))
    return g / g.sum()


def _sepconv3d(x: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """Separable 3D convolution, VALID padding (as in the reference VIF-P implementation: the
    borders are dropped rather than reflected, so no synthetic edge information is created)."""
    n = k1.numel()
    for dim in range(2, 5):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = n
        x = F.conv3d(x, k1.view(shape))
    return x


def _sum_preserving_up(x: torch.Tensor, factor: int, out_shape: tuple[int, int, int],
                       ) -> torch.Tensor:
    """Replicate each voxel `factor^3` times, divided by `factor^3`, then pad/crop to
    `out_shape`. Total mass is preserved exactly, which is the whole point (see the module
    docstring): the map must still sum to the scalar VIF."""
    if factor > 1:
        x = x.repeat_interleave(factor, -3).repeat_interleave(factor, -2) \
             .repeat_interleave(factor, -1) / float(factor ** 3)
    out = x.new_zeros(x.shape[:2] + out_shape)
    d, h, w = (min(a, b) for a, b in zip(x.shape[2:], out_shape))
    # centre the valid block: `_sepconv3d` trims symmetrically, and the decimation below keeps
    # the same centre, so a centred paste puts each contribution back where it came from.
    o = [(out_shape[i] - [d, h, w][i]) // 2 for i in range(3)]
    out[:, :, o[0]:o[0] + d, o[1]:o[1] + h, o[2]:o[2] + w] = x[:, :, :d, :h, :w]
    return out


def _vif_terms(ref: torch.Tensor, dist: torch.Tensor, n_scales: int = 4):
    """Yield `(num_map_s, den_total_s, decimation_factor_s)` for each of Sheikh's 4 scales.

    THE SECOND MOMENTS ARE MEAN-CENTRED FIRST, and that is a precision fix, not a change of
    formula. Written the textbook way, `s1sq = conv(x*x) - mu*mu` subtracts two numbers of order
    `mu^2`; inputs are rescaled to 0-255 here (Sheikh's `sigma_nsq = 2` is calibrated there), so
    `mu^2 ~ 6.5e4` while the local variance in a smooth head region is O(1) -- a catastrophic
    cancellation that costs ~5 significant digits and is the reason this module used to run
    entirely in float64. Centring on a SHIFT that is already close to the local mean,

        s1sq = conv((x - c)^2) - (mu - c)^2,        c = the volume mean,

    leaves both terms O(variance) and the subtraction well conditioned, which lets the whole
    computation run in float32. MEASURED (128^3, batch 1, A6000): 412.8 ms -> ~18 ms, i.e. the
    VIF* target was 33% of a `bench_thies_train_qm.py` step and is now ~1.5%. The identity is
    exact in real arithmetic for ANY constant `c` (conv is normalized to sum 1), so this is not
    an approximation -- gate check G4 still pins `sum(map) == scalar` to 1e-5, and G10 pins the
    map itself against the float64 result.
    """
    c1 = ref.mean()
    c2 = dist.mean()
    ref = ref - c1
    dist = dist - c2
    for scale in range(1, n_scales + 1):
        n = 2 ** (n_scales - scale + 1) + 1           # 17, 9, 5, 3
        sd = n / 5.0
        k = _gauss1d(n, sd, ref.device, ref.dtype)
        if scale > 1:                                  # lowpass then decimate by 2
            ref = _sepconv3d(ref, k)[:, :, ::2, ::2, ::2]
            dist = _sepconv3d(dist, k)[:, :, ::2, ::2, ::2]

        # mu1/mu2 are the means of the CENTRED fields; every use below is either a difference of
        # second moments (where the shift cancels) or `s12`, which is a covariance and is
        # shift-invariant too. The un-centred means are never needed.
        mu1 = _sepconv3d(ref, k)
        mu2 = _sepconv3d(dist, k)
        mu1sq, mu2sq, mu1mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1sq = _sepconv3d(ref * ref, k) - mu1sq
        s2sq = _sepconv3d(dist * dist, k) - mu2sq
        s12 = _sepconv3d(ref * dist, k) - mu1mu2

        s1sq = s1sq.clamp_min(0.0)
        s2sq = s2sq.clamp_min(0.0)

        g = s12 / (s1sq + _EPS)
        sv_sq = s2sq - g * s12

        # Sheikh's degenerate-case handling, verbatim in effect:
        deg = s1sq < _EPS
        g = torch.where(deg, torch.zeros_like(g), g)
        sv_sq = torch.where(deg, s2sq, sv_sq)
        s1sq = torch.where(deg, torch.zeros_like(s1sq), s1sq)

        deg2 = s2sq < _EPS
        g = torch.where(deg2, torch.zeros_like(g), g)
        sv_sq = torch.where(deg2, torch.zeros_like(sv_sq), sv_sq)

        g = g.clamp_min(0.0)
        sv_sq = sv_sq.clamp_min(_EPS)

        num = torch.log2(1.0 + (g ** 2) * s1sq / (sv_sq + SIGMA_NSQ))
        den = torch.log2(1.0 + s1sq / SIGMA_NSQ)
        yield num, den.sum(dim=(2, 3, 4), keepdim=True), 2 ** (scale - 1)


@torch.no_grad()
def vif_map_3d(dist: torch.Tensor, ref: torch.Tensor, *, input_range: float = 1.0,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(B,1,D,H,W) -> (B,1,D,H,W) VIF map that SUMS to the scalar VIF, per volume.

    `dist` is the motion-affected volume, `ref` the motion-free one -- that is the paper's
    assignment (L322-325) and swapping them is not symmetric.

    `input_range` is the value that should map to 255. Volumes are expected in the fixed
    ~[0,1] scaling of `recon.to_unit`, so the default is right; pass 255.0 if you already did it.

    `dtype` is the WORKING precision. float32 is the default and is what the training loop pays;
    it is only safe because `_vif_terms` mean-centres before forming the second moments (see
    there -- the un-centred form loses ~5 digits at the 0-255 scale and forced float64). Pass
    `torch.float64` for the reference value; gate check G10 holds the two together.
    """
    if ref.shape != dist.shape or ref.dim() != 5:
        raise ValueError(f"need matching (B,1,D,H,W); got {tuple(dist.shape)} vs {tuple(ref.shape)}")
    s = 255.0 / float(input_range)
    ref = ref.to(dtype) * s
    dist = dist.to(dtype) * s
    out_shape = tuple(ref.shape[2:])

    num_total = torch.zeros_like(ref)
    den_total = ref.new_zeros(ref.shape[0], 1, 1, 1, 1)
    for num, den, fac in _vif_terms(ref, dist):
        num_total = num_total + _sum_preserving_up(num, fac, out_shape)
        den_total = den_total + den
    return (num_total / (den_total + _EPS)).to(dist.dtype)


@torch.no_grad()
def vif_scalar_3d(dist: torch.Tensor, ref: torch.Tensor, *, input_range: float = 1.0,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(B,) the plain Sheikh & Bovik scalar. Computed independently of `vif_map_3d` so gate G4
    can check that the map really does sum to it."""
    s = 255.0 / float(input_range)
    ref = ref.to(dtype) * s
    dist = dist.to(dtype) * s
    num = ref.new_zeros(ref.shape[0])
    den = ref.new_zeros(ref.shape[0])
    for n_, d_, _ in _vif_terms(ref, dist):
        num = num + n_.sum(dim=(1, 2, 3, 4))
        den = den + d_.reshape(d_.shape[0], -1).sum(-1)
    return (num / (den + _EPS)).to(dist.dtype)


@torch.no_grad()
def vif_star_map_3d(dist: torch.Tensor, ref: torch.Tensor, *, input_range: float = 1.0,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """THE REGRESSION TARGET: `VIF* = 1 - K * VIF_map`, K = number of voxels (L330-332).

    Its spatial MEAN is `1 - VIF_scalar`, i.e. 0 for a perfect reconstruction and ->1 as the
    motion destroys the information -- which is why the optimizer MINIMIZES the network's mean
    output (L347-349).
    """
    K = float(dist.shape[2] * dist.shape[3] * dist.shape[4])
    return 1.0 - K * vif_map_3d(dist, ref, input_range=input_range, dtype=dtype)
