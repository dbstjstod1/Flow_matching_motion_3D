"""Detector-domain ramp filter and the FDK scale constant.

Lifted verbatim from the 2D project's `recon_2d.py` (and identical to the copy the 4DCT
project imported from there) so that this repo carries no 2D fan-beam geometry. Both pieces
are dimension-free: the ramp filters along the DETECTOR u axis, which in 3D means it is
applied row by row -- `fdk_conebeam_3d_batched` reshapes (V,nv,nu) -> (V*nv, nu) and calls
straight into `ramp_filter`. Nothing here knows about the cone.

The FDK is SELF-NORMALIZED: the deterministic geometry constant SOD*SDD (x1/2 for a full-fan
2pi orbit) is applied inside `fdk_conebeam_3d_batched` (`_fdk_physical_norm`), so it returns
absolute mu [1/mm] like RTK. `calibrate_scale` remains ONLY as a gate diagnostic, where it must
fit ~1.0. NOTHING in the pipeline is normalized by a fitted constant (2026-07-28).
"""

from __future__ import annotations

import torch


# THE ONE PLACE THE PIPELINE'S RAMP WINDOW IS DECIDED. Everything that ramp-filters -- both FDK
# entry points in `projector_3d` and the estimator's `ramp` sinogram loss -- takes its default from
# here, so the choice cannot drift between training and inference. See `ramp_filter`'s docstring
# for what `shepphann` is (LEAP's default ramp + LEAP's minimum low-pass) and
# `projector_3d._RAMP_WINDOW_NOTE` for why an apodized default is required by the SF cube basis.
DEFAULT_RAMP_WINDOW = "shepphann"

_RTK_RAMP_CACHE: dict = {}


def _rtk_ramp_spectrum(N: int, du: float, device, hann_cut: float = 0.0) -> torch.Tensor:
    """RTK's ramp filter response on the rfft bins of an N-point transform. (N//2+1,) float32.

    Transcribed from `rtkFFTRampImageFilter.hxx` (UpdateFFTConvolutionKernel): the analytic
    band-limited ramp is SAMPLED IN THE SPATIAL DOMAIN (Kak & Slaney ch.3 eq.61) over the full
    padded length N, index-wrapped so h[-n] lands at h[N-n], and the filter is that kernel's
    DFT -- which is real (the kernel is even) and strictly positive, in particular at DC.
    Multiplied by `du` so the discrete convolution approximates the analytic integral and the
    result stays on the same intensity scale as the `|f|` branch (`freqs` are cycles/mm there).

    `hann_cut > 0` reproduces `rtkfdk --hann`: a Hann window over the DISCRETE rfft bins,
    zeroed above `ncut = round(hann_cut * n_bins)`. RTK windows the BINS of the padded
    transform, not a physical frequency, so this matches RTK bit-for-bit only at the same N.
    """
    key = (N, round(du, 9), round(hann_cut, 9), str(device))
    hit = _RTK_RAMP_CACHE.get(key)
    if hit is not None:
        return hit
    n = torch.arange(N, dtype=torch.float64)
    dist = torch.minimum(n, N - n)                 # index-wrapped distance from 0
    h = torch.where(dist % 2 == 1, -1.0 / (torch.pi * dist.clamp_min(1.0) * du) ** 2,
                    torch.zeros_like(n))
    h[0] = 1.0 / (4.0 * du * du)
    spec = torch.fft.rfft(h).real * du
    if hann_cut > 0.0:
        nb = spec.shape[0]
        ncut = int(round(nb * min(1.0, float(hann_cut))))
        w = torch.zeros(nb, dtype=torch.float64)
        if ncut > 0:
            k = torch.arange(ncut, dtype=torch.float64)
            w[:ncut] = 0.5 * (1.0 + torch.cos(torch.pi * k / ncut))
        spec = spec * w
    spec = spec.to(device=device, dtype=torch.float32)
    _RTK_RAMP_CACHE[key] = spec
    return spec


def ramp_filter(proj: torch.Tensor, du: float, window: str = DEFAULT_RAMP_WINDOW,
                cutoff: float = 1.0) -> torch.Tensor:
    """Apply a ramp (Ram-Lak) filter along the last axis. proj: (V, nu).

    `window` apodizes the ramp; `cutoff` (in [0,1]) is the fraction of the Nyquist
    frequency above which the filter is zeroed.

      ramlak    : |f|, no apodization                -- sharpest, noisiest
      shepp     : |f| * sinc(f / (2 fc))             -- mild; == LEAP's DEFAULT ramp (order 2)
      cosine    : |f| * cos(pi f / (2 fc))
      hann      : |f| * 0.5 (1 + cos(pi f / fc))     -- == LEAP's set_FBPlowpass(2.0)
      shepphann : |f| * sinc(...) * 0.5(1 + cos ...) -- THE DEFAULT, see below
      rtk       : RTK's ram-lak (see below)          -- what SPARE's FDKRecon was made with
      rtkhann   : RTK's ram-lak * RTK's Hann         -- rtkfdk --hann <cutoff>

    `shepphann` IS THE PIPELINE DEFAULT and is not an ad-hoc product: it is exactly LEAP's
    `set_rampFilter(2)` (their default, = our `shepp`) composed with `set_FBPlowpass(2.0)` (their
    low-pass at the minimum FWHM their docs recommend, = our `hann`). Both equivalences were
    verified against the installed toolkit by measuring its delta response -- `shepp` vs LEAP
    order-2 agrees to max 4e-4 over the band, and `hann` vs LEAP's isolated low-pass to **1.3e-5**,
    i.e. they are the same function (LEAP's FWHM-2-pixel low-pass is the 3-tap binomial
    [1/4, 1/2, 1/4], whose response is 0.5(1 + cos(2 pi f du)) = our Hann at cutoff 1).

    WHY A DEFAULT WITH APODIZATION AT ALL: see `projector_3d._RAMP_WINDOW_NOTE`. Short version --
    the SF projector integrates the CUBE voxel basis, which carries real energy above the voxel
    Nyquist, and our detector RESOLVES it (0.64 mm pitch is 0.4187 mm at isocenter against 1 mm
    voxels, i.e. 2.39x finer), so an unapodized ramp reconstructs the voxel grid as a crosshatch
    texture. `shepphann` removes it (13.2 -> 0.0 HU excess sd in homogeneous brain) at the same
    90% bone-edge sharpness that the retired trilinear-basis operator gave.

    `cutoff` is a HARD zero above `cutoff * f_Nyquist`. Do NOT reach for it to fight the cube-basis
    texture: cutting at the voxel Nyquist (cutoff 0.419 here) was measured and costs 15 more points
    of bone-edge sharpness than `shepphann` while removing no more texture -- the texture lives in a
    narrow near-Nyquist band, not uniformly above it.

    `rtk`/`rtkhann` REPRODUCE RTK's FILTER DISCRETIZATION, and the difference from `ramlak`
    is NOT cosmetic. RTK (`rtkFFTRampImageFilter::UpdateFFTConvolutionKernel`) does not
    sample |f| in the frequency domain: it samples the ANALYTIC band-limited ramp kernel in
    the SPATIAL domain (Kak & Slaney ch.3 eq.61: h[0] = 1/(4 du^2), h[n odd] = -1/(pi n du)^2,
    h[n even] = 0, wrapped to length N) and takes ITS DFT as the filter. The two agree at high
    frequency but differ near DC: the sampled kernel's spectrum stays STRICTLY POSITIVE
    (DC ~ 1/(N^2 du^2) instead of |0| = 0), i.e. sampling |f| discards the mean of every
    filtered row. A projection row's mean is huge (a body-sized line integral), so the |f|
    version depresses the whole reconstruction interior and rings a halo into the air around
    it -- a low-frequency radial bowl worth ~10% rel-rmse against RTK's own reconstruction of
    the same projections (MEASURED on SPARE-MC, see docs/SPARE_SIMULATION.md).
    `rtkhann` applies RTK's window semantics: Hann over the DISCRETE rfft bins, cut at
    `round(cutoff * n_bins)` -- on the padded grid, so it is a different curve from `hann`.
    """
    V, nu = proj.shape
    device, dtype = proj.device, proj.dtype
    N = 1
    while N < 2 * nu:
        N *= 2
    P = torch.fft.rfft(proj, n=N, dim=-1)
    if window in ("rtk", "rtkhann"):
        ramp = _rtk_ramp_spectrum(N, du, device=device,
                                  hann_cut=(float(cutoff) if window == "rtkhann" else 0.0))
        P = P * ramp[None, :].to(P.dtype)
        out = torch.fft.irfft(P, n=N, dim=-1)[:, :nu]
        return out.to(dtype)
    freqs = torch.fft.rfftfreq(N, d=du).to(device=device, dtype=dtype)  # cycles/mm
    fnyq = freqs.abs().max().clamp_min(1e-12)
    fc = (float(cutoff) * fnyq).clamp_min(1e-12) if torch.is_tensor(fnyq) else cutoff * fnyq
    ramp = freqs.abs().clone()
    r = freqs.abs() / fc
    if window == "ramlak":
        pass
    elif window == "shepp":
        ramp = ramp * torch.sinc(0.5 * r)                 # sinc(x)=sin(pi x)/(pi x)
    elif window == "cosine":
        ramp = ramp * torch.cos(0.5 * torch.pi * r.clamp(max=1.0))
    elif window == "hann":
        ramp = ramp * 0.5 * (1.0 + torch.cos(torch.pi * r.clamp(max=1.0)))
    elif window == "shepphann":                       # = LEAP ord2 + set_FBPlowpass(2.0)
        ramp = (ramp * torch.sinc(0.5 * r)
                * 0.5 * (1.0 + torch.cos(torch.pi * r.clamp(max=1.0))))
    else:
        raise ValueError(f"unknown ramp window '{window}' "
                         f"(ramlak|shepp|cosine|hann|shepphann|rtk|rtkhann)")
    ramp = torch.where(r > 1.0, torch.zeros_like(ramp), ramp)
    P = P * ramp[None, :].to(P.dtype)
    out = torch.fft.irfft(P, n=N, dim=-1)[:, :nu]
    return out.to(dtype)


def calibrate_scale(recon_raw: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Least-squares scalar so that scale*recon_raw best matches reference.

    DIAGNOSTIC ONLY (2026-07-28). This used to produce the pipeline's normalization constant.
    That is gone: the FDK self-normalizes from the GIVEN geometry
    (`projector_3d._fdk_physical_norm` = SOD*SDD/2), so there is nothing to fit -- and fitting
    was harmful, because a scalar regressed on FDK(A(v)) vs v absorbs any operator or level
    error (units, voxel size, detector pitch, and in the 4DCT sibling's case scatter/bowtie/air
    constants) and leaves the pipeline self-consistent and wrong. The 4DCT sibling deleted its
    own copy on 2026-07-21 for exactly this reason. Keep using this in GATES, where the claim is
    that it comes out 1.0 (gate_cq500 check 6)."""
    r = recon_raw
    ref = reference.to(r.device, r.dtype)
    if mask is not None:
        r = r[mask]
        ref = ref[mask]
    num = (r * ref).sum()
    den = (r * r).sum().clamp_min(1e-12)
    return float((num / den).item())
