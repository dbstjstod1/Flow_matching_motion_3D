"""Detector-domain ramp filter and the FDK scale constant.

Lifted verbatim from the 2D project's `recon_2d.py` (and identical to the copy the 4DCT
project imported from there) so that this repo carries no 2D fan-beam geometry. Both pieces
are dimension-free: the ramp filters along the DETECTOR u axis, which in 3D means it is
applied row by row -- `fdk_conebeam_3d_batched` reshapes (V,nv,nu) -> (V*nv, nu) and calls
straight into `ramp_filter`. Nothing here knows about the cone.

`calibrate_scale` fits ONE least-squares scalar that folds together du, dbeta, pi and
SDD/SOD. It is an operator constant -- a property of the geometry and the filter, not of the
image -- so it is calibrated once against a reference reconstruction and then reused for every
subsequent recon (including the motion-corrupted ones, which must share the intensity scale
with the clean ones or the flow-matching prior sees a brightness shift that is not physics).
"""

from __future__ import annotations

import torch


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


def ramp_filter(proj: torch.Tensor, du: float, window: str = "hann",
                cutoff: float = 1.0) -> torch.Tensor:
    """Apply a ramp (Ram-Lak) filter along the last axis. proj: (V, nu).

    `window` apodizes the ramp; `cutoff` (in [0,1]) is the fraction of the Nyquist
    frequency above which the filter is zeroed.

      ramlak  : |f|, no apodization                -- sharpest, noisiest
      shepp   : |f| * sinc(f / (2 fc))             -- mild, the usual FBP/FDK default
      cosine  : |f| * cos(pi f / (2 fc))
      hann    : |f| * 0.5 (1 + cos(pi f / fc))     -- strongest smoothing
      rtk     : RTK's ram-lak (see below)          -- what SPARE's FDKRecon was made with
      rtkhann : RTK's ram-lak * RTK's Hann         -- rtkfdk --hann <cutoff>

    NOTE: `hann` (the old hard-coded behaviour) costs real resolution -- it halves the
    ramp at fc/2 and kills everything near Nyquist. Use `shepp` (or `ramlak` on noiseless
    simulated data) when the reconstruction looks blurred.

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
    else:
        raise ValueError(f"unknown ramp window '{window}' (ramlak|shepp|cosine|hann)")
    ramp = torch.where(r > 1.0, torch.zeros_like(ramp), ramp)
    P = P * ramp[None, :].to(P.dtype)
    out = torch.fft.irfft(P, n=N, dim=-1)[:, :nu]
    return out.to(dtype)


def calibrate_scale(recon_raw: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Least-squares scalar so that scale*recon_raw best matches reference."""
    r = recon_raw
    ref = reference.to(r.device, r.dtype)
    if mask is not None:
        r = r[mask]
        ref = ref[mask]
    num = (r * ref).sum()
    den = (r * r).sum().clamp_min(1e-12)
    return float((num / den).item())
