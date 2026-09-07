"""W3DM (JRM-ADM's wavelet-domain x0-DDPM) as a drop-in prior for OUR posterior loop.

THE UNIFIED-LOOP BENCHMARK ARM (user-approved design, 2026-08-18): keep run_posterior3d's
loop -- schedule, motion estimator, CG data step -- byte-identical, and swap ONLY the
clean-direction predictor:

    FM arm  :  x <- x + dt * v_net(x, t)                       (analytic-tangent velocity)
    W3DM arm:  x0_hat = W3DM( sqrt(abar) * x + sqrt(1-abar) * eps )   then
               x <- x + (dt/(1-t)) * (x0_hat - x)              (InDI-style fractional step)

Each net is consumed by the step rule matching its OWN parameterization; step count, dt, and
everything around the prior stay identical. The renoise is an EPHEMERAL input adapter -- their
net only ever saw noised inputs, so the current (noise-free, artifact-bearing) x is pushed to
the matched noise level before the call, and the noise never enters the loop state: the
estimator and the CG step always see the clean x, exactly as in the FM arm.

Schedule map: loop time t in [0,1) -> DDPM timestep t_ddpm = round((1-t) * tmax), clamped to
[1, T-1]. `tmax` is THE knob of this arm (how much of the artifact image the renoise masks at
the cold start; SDEdit's t0). It is swept on val patients and then FROZEN for the cohort --
stated openly in the paper, defensible because the native-solver port exists as the
no-adapter counterpart.

Conversions mirror the A2c port exactly (gated there): mu(ours,0.02) <-> HU <-> [-1,1] over
the [-1000,2000] window; z-flip around the net (the retrained prior saw superior-first heads);
volume 256^3 -> wavelet (8,128,128,128), all multiples of 16 -- no padding needed. Model class
imported straight from the vendored repo (src.models.w3dm.UNetModel -- NOT via creator_utils,
which drags in torch_radon; and never chain .to(), it returns None upstream).
"""
from __future__ import annotations

import os
import sys

import torch

_JRM_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "refs", "jrm-adm")

MU_WATER_OURS = 0.02
HU_LO, HU_HI = -1000.0, 2000.0


class W3DMPrior:
    def __init__(self, weights: str, device: str = "cuda",
                 beta_start: float = 1e-4, beta_end: float = 0.02, T: int = 1000):
        if _JRM_ROOT not in sys.path:
            sys.path.insert(0, _JRM_ROOT)
        from src.models.w3dm import UNetModel
        from src.physics.physics import Wavelet
        # the exact create_model() defaults of the vendored repo (creator_utils.py:14),
        # inlined because that module imports torch_radon at module level
        self.model = UNetModel(
            image_size=192, in_channels=8, model_channels=64, out_channels=8,
            num_res_blocks=2, attention_resolutions=[], dropout=0,
            channel_mult=(1, 2, 2, 4, 4), num_classes=None, use_checkpoint=False,
            use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
            use_scale_shift_norm=False, resblock_updown=True,
            use_new_attention_order=False, dims=3, num_groups=32,
            bottleneck_attention=False, additive_skips=True, resample_2d=False)
        self.model.to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights, map_location=device,
                                              weights_only=True))
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.wavelet = Wavelet()
        self.T = T
        beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.abar = torch.cumprod(1.0 - beta, dim=0)
        self.device = device

    def _mu_to_norm(self, mu: torch.Tensor) -> torch.Tensor:
        hu = (mu / MU_WATER_OURS - 1.0) * 1000.0
        hu = torch.clamp(hu, HU_LO, HU_HI)
        return 2.0 * (hu - HU_LO) / (HU_HI - HU_LO) - 1.0

    def _norm_to_mu(self, xn: torch.Tensor) -> torch.Tensor:
        hu = (xn + 1.0) * (HU_HI - HU_LO) / 2.0 + HU_LO
        return (hu / 1000.0 + 1.0) * MU_WATER_OURS

    @torch.no_grad()
    def predict_clean(self, x_mu: torch.Tensor, t: float, *, tmax: int = 500,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """(D,H,W) mu -> x0_hat (D,H,W) mu. `t` is LOOP time (0 = cold start)."""
        t_ddpm = int(round((1.0 - float(t)) * tmax))
        t_ddpm = max(1, min(t_ddpm, self.T - 1))
        a = self.abar[t_ddpm]

        xn = self._mu_to_norm(x_mu)[None, None]              # (1,1,D,H,W), [-1,1]
        xn = torch.flip(xn, dims=(2,))                       # -> the prior's superior-first z
        eps = (torch.randn(xn.shape, generator=generator, device=xn.device)
               if generator is not None else torch.randn_like(xn))
        x_noisy = torch.sqrt(a) * xn + torch.sqrt(1.0 - a) * eps
        xw = self.wavelet.transform(x_noisy)
        tt = torch.full((1,), t_ddpm, device=self.device, dtype=torch.long)
        x0w = self.model(xw, tt)
        x0 = self.wavelet.transposed_transform(x0w)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = torch.flip(x0, dims=(2,))                       # back to the loop's frame
        return self._norm_to_mu(x0)[0, 0]


@torch.no_grad()
def w3dm_predict(prior: W3DMPrior, x_mu: torch.Tensor, t: float, dt: float, *,
                 tmax: int = 500, generator: torch.Generator | None = None) -> torch.Tensor:
    """One unified-loop prior step with the W3DM net: the InDI-style fractional step toward
    the net's endpoint prediction -- the x0-parameterized twin of fm_predict's Euler step."""
    x0_hat = prior.predict_clean(x_mu, t, tmax=tmax, generator=generator)
    v = (x0_hat - x_mu) / max(1.0 - float(t), 1e-3)
    return x_mu + dt * v
