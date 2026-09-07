"""Coordinate networks over the view index: v -> theta(v) in R^6.

Ported from the 2D project's `MotionNetHash3DoF`, which was itself a port of AI_Geocal's
`MotionNetHash_9DoF`. The only structural change is the head width (3 -> 6) and the bounds.

WHY A NETWORK AT ALL, AND WHY A BAND-LIMITED ONE. Free per-view parameters have V*6 degrees of
freedom against V projections, so nothing but a prior stops the fit from putting a different
rigid pose on every view and explaining the data with jitter. AI_Geocal's answer is exactly this
net: the hash grid + MLP is a low-capacity continuous function of the normalized view coordinate,
so smoothness is structural rather than a penalty term (it carries no explicit smoothness loss).

DEFAULTS ARE THE DEPLOYED FULLBAND ONES (stock Instant-NGP 16/16/1.5, as in AI_Geocal), paired
with lr 3e-3 downstream. The 2D project's band-limited preset ("hashbl", 4/2/2.0 -- a 16-cell
finest grid) was this file's default until 2026-08-07, when the preset and its `--est_band`
switch were REMOVED (user's call): the 3D oracle sweep showed bandwidth x lr is ONE knob and
fullband @ small lr beats hashbl @ its own lr 5x (rot 0.035 vs 0.173 deg at equal cost), and it
transferred in-loop. The knobs stay exposed so any band limit can be re-tuned when the real
scan's view count and motion bandwidth are known; git history has the preset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    HAVE_TCNN = True
except Exception:
    HAVE_TCNN = False


class FourierFeatureEncoder(nn.Module):
    """[sin(pi j s), cos(pi j s)]_{j=1..M} on s in [0,1]. Band-limited BY CONSTRUCTION.

    A deliberate contrast with the hash grid: its bandwidth is exactly M, visible in the
    signature rather than emergent from three interacting hyper-parameters.
    """

    def __init__(self, m: int = 8):
        super().__init__()
        self.m = m
        self.n_output_dims = 2 * m

    def forward(self, x):                       # x: (B, >=1) with x[:, 0] = s
        s = x[..., :1]
        j = torch.arange(1, self.m + 1, device=x.device, dtype=x.dtype)[None, :]
        a = math.pi * j * s
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class _HashGrid2D(nn.Module):
    """Pure-PyTorch multiresolution hash grid, for when tinycudann will not build.

    2D because that is the smallest input tcnn's HashGrid accepts, and we feed it (s, 0) -- the
    second coordinate is inert. Kept bit-compatible with the 2D project's vendored version,
    including the prime-XOR hash.
    """

    _PRIMES = (1, 2654435761)

    def __init__(self, n_levels=4, n_features=2, log2_hashmap_size=15,
                 base_resolution=2, per_level_scale=2.0):
        super().__init__()
        self.n_levels, self.n_features = n_levels, n_features
        self.base_res, self.scale = base_resolution, per_level_scale
        self.hashmap = 2 ** log2_hashmap_size
        self.emb = nn.ModuleList([
            nn.Embedding(self.hashmap, n_features) for _ in range(n_levels)])
        for e in self.emb:
            nn.init.uniform_(e.weight, -1e-4, 1e-4)
        self.n_output_dims = n_levels * n_features

    def _hash(self, ij):                        # ij: (B, 2) long
        h = (ij[..., 0] * self._PRIMES[0]) ^ (ij[..., 1] * self._PRIMES[1])
        return h % self.hashmap

    def forward(self, x):                       # x: (B, 2) in [0,1]
        out = []
        for lv in range(self.n_levels):
            res = int(self.base_res * (self.scale ** lv))
            p = x * res
            f = torch.floor(p).long()
            w = p - f.float()
            acc = 0.0
            for dx in (0, 1):                   # bilinear over the 4 corners
                for dy in (0, 1):
                    c = f + torch.tensor([dx, dy], device=x.device)
                    wc = (w[..., 0] if dx else 1 - w[..., 0]) * (w[..., 1] if dy else 1 - w[..., 1])
                    acc = acc + wc[..., None] * self.emb[lv](self._hash(c))
            out.append(acc)
        return torch.cat(out, dim=-1)


def motion6_to_params(raw: torch.Tensor, *, trans_max_mm: float = 15.0,
                      rot_max_deg: float = 8.0) -> torch.Tensor:
    """(B,6) unbounded net output -> (B,6) [tx,ty,tz mm | wx,wy,wz rad].

    tanh bounds, as in AI_Geocal (which uses 10 mm / 10 deg). Zero raw output gives EXACTLY zero
    motion, so an untrained net starts at the nominal orbit and the estimator's cold start is the
    uncorrected geometry -- the same t=0 state the flow-matching prior was trained on.
    """
    t = trans_max_mm * torch.tanh(raw[..., 0:3])
    w = math.radians(rot_max_deg) * torch.tanh(raw[..., 3:6])
    return torch.cat([t, w], dim=-1)


class MotionNet6DoF(nn.Module):
    """view index -> 6-DoF rigid motion. Bandwidth is the whole design; see the module docstring.

    `enc`:
      "hash"    multiresolution hash grid (tcnn if available, else the torch fallback).
                DEFAULTS = the deployed FULLBAND settings (stock Instant-NGP).
      "fourier" explicit M-term Fourier features.
    """

    def __init__(self, n_views: int, *, enc: str = "hash",
                 n_levels: int = 16, n_features_per_level: int = 2, log2_hashmap_size: int = 15,
                 base_resolution: int = 16, per_level_scale: float = 1.5, fourier_m: int = 8,
                 width: int = 64, depth: int = 4,
                 trans_max_mm: float = 15.0, rot_max_deg: float = 8.0):
        super().__init__()
        self.n_views = n_views
        self.trans_max_mm, self.rot_max_deg = trans_max_mm, rot_max_deg

        if enc == "fourier":
            self.enc = FourierFeatureEncoder(fourier_m)
        elif HAVE_TCNN:
            self.enc = tcnn.Encoding(n_input_dims=2, encoding_config={
                "otype": "HashGrid", "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution, "per_level_scale": per_level_scale})
        else:
            self.enc = _HashGrid2D(n_levels, n_features_per_level, log2_hashmap_size,
                                   base_resolution, per_level_scale)

        layers, d = [], self.enc.n_output_dims
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(inplace=True)]
            d = width
        head = nn.Linear(d, 6)
        # zero-init the head so theta(v) == 0 at step 0 for every view, exactly.
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.mlp = nn.Sequential(*layers, head)

    def forward(self, v_idx: torch.Tensor) -> torch.Tensor:
        """(B,) view indices (long or float) -> (B, 6) motion parameters."""
        s = v_idx.float() / max(self.n_views - 1, 1)
        x = torch.stack([s, torch.zeros_like(s)], dim=-1)         # the grid wants >= 2 dims
        raw = self.mlp(self.enc(x).float())
        return motion6_to_params(raw, trans_max_mm=self.trans_max_mm, rot_max_deg=self.rot_max_deg)

    def all_params(self, device=None) -> torch.Tensor:
        """(V, 6) -- the whole trajectory."""
        v = torch.arange(self.n_views, device=device or next(self.parameters()).device)
        return self.forward(v)
