"""Compact time-conditioned 3D U-Net velocity field for PATCH flow matching (prompt 6).

3D twin of `unet.py`, intended for 3D-PATCH inputs (e.g. 64^3, DiffusionBlend-style
patch prior — the FM prior cannot hold a full volume, so it is trained and evaluated
on patches and blended by `prior_patch.py`). The FM parameterization is unchanged
(clean endpoint: x1_hat = x_t + (1-t) * v). Default width is slimmer than the 2D
model (3D activations are heavy); patch size must be divisible by 2^(len(ch_mults)-1).

With `in_ch=5` the patch is conditioned on GLOBAL CONTEXT (arXiv:2512.18161):
channels are [patch x_t, downsampled full x_t, coord z, coord y, coord x] — built by
`prior_patch.make_tile_inputs` / `crop_pairs(ctx=...)`. Pure input conditioning: only
`in_conv` grows, the velocity output stays single-channel and predicts channel 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import timestep_embedding


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(t_emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet3D(nn.Module):
    """U-Net velocity field. forward(x, t) -> velocity, x: (B,C,D,H,W), t: (B,)."""

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 32,
        ch_mults=(1, 2, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        t_dim = base * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.time_base = base

        self.in_conv = nn.Conv3d(in_ch, base, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [base]
        ch = base
        for i, mult in enumerate(ch_mults):
            out = base * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(ch, out, t_dim))
                ch = out
                chs.append(ch)
            self.down_blocks.append(blocks)
            if i != len(ch_mults) - 1:
                self.downsamples.append(Down3D(ch))
                chs.append(ch)
            else:
                self.downsamples.append(None)

        # Bottleneck
        self.mid1 = ResBlock3D(ch, ch, t_dim)
        self.mid2 = ResBlock3D(ch, ch, t_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out = base * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock3D(ch + chs.pop(), out, t_dim))
                ch = out
            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Up3D(ch))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv3d(ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t_emb = self.t_mlp(timestep_embedding(t, self.time_base))

        h = self.in_conv(x)
        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for blk in blocks:
                h = blk(h, t_emb)
                skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), t_emb)
            if up is not None:
                h = up(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
