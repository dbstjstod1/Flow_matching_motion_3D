"""The frozen autofocus objective: a 3D U-Net that regresses the VIF* map from ONE volume.

TMI II-B.3, L334-343:

    "We then train a 3D U-net architecture [36] to regress the adjusted VIF map VIF* from the
     motion-affected volume. The U-net consists of basic building blocks with 3D convolutions and
     ReLU activation function. The number of feature maps per level is 8l with l = 1, ..., 4
     levels. Because we train a regression task, the final layer is a 1 x 1 convolution and no
     further final activation function is used. The model is trained with an L1-loss and Adam
     optimizer with a learning rate of 0.001 and a batch size of 16 with input and output volumes
     of size 128 x 128 x 128. After training, the predicted map is simply averaged to get a
     scalar value."

  [36] A. Wolny et al., eLife 9:e57613, 2020 (plant-seg) -- the cited U-Net.

WHY IT REGRESSES A MAP AND NOT A SCALAR (L354-367) -- this is the paper's central claim and the
reason the architecture is what it is:

    "In the gradient-based case, an informative volume gradient obtained via backpropagation
     through the trained network is equally important as the forward mapping. To ensure that an
     informative volume gradient is obtained, we propose to train a volume-to-volume network
     which regresses a full spatially resolved quality map. This is in contrast to existing
     approaches which directly regress a scalar quality metric using contracting architectures."

So do NOT "simplify" this to a scalar head -- a contracting architecture is the Huang et al.
baseline the paper beats, not the paper's method.

THE FEATURE-MAP WIDTHS -- SETTLED 2026-08-05, AND WE HAD IT WRONG
-----------------------------------------------------------------
The sentence is "The number of feature maps per level is 8^l with l = 1, ..., 4 levels"
(TMI 2025, II-B.3, p.1102). Every TEXT extraction of the paper -- the arXiv dump in
`refs/`, and `pdftotext` on the IEEE PDF -- flattens the superscript to "8l", and on that
reading we deployed the literal 8*l = (8, 16, 24, 32). Two pieces of evidence overturn it:

  1. the PUBLISHED PDF typesets it as an EXPONENT, `8^l`, which rules out 8*l. Taken at face
     value 8^l = (8, 64, 512, 4096), which is not a network anyone trains.
  2. the cited backbone is Wolny et al.'s plant-seg [36], and the author keeps her own fork of
     it at github.com/mareikethies/pytorch-3dunet, where

         def number_of_features_per_level(init_channel_number, num_levels):
             return [init_channel_number * 2 ** k for k in range(num_levels)]

     returns **(8, 16, 32, 64)** for `f_maps=8, num_levels=4` -- an exponential family, which
     is what the superscript is describing.

So the default is now (8, 16, 32, 64): 0.350 M parameters against the 0.170 M we ran before.
The old reading UNDER-PROVISIONED the baseline by ~2x, which is the direction a baseline must
never err in, so any comparison produced before this date is void. `f_maps=(8,16,24,32)`
restores it for an ablation. PROVENANCE.md section 4.4.

The paper says "3D convolutions and ReLU" and names no normalization; plant-seg's blocks are
GroupNorm-Conv-ReLU. Default here is `norm="none"` (the literal reading); `norm="group"` gives
plant-seg's block. Both are gated to run; only the default is "Thies as written".
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["QualityMetricUNet3D", "THIES_F_MAPS"]

THIES_F_MAPS = (8, 16, 32, 64)   # "8^l, l = 1..4" = plant-seg's init*2^k (see the header)
LITERAL_8L_F_MAPS = (8, 16, 24, 32)   # the pre-2026-08-05 reading; ablation only


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "group":
        # plant-seg uses 8 groups; fall back to whatever divides the channel count.
        g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
        return nn.GroupNorm(g, ch)
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    raise ValueError(f"norm must be none|group|batch, got {kind!r}")


class _Block(nn.Module):
    """The "basic building block": two 3x3x3 convolutions, each followed by ReLU."""

    def __init__(self, cin: int, cout: int, norm: str):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=(norm == "none")),
            _norm(norm, cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=(norm == "none")),
            _norm(norm, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class QualityMetricUNet3D(nn.Module):
    """(B,1,D,H,W) volume -> (B,1,D,H,W) predicted VIF* map. `.score(v)` gives the scalar.

    At optimization time the weights are FROZEN (L349-351: "at optimization time, the parameters
    of the quality metric network are fully trained and frozen") but the INPUT still needs a
    gradient, so callers must keep `requires_grad` on the volume and must NOT wrap the call in
    `torch.no_grad()`. `freeze()` sets `requires_grad_(False)` on the parameters only, which is
    exactly that distinction.
    """

    def __init__(self, f_maps=THIES_F_MAPS, *, in_ch: int = 1, out_ch: int = 1,
                 norm: str = "none"):
        super().__init__()
        f = tuple(int(v) for v in f_maps)
        if len(f) < 2:
            raise ValueError("need at least 2 levels")
        self.f_maps = f

        self.enc = nn.ModuleList()
        cin = in_ch
        for c in f:
            self.enc.append(_Block(cin, c, norm))
            cin = c
        self.pool = nn.MaxPool3d(2)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(f) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(f[i], f[i - 1], 2, stride=2))
            self.dec.append(_Block(2 * f[i - 1], f[i - 1], norm))

        # "the final layer is a 1 x 1 convolution and no further final activation function"
        self.head = nn.Conv3d(f[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) the scalar objective = the spatial MEAN of the predicted map (L367-369,
        "To obtain a scalar for optimization, we then simply average the volumetric quality map,
        ensuring that each spatial position has equal contribution to the final value")."""
        return self.forward(x).mean(dim=(1, 2, 3, 4))

    def freeze(self) -> "QualityMetricUNet3D":
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()
