# Thies et al. — released motion-simulation code (verbatim copy, 2026-07-28)

Source: <https://github.com/mareikethies/moco_diff_likelihood> (branch `main`), the repo for
**Thies et al., "Differentiable Score-Based Likelihoods: Learning CT Motion Compensation From
Clean Images", MICCAI 2024** — the *2D fan-beam* sibling of the TMI 2025 paper we target
(`refs/thies_2401.09283_TMI2025.txt`).

**Why these files.** The TMI paper releases only its analytic backprojection Jacobian
(<https://github.com/mareikethies/geometry_gradients_CT>) — no motion sampler, no training
pipeline — and defers its training sampler to a supplementary pseudo-code that is not in the
arXiv entry (v1 2024-01-17 / v2 2024-10-21, no ancillary files). This repo is the closest
released code, and its constants match the TMI paper exactly: CQ500, 360 views, `num_nodes=10`,
`max_translation=10` mm, `max_rotation=0.26` rad (= 15°), `do_zero_center=True`, Akima splines.

**Caveat that must travel with any claim made from these files:** this is the *2D 3-DoF*
(r, tx, ty) code, not the TMI *3D 6-DoF* code, which is unreleased. The amplitude *convention*
below is consistent across both files here and is independently confirmed by the TMI paper's own
initial-RPE figure, so it is on firm ground; the exact 3D sampler is not.

## Files

| file | what it is |
|---|---|
| `motion_compensation_data_loader.py` | **the EVALUATION motion sampler** (`amplitude_rotation=5, amplitude_translation=5`) |
| `autofocus_data_set.py` | **the TRAINING motion sampler** for their quality-metric net (max 10 mm / 15°) |
| `motion_models/motion_models_2d_torch.py` | the spline motion model; `spline_akima`, `is_radian` handling, `do_zero_center` |
| `motion_models/akima_spline.py` | Akima interpolation |
| `default_motion_configs.py` | score-model config (not motion — kept for completeness) |

## THE FINDING: "amplitude" is PEAK-TO-PEAK, not the ± bound

`motion_compensation_data_loader.py:30-32`, with `is_radian=False` (i.e. degrees) at line 41:

```python
amplitude_rotation = 5      # the paper's "5 deg"
amplitude_translation = 5   # the paper's "5 mm"
r  = (torch.rand(num_nodes) - 0.5) * self.amplitude_rotation      # -> U(-2.5 deg, +2.5 deg)
tx = (torch.rand(num_nodes) - 0.5) * self.amplitude_translation   # -> U(-2.5 mm,  +2.5 mm)
```

So *"a random motion pattern ... with an amplitude of 5 mm for translation and 5° for rotation"*
(TMI IV, L501-506) means node values drawn from **±2.5**, not ±5. Our `akima_motion` drew
`uniform(-amp, +amp)`, i.e. **exactly 2× their node amplitude**.

Independently confirmed against the paper's own numbers — TMI L568 reports *"an initial median
RPE of around 3 mm"* for the uncompensated scan at this setting:

| node draw | initial median RPE (our cfg) | (at SOD 785 / SDD 1200) |
|---|---|---|
| `U(-5, +5)` — what we had | 6.10 mm | 6.23 mm |
| `U(-2.5, +2.5)` — theirs | **3.05 mm** | **3.11 mm** |
| paper's stated figure | — | **"around 3 mm"** |

## Their TRAINING amplitude sampler

`autofocus_data_set.py:64-87`:

```python
self.max_rotation = 0.26      # "this corresponds to 15 deg"
self.max_translation = 10     # [mm]

std_rot = self.max_rotation * torch.rand(1)          # per-DoF, unequal  <- the paper's clause
std_tx  = self.max_translation * torch.rand(1)
std_ty  = self.max_translation * torch.rand(1)

r  = (torch.rand(num_nodes) - 0.5) * min(abs(torch.normal(0., std_rot)), self.max_rotation)
tx = (torch.rand(num_nodes) - 0.5) * min(abs(torch.normal(0., std_tx)),  self.max_translation)
ty = (torch.rand(num_nodes) - 0.5) * min(abs(torch.normal(0., std_ty)),  self.max_translation)
```

Two things this settles:
1. **Per-DoF unequal amplitude is a `U(0,1)` scaling — of the *std*, not of the amplitude.**
   The amplitude itself is then a **clipped half-normal** `min(|N(0, std_d)|, A_d)`.
2. **The *"perturb the data only slightly"* clause is NOT a separate branch** — it is the
   half-normal's mass near zero. (We do not need it: our geometry bridge sweeps residual
   amplitude uniformly over [0, |theta|] on every draw, exactly. See the amplitude memory.)

## A probable unit bug in THIS repo (do not propagate)

`autofocus_data_set.py` sets `max_rotation = 0.26` — radians, per its own comment "corresponds to
15°" — but passes `is_radian=False` at line 84, and `spline_akima` then does `r = r/180*pi`. So
the training rotations there are effectively **±0.13 degrees**, ~115× smaller than intended. The
evaluation loader is self-consistent (`5` passed as degrees). The TMI paper says 15°, so the
intent is unambiguous; only this file's plumbing disagrees.
