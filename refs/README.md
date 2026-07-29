# Reference papers, kept locally so claims can be checked against the source

## `thies_2401.09283_TMI2025.txt`
M. Thies et al., *A gradient-based approach to fast and accurate head motion compensation in
cone-beam CT*, IEEE Trans. Medical Imaging, 2025. arXiv:2401.09283, DOI 10.1109/TMI.2024.3474250.
Author's accepted version, full text (1074 lines / ~12.5k words): I. Introduction, II. Methods,
III. Data, IV. Experiments and Results, V. Discussion, References, VI. Conclusion.

**This is THE paper that defines our task** (CQ500 head, geometry, 256³ @ 1 mm, Akima motion,
RPE). It is NOT the paper that defines our 3D UNet — that is arXiv:2512.18161, which is LIDC
*lung* and whose numbers are not our target. Do not mix them.

**READING CAVEAT.** Extracted from a two-column PDF, so the left and right columns are
INTERLEAVED on each line — a single line often contains the end of one sentence from column 1 and
an unrelated fragment from column 2. Reading it linearly will produce nonsense. Use `grep -n` for
a term, then read a ±10-line window and mentally separate the columns. Every quotation taken from
this file for the project memories was reconstructed that way and checked for sense.

### The numbers we keep coming back to (verified against this file, line numbers as of 2026-07-26)
| what | value | line |
|---|---|---|
| **evaluation** motion amplitude | **5 mm translation / 5° rotation**, one random pattern per patient, held constant across methods | 504 |
| motion model | Akima spline, 10 nodes to simulate, 30 to estimate (180 dof) | II-B.1 |
| **quality-net TRAINING** motion | max **10 mm / 15°**, resampled every time a sample is used, incl. unequal-per-DoF and barely-perturbing patterns | 482 |
| geometry | 360 views over 2π, SID 785 / SDD 1200, detector 500×700 @ 0.64 mm, ramp+cosine | 473 |
| grids | motion estimated on **128³ @ 2 mm**; image results on **256³ @ 1 mm** | 500-508 |
| optimizer | plain GD, 100 iterations, s0 = 100, exponential decay t = 0.97, x⁰ = 0 | 378-382 |
| quality net | 3D UNet, features 8·l (l=1..4), L1 loss, Adam lr 1e-3, batch 16, 128³ | 330-343 |
| dataset | CQ500 491 → 320, split 150/50/120; 30 test patients evaluated | III |
| headline | RPE 3.00 → **0.61 mm**; SSIM 0.83 → **0.94**; RMSE 120.75 → 58.49 HU | Table I |
| eval protocol | all motion-compensated recons **rigidly registered to their GT recon in 3D** before scoring | 479-481 |

Note the paper's own train/eval amplitude asymmetry: the quality net sees 10 mm / 15° in training
but is evaluated at 5 mm / 5°. Ours is now aligned at 5/5 everywhere (train, val, inference).

See also the project memories: `thies-method-structure`, `thies-amplitude-mismatch`,
`two-reference-papers`, `cq500-and-standard-geometry`.
