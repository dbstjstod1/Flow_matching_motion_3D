# Geometry-Bridge Flow Matching for Blind Rigid-Motion Correction in Head Cone-Beam CT

**Sungho Yun and Seungryong Cho**  
Korea Advanced Institute of Science and Technology (KAIST)

**[Paper (arXiv v1)](https://arxiv.org/abs/2611.23788v1)** · [Installation](#installation) · [Training and reconstruction](#training-and-reconstruction) · [Evaluation](#evaluation) · [Citation](#citation)

This is the authors' code repository for the paper linked above. It provides the
3D geometry-bridge flow-matching prior and a CQ500 experiment workflow for training,
motion correction, and evaluation. This release follows arXiv v1: **Akima + GD**
for the main method and image-linear ablation, and **cubic B-spline + RMSprop**
for the motion-estimator ablation. It contains the public CQ500 experiments only.

The method learns how a reconstructed volume changes as acquisition motion is
reduced. During reconstruction, learned image updates alternate with motion
estimation and projection-data consistency to recover the image and head motion
jointly.

The repository includes:

- CQ500 preparation, patient splits, and simulated motion-corrupted projections.
- Training and reconstruction with geometry-bridge and image-linear priors.
- Image-quality and motion evaluation, with learned-autofocus and JRM-ADM comparisons.

## Installation

Tested with Linux, Python 3.11, PyTorch 2.8 / CUDA 12.8, Triton 3.4 and an
RTX A6000 (48 GB). Training and reconstruction require an NVIDIA GPU.

```bash
git clone https://github.com/dbstjstod1/Flow_matching_motion_3D.git
cd Flow_matching_motion_3D

pip install -r requirements.txt
```

Build the pinned, patched **LEAP** projector using
[the build instructions](third_party/leap/FM3D_PATCH.md). Both pose estimators run
in PyTorch; the B-spline interpolation basis is included with
[its provenance](fm3d/assets/README.md).

## Prepare the data

Download and unpack [CQ500](http://headctstudy.qure.ai/dataset) under `data/CQ500`, then index it:

```bash
python scripts/prep_cq500.py --root data/CQ500
```

The launcher verifies the public patient IDs and series sizes against
[the split manifest](configs/cq500_split.json): **150 training, 50 validation, and
30 held-out test patients**. Each test patient uses seed `1000 + patient_index`.
All methods share the same patient, motion and projection data.

## Training and reconstruction

Run from the repository root. [configs/paper.json](configs/paper.json) contains the
fixed settings. Add `--dry-run` to inspect a command without loading data or CUDA.
Select a GPU with `CUDA_VISIBLE_DEVICES`. The default estimator is `akima_gd`.

```bash
# Train the geometry-bridge prior (500,000 iterations).
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train \
    --root data/CQ500 --out logs/fm3d_databridge

# Reconstruct one test patient.
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py infer \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --patient 0 --out data/test_case/p00

# Reconstruct and summarize the full 30-patient cohort.
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py cohort \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --out data/test30
python scripts/summarize_cohort.py data/test30
```

Training uses 32³ patches, global context, batch 64, AdamW, learning rate
10⁻⁴ → 10⁻⁶, and EMA 0.999. The geometry bridge is selected by `--bridge data`:
`x_t = FDK(A_{(1-t)theta}(CT), P_nominal)`, with analytic velocity targets.
Pose fitting uses 30 Akima nodes and gradient descent at an initial learning rate
of 1000, decayed by 0.97 within each fit, on the mean squared projection residual.
Each of the 50 reconstruction updates includes 200 pose-fitting steps
(24 sampled views per step), 5 CG iterations, and relaxed TV. Spline coefficients
persist across updates; the learning-rate schedule restarts each update.

Pretrained checkpoints are not distributed with this repository. Train the priors
using the commands above; [checkpoint hashes](configs/checkpoints.json) identify
the weights used for the reference results below. New training runs can vary
numerically across hardware and software versions.

Each inference case saves `launch.json`, `run.log` and `result.pt`; the launcher reuses only completed cases with an identical launch record.
An interrupted or changed run requires a fresh output directory.

## Ablations and comparisons

Change only the prior's training bridge; retain the same reconstruction settings:

```bash
python scripts/reproduce.py train --bridge linear \
    --root data/CQ500 --out logs/fm3d_linbridge
python scripts/reproduce.py cohort --bridge linear \
    --root data/CQ500 --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth \
    --out data/linear_test30
python scripts/summarize_cohort.py data/linear_test30
```

For the motion-estimator ablation, use the same geometry-bridge prior with
20 cubic B-spline controls and RMSprop at 0.001:

```bash
python scripts/reproduce.py cohort --estimator bspline_rmsprop \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --out data/bspline_test30
python scripts/summarize_cohort.py data/bspline_test30
```

RMSprop uses alpha 0.99, epsilon 1e-8, no momentum, and no centering. Its moments
reset at each reconstruction update while the fitted coefficients persist.
Both estimators use unbounded coefficients in mm/degrees and return rotations in
axis-angle radians. The simulated ground-truth motion remains Akima for all arms.

[Comparison instructions](docs/baselines.md) cover learned autofocus and JRM-ADM,
including training, the common test inputs and evaluation.

## Evaluation

`result.pt` stores the refined volume (`x_t`), estimated-pose FDK (`x_final`),
estimated and true trajectories (`theta`, `theta_true`), and their image scores
(`final_xt`, `final`). Poses use mm and axis-angle radians. Image scores use
ground-truth CT after rigid alignment, inside the measured field of view, in
attenuation units without display-window clipping. RPE is in detector mm after
ground-truth-independent mean-pose removal. Report the stored image scores:
saved volumes are fp16, while the scores were evaluated before that conversion.

Reference results on the same 30 patients (refined volume, mean ± population SD):

| Training bridge / pose estimator | PSNR (dB) | SSIM | RPE (mm) |
|---|---:|---:|---:|
| Geometry / Akima + GD (main) | 37.12 ± 1.65 | 0.980 ± 0.010 | 0.268 ± 0.090 |
| Image-linear / Akima + GD | 35.53 ± 2.24 | 0.968 ± 0.016 | 0.402 ± 0.178 |
| Geometry / B-spline + RMSprop | 36.93 ± 1.55 | 0.978 ± 0.009 | 0.270 ± 0.079 |

[Reference results](configs/reference_results.json) retain full precision. The original
geometry/Akima cohort is reported at about **9.1 minutes per patient**. Timing conditions and
validation details are documented in the [reproducibility notes](docs/release_validation.md).
Test motion widths of 10 mm / 10° mean control points sampled in ±5 mm / ±5°
before zero-centering, rather than realized trajectory extrema.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python scripts/gate_motion_schemes.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_context_unet.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_rpe.py
```

CUDA / patched-LEAP checks: `scripts/gate_geometry.py` and
`scripts/gate_leap_forward_tangent.py`. With CQ500 installed, also run
`scripts/gate_bridge_data.py` and `scripts/gate_bridge_linear.py`.
[Reproducibility notes](docs/release_validation.md) describe the recorded checks,
checkpoint availability, and execution details.

## Repository structure

`fm3d/` contains geometry, motion fitting, projectors, the image network and metrics;
`scripts/` contains preparation, training, reconstruction and evaluation;
`configs/` contains the experiment settings, patient split, and reference results;
`bench/` and `baselines/` contain the comparison adapters. Third-party source retains its license and provenance notices.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yun2026geometrybridge,
  title   = {Geometry-Bridge Flow Matching for Blind Rigid-Motion Correction in Head Cone-Beam CT},
  author  = {Yun, Sungho and Cho, Seungryong},
  journal = {arXiv preprint arXiv:2611.23788},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.23788}
}
```
