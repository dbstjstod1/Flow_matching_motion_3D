# Geometry-Bridge Flow Matching for Blind Rigid-Motion Correction in Head Cone-Beam CT

CQ500 reproduction code for the manuscript by Sungho Yun and Seungryong Cho (KAIST),
aligned with the **17 November 2026 manuscript**. This release contains the CQ500
simulation study: a geometry-bridge image prior, **16-level Hash-INR pose estimation**,
the image-linear training ablation, and the learned-autofocus / JRM-ADM comparisons.
Acquired clinical CBCT code, data and manuscript editing files are excluded.

```text
CQ500 CT → simulated moving projections → uncorrected FDK
repeat 50 times:
    predict image velocity with the frozen flow-matching prior
    fit rigid poses with Hash-INR + Adam (200 steps, 24 sampled views)
    refine the volume with CG (5 iterations) and relaxed TV
return refined volume, poses, and FDK at the estimated poses
```

## Install and prepare CQ500

Tested with Linux, Python 3.11, PyTorch 2.8 / CUDA 12.8, Triton 3.4 and an
RTX A6000 (48 GB). Training and reconstruction require an NVIDIA GPU.

```bash
pip install -r requirements.txt
pip install --no-build-isolation \
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@v2.0#subdirectory=bindings/torch'
```

Build the pinned, patched **LEAP** projector using
[the build instructions](third_party/leap/FM3D_PATCH.md). The reported Hash-INR
uses tiny-cuda-nn; the development fallback does not reproduce its numerical behavior.

Unpack [CQ500](http://headctstudy.qure.ai/dataset) under `data/CQ500`, then index it:

```bash
python scripts/prep_cq500.py --root data/CQ500
```

The launcher verifies the public patient IDs and series sizes against
[the split manifest](configs/cq500_split.json): **150 training, 50 validation, and
the first 30 held-out patients**. Each test patient uses seed `1000 + patient_index`.
All methods share the same patient, motion and projection data.

## Train and reconstruct

Run from the repository root. [configs/paper.json](configs/paper.json) contains the
fixed settings. Add `--dry-run` to inspect a command without loading data or CUDA.
Select a GPU with `CUDA_VISIBLE_DEVICES`.

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
Pose fitting uses a 16-level hash encoding, a 4×64 ReLU MLP, Adam at 0.003,
tanh bounds of ±15 mm / ±8°, and the mean squared projection residual.

Checkpoints and patient images are not bundled. Train the priors with the commands
above; [checkpoint hashes](configs/checkpoints.json) identify the archived weights
used for the reported results. New training runs can vary numerically across hardware
and software versions. Each inference case saves `launch.json`, `run.log` and
`result.pt`; the launcher reuses only completed cases with an identical launch record.
An interrupted or changed run requires a fresh output directory.

## Image-linear ablation and comparisons

Change only the prior's training bridge; retain the same reconstruction settings:

```bash
python scripts/reproduce.py train --bridge linear \
    --root data/CQ500 --out logs/fm3d_linbridge
python scripts/reproduce.py cohort --bridge linear \
    --root data/CQ500 --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth \
    --out data/linear_test30
python scripts/summarize_cohort.py data/linear_test30
```

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

Archived results on 30 patients (mean ± population SD):

| Training bridge / output | PSNR (dB) | SSIM | RPE (mm) |
|---|---:|---:|---:|
| Geometry / refined volume | 36.66 ± 1.58 | 0.978 ± 0.008 | 0.276 ± 0.086 |
| Geometry / estimated-pose FDK | 31.80 ± 0.76 | 0.753 ± 0.032 | 0.276 ± 0.086 |
| Image-linear / refined volume | 35.29 ± 2.28 | 0.966 ± 0.015 | 0.415 ± 0.167 |

[Reference results](configs/reference_results.json) retain full precision. The original
geometry cohort averaged about **9.0 minutes per patient**; GPU isolation was not recorded.
Test motion widths of 10 mm / 10° mean control points sampled in ±5 mm / ±5°
before zero-centering, rather than realized trajectory extrema.

## Checks and code

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python scripts/gate_context_unet.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_rpe.py
```

CUDA / patched-LEAP checks: `scripts/gate_geometry.py` and
`scripts/gate_leap_forward_tangent.py`. With CQ500 installed, also run
`scripts/gate_bridge_data.py` and `scripts/gate_bridge_linear.py`.
[Validation scope](docs/release_validation.md) records what was checked for this release.

`fm3d/` contains geometry, motion fitting, projectors, the image network and metrics;
`scripts/` contains preparation, training, reconstruction and evaluation;
`configs/` fixes the paper protocol; `bench/` and `baselines/` contain the comparison
adapters. Third-party source retains its license and provenance notices.
