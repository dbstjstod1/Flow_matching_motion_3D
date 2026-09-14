# Geometry-Bridge Flow Matching for Blind Rigid-Motion Correction in Head Cone-Beam CT

Implementation of the manuscript by Sungho Yun and Seungryong Cho (KAIST).
The current paper configuration uses **Akima splines + gradient descent** for pose fitting,
with a frozen 3D flow-matching image prior, CG data consistency and relaxed TV updates.
The earlier hash-MLP estimator remains available as `--estimator net` for legacy experiments.

The geometry bridge learns image changes caused by progressively reducing simulated acquisition
motion while keeping the source anatomy fixed. Training uses
`x_t = FDK(A_{(1-t)theta}(x), P_nom)` and analytic forward-projector geometry derivatives
as velocity targets. During inference, the measured projections stay fixed:

```text
x = FDK(y, nominal poses); theta = 0
repeat 50 times:
    x_pred = x + dt * v_phi(x, t)
    theta  = fit poses to y using x_pred       # 200 Akima + GD steps
    z      = CG(A_theta, y, start=x_pred)      # 5 iterations, all 360 views
    x      = z + 0.3 * (TV_denoise(z) - z)
return x, theta, FDK(y, theta)
```

## Setup

Tested environment: Linux, Python 3.11, PyTorch 2.8 / CUDA 12.8, Triton 3.4,
RTX A6000 (48 GB). One GPU runs training or inference.

```bash
pip install -r requirements.txt
```

Build [LEAP](https://github.com/LLNL/LEAP) at commit `0c8846f` with the provided Joseph-kernel
patch; follow [third_party/leap/FM3D_PATCH.md](third_party/leap/FM3D_PATCH.md).
An unpatched projector is rejected. **tiny-cuda-nn is not required** for the paper's Akima or
B-spline estimators; it is optional for the legacy hash-MLP.

Obtain and unpack [CQ500](http://headctstudy.qure.ai/dataset) under `data/CQ500`, then run:

```bash
python scripts/prep_cq500.py --root data/CQ500
```

The paper uses 150 training patients, 50 validation patients and the **first 30** patients of
the remaining test split after series selection. The new launcher checks the public CQ500 IDs
and series sizes against [configs/cq500_split.json](configs/cq500_split.json), so an incomplete
dataset cannot silently shift the evaluated patients. The manifest contains no images or DICOM
paths. Datasets, checkpoints and generated results are not stored in Git.

## Reproduce the proposed method

All paper settings are explicit in [configs/paper.json](configs/paper.json). Run commands from
the repository root. Set `CUDA_VISIBLE_DEVICES` to select a GPU. Add `--dry-run` to any launcher
command to inspect the complete underlying command without loading data or CUDA.

Train the geometry-bridge prior:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py train \
    --root data/CQ500 --out logs/fm3d_databridge
```

Training uses 500,000 iterations, 32³ patches, batch 64, AdamW, cosine learning rate
10⁻⁴ → 10⁻⁶ and EMA 0.999. Use `ckpt_iter500000.pth` for paper evaluation.
[configs/checkpoints.json](configs/checkpoints.json) records the hashes of the evaluated
checkpoints; pretrained weights are not bundled. Training is the provided route to obtain a prior.

Reconstruct one test patient, or the complete paired cohort:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py infer \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --patient 0 --out data/test30/p00

CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py cohort \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --out data/test30

python scripts/summarize_cohort.py data/test30
```

The cohort fixes `(split=test, run=i, seed=1000+i)` for `i=0..29`. Each case records its command,
source hashes, checkpoint hash and completion status in `launch.json`; progress is in `run.log`.
A repeated cohort command skips only successful cases with the same launch manifest.
Incomplete or incompatible outputs are preserved and require a fresh output directory.
Training checkpoints can be resumed through `scripts/train_fm3d.py --resume` with the same
training settings; the cohort launcher does not resume a partly reconstructed volume.

The paper reports about **9.1 minutes per patient** on an RTX A6000. Runtime covers the complete
process; GPU load, compilation and software versions affect it. CUDA atomic reductions can also
cause small numerical variation even with fixed seeds.

### Outputs and evaluation

`result.pt` contains:

| Key | Meaning |
|---|---|
| `x_t` | Final refined volume (primary output; saved as fp16) |
| `x_final` | FDK at the estimated poses (saved as fp16) |
| `theta`, `theta_true` | `(360, 6)` poses: translations in mm, axis–angle rotation vectors in radians |
| `final_xt`, `final` | Stored full-precision image scores for the final iterate and estimated-pose FDK |
| `theta_hist` | Pose estimates after each outer update |

Both image outputs are scored against ground-truth CT after rigid alignment, within the measured
field of view. Metrics use attenuation values, not the display HU window: GT maximum attenuation
in the mask sets the PSNR peak and SSIM range, with a uniform 7×7×7 SSIM window.
RPE is measured in detector-plane mm after ground-truth-independent mean-pose removal.
Estimated-pose FDK applies no learned image update, but its poses were recovered with the prior's aid.
The motion-free training FDK endpoint and true-pose FDK of the corrupted scan are distinct references.

Motion amplitudes of 10 mm / 10° are **full control-point sampling widths**: nodes are drawn in
±5 mm / ±5° before zero-centering. They do not guarantee a realized peak-to-peak trajectory range.

Archived 30-patient reference results (mean ± population SD):

| Output | PSNR (dB) | SSIM | RPE (mm) |
|---|---:|---:|---:|
| Proposed, final iterate | 37.12 ± 1.65 | 0.980 ± 0.010 | 0.268 ± 0.090 |
| Proposed, estimated-pose FDK | 31.93 ± 0.89 | 0.753 ± 0.032 | 0.268 ± 0.090 |

These are the manuscript's archived results, not a new training run performed for this release.
Full-precision reference summaries are in [configs/reference_results.json](configs/reference_results.json).

## Ablations

**Training bridge:** train an image-linear prior with the same endpoints, architecture and budget,
then use the same inference loop:

```bash
python scripts/reproduce.py train --bridge linear --root data/CQ500 --out logs/fm3d_linbridge
python scripts/reproduce.py cohort --bridge linear --root data/CQ500 \
    --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth --out data/linear_test30
python scripts/summarize_cohort.py data/linear_test30
```

**Pose-fitting scheme:** retain the geometry-bridge prior and replace 30-node Akima + GD
(initial step 1000, decay 0.97 within each fit) with 20-control-point cubic B-splines + RMSprop
(step 0.001). The spline basis used in the experiment is bundled with its provenance.

```bash
python scripts/reproduce.py cohort --estimator bspline_rmsprop --root data/CQ500 \
    --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --out data/bspline_test30
python scripts/summarize_cohort.py data/bspline_test30
```

Step sizes were selected by minimum mean zero-centered RPE on validation indices 0, 1, 2
(seeds 2000, 2001, 2002), before test evaluation. For a validation sweep or a modified protocol,
use `scripts/run_posterior3d.py` directly with explicit `--split val --run ... --seed ... --lr ...`.
Its default estimator is also Akima + GD.

## Comparison methods

- **Learned autofocus:** `scripts/bench_thies_train_qm.py` and `scripts/bench_thies_estimate.py`.
  Use `scripts/bench_thies_original.py` with `FM3D_THIES_VENDOR_BP=1` for the original released
  CUDA kernels used in the manuscript's runtime comparison. See
  [docs/baselines.md](docs/baselines.md) for commands and metric scope.
- **JRM-ADM:** the authors' sampler and optimizer stack, with our retrained prior and the common
  360-view scans. Export/retraining/running/scoring instructions are in
  [baselines/jrm_adm/README.md](baselines/jrm_adm/README.md).

## Code map and checks

| Path | Role |
|---|---|
| `fm3d/spline_motion.py` | Paper's Akima + GD and B-spline + RMSprop estimators |
| `fm3d/rigid_motion.py`, `geometry_3d.py` | Motion simulation, acquisition geometry and pose evaluation |
| `fm3d/leap_projector.py`, `triton_leap_grad.py` | Patched LEAP operator and analytic geometry derivatives |
| `fm3d/unet_3d.py`, `prior_patch.py` | Velocity network, global context and patch aggregation |
| `scripts/train_fm3d.py` | Geometry/image-linear bridge training |
| `scripts/run_posterior3d.py` | Joint reconstruction loop and final image scoring |
| `scripts/reproduce.py`, `summarize_cohort.py` | Fixed paper protocol, cohort launching and reporting |

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python scripts/gate_motion_schemes.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_context_unet.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_rpe.py
```

Operator checks requiring CUDA and patched LEAP: `scripts/gate_geometry.py` and
`scripts/gate_leap_forward_tangent.py`.
Checks requiring CQ500: `scripts/gate_bridge_data.py` and `scripts/gate_bridge_linear.py`.
Third-party code keeps its original licenses and provenance notices. The study uses simulated
projections from clinical CT volumes; acquired-CBCT validation is outside this release.
