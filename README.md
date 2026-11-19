# Geometry-Bridge Flow Matching

**[Geometry-Bridge Flow Matching for Blind Rigid-Motion Correction in Head Cone-Beam CT](https://arxiv.org/abs/2611.23788)**

**Sungho Yun and Seungryong Cho**  
Korea Advanced Institute of Science and Technology (KAIST)

[Paper](https://arxiv.org/abs/2611.23788) · [Installation](#installation) · [Data](#prepare-the-data) · [Training and reconstruction](#training-and-reconstruction) · [Results](#evaluation) · [Citation](#citation)

Official implementation and experiment code for our paper. This repository includes
the 3D geometry-bridge flow-matching prior, the blind motion-correction pipeline,
and the CQ500 experiments, ablations, and baseline comparisons.

The method learns how FDK reconstructions change as acquisition motion is reduced.
During reconstruction, learned image updates alternate with spline-based motion
estimation, projection-data consistency, and total-variation regularization.
The outputs are a refined image volume, an estimated motion trajectory, and an
FDK reconstruction at the estimated poses.

## Installation

The experiments use Linux, Python 3.11, PyTorch 2.8 / CUDA 12.8, Triton 3.4, and
an NVIDIA RTX A6000 (48 GB). Training and reconstruction require an NVIDIA GPU.

```bash
git clone https://github.com/dbstjstod1/Flow_matching_motion_3D.git
cd Flow_matching_motion_3D

pip install -r requirements.txt
```

Build the patched [LEAP projector](third_party/leap/FM3D_PATCH.md) before running
training or reconstruction. Both motion estimators are implemented in PyTorch;
the cubic B-spline interpolation basis is included in the repository.

## Prepare the data

Download [CQ500](http://headctstudy.qure.ai/dataset), unpack it under
`data/CQ500`, and build the index:

```bash
python scripts/prep_cq500.py --root data/CQ500
```

The [patient split](configs/cq500_split.json) contains 150 training, 50 validation,
and 30 test patients. Patient `i` in the test set uses seed `1000 + i`; all methods
are evaluated with the same patient, simulated motion, and projection data.

## Training and reconstruction

Pretrained weights are not currently provided. Train the prior first, then use
its 500,000-iteration checkpoint for reconstruction. The
[checkpoint manifest](configs/checkpoints.json) lists hashes of the weights used
in the experiments.

Run these commands from the repository root. Select a GPU with
`CUDA_VISIBLE_DEVICES`; add `--dry-run` to preview a training or reconstruction
command without loading data or CUDA.

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

The default reconstruction uses **30-node Akima splines with gradient descent**.
Each of 50 reconstruction updates includes 200 pose-fitting steps, with 24 views
sampled per step, followed by 5 CG iterations and TV refinement. Motion estimation
uses half-resolution volumes and detectors for the first 25 updates, then full
resolution. Image refinement runs at full resolution throughout.

Training uses 32³ patches with global context, batch size 64, AdamW with learning
rate 10⁻⁴ → 10⁻⁶, and EMA 0.999. The complete training and reconstruction settings
are in [configs/paper.json](configs/paper.json); optimizer and interpolation details
are in the [implementation notes](docs/release_validation.md#motion-estimation).

Each case saves `result.pt`, `launch.json`, and `run.log`. Completed runs with
identical settings can be reused. Choose a new output directory when changing
settings or restarting an interrupted run.

## Ablations and comparisons

### Image-linear training bridge

Train the image-linear prior and evaluate it with the same Akima-based
reconstruction settings:

```bash
python scripts/reproduce.py train --bridge linear \
    --root data/CQ500 --out logs/fm3d_linbridge
python scripts/reproduce.py cohort --bridge linear \
    --root data/CQ500 --ckpt logs/fm3d_linbridge/ckpt_iter500000.pth \
    --out data/linear_test30
python scripts/summarize_cohort.py data/linear_test30
```

### Motion estimator

Use the geometry-bridge prior with **20-control-point cubic B-splines and RMSprop**:

```bash
python scripts/reproduce.py cohort --estimator bspline_rmsprop \
    --root data/CQ500 --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --out data/bspline_test30
python scripts/summarize_cohort.py data/bspline_test30
```

### Baseline methods

See the [baseline guide](docs/baselines.md) for training and evaluating learned
autofocus and JRM-ADM on the same data.

## Evaluation

Results on the 30-patient CQ500 test set (refined volume, mean ± standard deviation):

| Training bridge / motion estimator | PSNR (dB) | SSIM | RPE (mm) |
|---|---:|---:|---:|
| Geometry / Akima + GD | **37.12 ± 1.65** | **0.980 ± 0.010** | **0.268 ± 0.090** |
| Image-linear / Akima + GD | 35.53 ± 2.24 | 0.968 ± 0.016 | 0.402 ± 0.178 |
| Geometry / B-spline + RMSprop | 36.93 ± 1.55 | 0.978 ± 0.009 | 0.270 ± 0.079 |

The geometry bridge improves PSNR by **1.59 dB** and reduces RPE by **33%** relative
to image-linear training. The main method takes approximately **9.1 minutes per
patient** on an RTX A6000. [Full-precision results](configs/reference_results.json)
include estimated-pose FDK scores and baseline comparisons; see the
[timing notes](docs/release_validation.md#timing) for measurement details.

`python scripts/summarize_cohort.py <cohort-directory>` writes `summary.json`
for a complete 30-patient cohort. It reports image quality and motion accuracy
and checks that the cases share the same experiment settings.

`result.pt` contains the refined volume (`x_t`), estimated-pose FDK (`x_final`),
and estimated and true motion trajectories (`theta`, `theta_true`). Translation
is in millimeters and rotation is in axis-angle radians. Image scores
(`final_xt`, `final`) are evaluated against CT after rigid alignment within the
measured field of view, using attenuation values without display-window clipping.
RPE is measured in detector-plane millimeters after mean-pose removal independent
of the ground truth. Use the stored image scores for reporting: they are computed
before saved volumes are converted to fp16.

## Checks

CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python scripts/gate_motion_schemes.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_context_unet.py
CUDA_VISIBLE_DEVICES='' python scripts/gate_rpe.py
```

With CUDA and patched LEAP installed, run `scripts/gate_geometry.py` and
`scripts/gate_leap_forward_tangent.py`. The bridge checks,
`scripts/gate_bridge_data.py` and `scripts/gate_bridge_linear.py`, also require CQ500.
See the [reproducibility notes](docs/release_validation.md) for validation details.

## Repository structure

| Directory | Contents |
|---|---|
| `fm3d/` | Image prior, spline motion estimation, projectors, and metrics |
| `scripts/` | Data preparation, training, reconstruction, and evaluation |
| `configs/` | Experiment settings, patient split, and reference results |
| `bench/`, `baselines/` | Learned-autofocus and JRM-ADM implementations and adapters |
| `third_party/` | LEAP patch and build instructions |

Third-party attribution and license notices are included with the corresponding code.

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
