# Reproducibility notes

This release follows the CQ500 experiments in
[arXiv:2611.23788v1](https://arxiv.org/abs/2611.23788v1), submitted 18 November 2026.
The main method uses 30-node Akima interpolation with gradient descent. The
motion-estimator ablation uses 20 cubic B-spline controls with RMSprop; the
image-linear ablation retains Akima and changes only the prior's training bridge.

## Data and checkpoints

Download CQ500 separately and follow the
[preparation instructions](../README.md#prepare-the-data). The selected patient
IDs and series sizes are in the [split manifest](../configs/cq500_split.json):
150 training, 50 validation, and 30 test patients. Test patient `i` uses seed
`1000+i`, with the same simulated Akima motion across all three experiment arms.

Pretrained checkpoints are not distributed with the repository. The
[checkpoint manifest](../configs/checkpoints.json) identifies the original weights
by hash; it is not a download manifest. The README provides commands to train
geometry-bridge and image-linear priors for 500,000 iterations. The original
training configurations differ only in the bridge and output directory.

## Reference results

[Reference results](../configs/reference_results.json) come from all 30 saved
results for each of geometry/Akima, image-linear/Akima, and geometry/B-spline.
The incomplete 22-case B-spline progress summary is not used. There is no
image-linear/B-spline reference cohort.

The file retains full-precision means and population standard deviations
(`ddof=0`). Rounded paper entries can differ in their final digit or SD convention.
Image metrics use the stored scores evaluated before volume conversion to fp16;
RPE is recomputed from the saved trajectories with the paper's zero-centered
pose convention. Each cohort must have a consistent estimator and prior bridge.

## Recorded validation

The release was checked against the original experiment outputs and implementations:

- Eleven CPU tests for both estimators, deterministic paired launch commands, both
  training bridges, checkpoint compatibility, and preservation of existing outputs.
- Akima interpolation values and gradients, including zero initialization,
  against the released learned-autofocus interpolation code; spline units,
  optimizer resets, coefficient continuity, and deterministic view sampling.
- The packaged 360-by-20 cubic B-spline basis and its SHA256 checksum. Its
  [provenance](../fm3d/assets/bspline_provenance.json) records forward and gradient
  agreement with `torch-cubic-spline-grids==0.5.2`. That package is not needed at runtime.
- Independent CPU aggregation of all 90 archived results. All three arms use
  identical ground-truth motion for every paired patient.
- Architecture, bridge, iteration, and SHA256 checks for both original 500k
  checkpoints, plus agreement of the selected patients with the split manifest.
- Global-context patch aggregation, RPE geometry/gauge checks, Python compilation,
  shell syntax, documentation links, and source import checks.

These checks did not include new prior training, a new reconstruction cohort, or
CUDA operator execution. The reported metrics come from the retained experiment
outputs. New training and inference can vary with hardware and software versions.
Run the CUDA/LEAP and data-dependent gates listed in the README in the target environment.

## Execution details

- **Pose fitting:** Akima starts each fit at learning rate 1000 with 0.97 decay;
  B-spline uses RMSprop at 0.001 (alpha 0.99, epsilon 1e-8, no momentum, uncentered).
  Both retain coefficients across all 50 outer updates and the coarse-to-fine
  transition. RMSprop moments reset each outer update. No tanh bounds or pose
  regularization are applied. Simulation motion and estimated motion are separate.
- **JRM-ADM initialization:** The original inference initialization seed was not
  recorded. The adapter records a seed for new runs; this does not establish
  bitwise equality with the reference volumes.
- **Timing:** The paper reports about 9.1 minutes per geometry/Akima case on an
  RTX A6000: 548.565 seconds averaged over 29 isolated process records, excluding
  the contended patient 24 timing (all 30 remain in the accuracy results).
  The 1.9-minute autofocus comparison uses the accelerated backprojector, not
  original-kernel diagnostic timing. These times include setup, scoring, and
  output. Document hardware and concurrent GPU workload when comparing runtimes.
- **Version history:** The later Hash-INR implementation remains available at
  commit `465bf2f996ff2fa1555a24de4ef4904001d1d289`. The current commands and
  reference results use the submitted spline protocol.
