# Reproducibility notes

These notes describe the CQ500 experiment settings, evaluation conventions, and
validation for [Geometry-Bridge Flow Matching](https://arxiv.org/abs/2611.23788).

## Data and checkpoints

Download CQ500 separately and follow the
[preparation instructions](../README.md#prepare-the-data). The selected patient
IDs and series sizes are in the [split manifest](../configs/cq500_split.json):
150 training, 50 validation, and 30 test patients. Test patient `i` uses seed
`1000+i`, with the same simulated motion across all methods.

Pretrained checkpoints are not distributed with the repository. The
[checkpoint manifest](../configs/checkpoints.json) identifies the experiment
weights by hash. The README provides commands to train geometry-bridge and
image-linear priors for 500,000 iterations. The two training configurations
differ only in the bridge and output directory.

## Motion estimation

The main reconstruction and image-linear ablation use 30-node Akima interpolation
with gradient descent. The motion-estimator ablation uses 20-control-point cubic
B-splines with RMSprop and the same geometry-bridge prior.

| Setting | Akima + GD | B-spline + RMSprop |
|---|---|---|
| Control points per pose component | 30 | 20 |
| Learning rate | 1000, multiplied by 0.97 after each fitting step | 0.001 |
| Optimizer state at each outer update | Learning-rate schedule restarts | RMSprop moments reset |
| Spline coefficients across outer updates | Retained | Retained |

RMSprop uses alpha 0.99, epsilon 1e-8, no momentum, and no centering. Both
estimators retain their coefficients through the coarse-to-fine transition.
Coefficients are unbounded, in millimeters and degrees; output rotations are
axis-angle radians. No pose regularization is applied.

The packaged 360-by-20 cubic B-spline basis has a fixed SHA256 checksum.
Its [provenance](../fm3d/assets/bspline_provenance.json) records agreement of
forward values and gradients with `torch-cubic-spline-grids==0.5.2`; the package
is not required at runtime.

## Simulated motion

Ground-truth trajectories use Akima interpolation for every experiment.
Test motion widths of 10 mm / 10 degrees refer to control-point sampling in
±5 mm / ±5 degrees before zero-centering. The realized trajectory range can
differ because of sampling and interpolation. Ground-truth motion generation
and estimated-motion parameterization are separate settings.

## Reference results

[Reference results](../configs/reference_results.json) contain full-precision
statistics for three complete 30-patient cohorts: geometry/Akima,
image-linear/Akima, and geometry/B-spline, along with baseline comparisons.
Summary statistics use population standard deviations (`ddof=0`).

Image metrics use the stored scores evaluated before volume conversion to fp16.
RPE is computed from the saved trajectories using the ground-truth-independent
mean-pose removal described in the paper. Cohort summaries verify that all cases
use the same estimator, prior bridge, checkpoint, and inference settings.

## Validation

The recorded validation covers:

- Eleven CPU tests for both estimators, deterministic paired launch commands, both
  training bridges, checkpoint compatibility, and preservation of existing outputs.
- Akima interpolation values and gradients, including zero initialization,
  checked against the learned-autofocus interpolation implementation.
- Spline units, optimizer resets, coefficient continuity, deterministic view
  sampling, and the B-spline basis checksum.
- Independent CPU aggregation of all 90 experiment results, with matching
  ground-truth motion for each paired patient across the three cohorts.
- Architecture, bridge, iteration, and SHA256 checks for both 500k checkpoints,
  plus agreement of the selected patients with the split manifest.
- Global-context patch aggregation, RPE geometry/gauge checks, Python compilation,
  shell syntax, documentation links, and source import checks.

This validation used existing experiment outputs and did not repeat prior training,
a full reconstruction cohort, or CUDA operator execution. New training and
inference can vary with hardware and software versions. Run the CUDA/LEAP and
data-dependent checks listed in the README in the target environment.

## Timing

The main method averages 9.1 minutes per patient on an RTX A6000
(548.565 seconds across 29 isolated process records). A run with concurrent GPU
activity was excluded from timing; all 30 patients are included in accuracy
results. Times include initialization, reconstruction, evaluation, and output.

The 1.9-minute learned-autofocus comparison uses the accelerated backprojector.
Record the implementation, hardware, and concurrent GPU workload when comparing
runtimes.

## JRM-ADM initialization

The original inference initialization seed was not recorded. The adapter records
a seed for new runs; this does not establish bitwise equality with the reference
volumes.
