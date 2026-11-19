# Reproducibility notes

This document describes the configuration, reference results, and recorded
validation for the code in this repository.

## Implementation and reference results

The reconstruction scripts use a 16-level Hash-INR pose estimator with Adam.
The [linked paper](https://arxiv.org/abs/2611.23788) uses spline-based pose fitting
for its main results and motion-estimator ablation. Those spline estimators are
not included in the current reconstruction scripts.

[Reference results](../configs/reference_results.json) record the existing
30-patient geometry-bridge and image-linear Hash-INR runs. They describe this
configuration, rather than the paper's spline-based results.

## Data and checkpoints

Download CQ500 separately and use the preparation instructions in the
[README](../README.md#prepare-the-data). The selected patient IDs and series sizes
are listed in the [split manifest](../configs/cq500_split.json).

Pretrained checkpoints are not distributed with the repository. The
[checkpoint manifest](../configs/checkpoints.json) contains hashes for the weights
used in the reference runs; it is not a download manifest. The README provides
commands for training new geometry-bridge and image-linear priors.

## Recorded validation

The validation record includes:

- Eight CPU tests covering all 30 paired launch commands, both training bridges,
  checkpoint bridge/iteration checks, output preservation, zero pose initialization,
  and persistent Adam state.
- Matching parameters between the released Hash-INR implementation and the
  reference research implementation after synthetic CPU optimization with the
  same initial seed and sampled views.
- Agreement between every inference configuration field and the archived
  metadata for all 60 cases, with implicit `lr=None` resolved to Adam 0.003.
- Architecture, bridge, and iteration checks for both original 500k prior
  checkpoints; agreement between the selected 150/50/30 patient IDs and the
  split manifest.
- Independent CPU aggregation of all 60 archived results, matching all eleven
  metric means and population standard deviations in the reference-results
  file (maximum absolute difference 1.11e-16).
- Global-context/patch aggregation and RPE geometry/gauge checks.
- Agreement between the supplied LEAP patch and the six locally patched source files.
- Python compilation, shell syntax, local documentation links, Git whitespace
  checks, and Python import consistency.

These checks did not include a new prior-training run, full reconstruction cohort,
or CUDA operator checks. The reference metrics come from the existing experiment
outputs. CUDA reductions, hardware, and software versions can affect new runs.

## Execution details

- **JRM-ADM initialization:** The original inference initialization seed was not
  recorded. The adapter records an explicit seed for new runs, but this does not
  establish bitwise equality with the reference volumes.
- **tiny-cuda-nn:** The original installed package reported version 2.0; its source
  commit was not recorded. The installation command pins the upstream v2.0 tag.
- **Timing:** The original geometry-bridge cohort averaged about 9.0 minutes per
  patient. Exclusive use of the GPU was not recorded, so compare runtimes with the
  hardware and concurrent GPU workload documented.
