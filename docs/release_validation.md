# CQ500 release validation — 17 November 2026

The public protocol follows the current manuscript's 16-level Hash-INR experiment.
Clinical data/adaptation, superseded pose-estimator ablations, duplicate launchers
and unrelated scanner presets are excluded. Existing research data and manuscript
files were not modified.

Verified for this release:

- Eight CPU tests cover all 30 paired launch commands, both training bridges,
  checkpoint bridge/iteration checks, output preservation, zero pose initialization
  and persistent Adam state.
- The released Hash-INR and original research implementation produce identical
  parameters after synthetic CPU optimization with the same initial seed and views.
- Every inference config field matches the archived metadata for all 60 cases
  (geometry and image-linear), with implicit `lr=None` resolved to Adam 0.003.
- Both original 500k prior checkpoints pass architecture/bridge/iteration checks;
  the complete dataset's selected 150/50/30 public IDs match the split manifest.
- Independent CPU aggregation of all 60 archived results matches all eleven metric
  means and population standard deviations in `configs/reference_results.json`
  (maximum absolute difference 1.11e-16).
- Global-context/patch aggregation and RPE geometry/gauge checks pass.
- The supplied LEAP patch matches the six locally patched source files.
- Python compilation, shell syntax, local documentation links and Git whitespace
  checks pass. Removed modules have no remaining Python import dependencies.

No new prior training, full reconstruction cohort, or CUDA operator gate was run
for this cleanup: CUDA was unavailable in the execution environment. Archived
results are reference outcomes, not results of a new end-to-end run. Checkpoints
and CQ500 images are not bundled; training instructions provide the route to new
weights. CUDA reductions and software versions can cause numerical variation.

The original JRM inference initialization seed was not recorded. The adapter now
records an explicit seed for future runs; this does not establish bitwise equality
with its archived volumes. The installed tiny-cuda-nn package reports version 2.0,
but its original source commit was not recorded; the installation command pins the
upstream v2.0 tag. Original proposed-method timing did not record GPU exclusivity.
