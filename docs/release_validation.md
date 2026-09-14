# Paper-code update: verification scope

The September 2026 update replaces the default hash-MLP pose fit with the paper's
30-node Akima + GD scheme and includes the 20-control-point B-spline + RMSprop ablation.
The public reconstruction loop retains the paper's CG-only, one-TV-update path; local
exploratory solvers and unpublished diagnostic runs are not needed to execute it.

Checks completed for this update:

- Akima values and gradients match the released Thies interpolation, including all-zero
  initialization. Both spline schemes pass unit conversion, update continuity, deterministic
  sampled-view optimization and step-schedule checks.
- Launcher tests parse all 30 case commands, both training-bridge configurations, the pose
  ablation and the default inference configuration. Completed runs are reused only with an
  identical launch identity; incompatible and nonempty output directories are preserved.
- CPU checks pass for RPE units/gauge behavior and global-context patch aggregation.
- Both archived 500k-iteration checkpoints pass the shape/architecture/bridge checks.
  The CQ500 selection matches the 150/50/30-case manifest.
- The new CPU cohort summarizer reads the archived 30 proposed reconstructions and reproduces
  mean PSNR 37.12106107076009 dB, SSIM 0.9795855422814687 and RPE
  0.2679147911672178 mm, including the estimated-pose FDK and component MAE statistics.
- The autofocus wrapper selects the authors' released backprojector, verified by class and
  source hash without running reconstruction.
- Python compilation, shell syntax and Git whitespace checks pass.

This update did not retrain a prior or repeat the complete GPU reconstruction cohort.
Both local GPUs were occupied by existing experiments. The full geometry/operator gates
require CUDA and patched LEAP; they are listed separately from CPU checks in the README.
Fixed seeds do not guarantee bitwise equality for CUDA atomic reductions or across software
versions. The reference results in `configs/reference_results.json` remain archived outcomes.

Patient images, model weights, local experiment queues and manuscript editing artifacts are
not included. The small B-spline array is interpolation geometry, with a recorded source
package/version and checksum in `fm3d/assets/`.
