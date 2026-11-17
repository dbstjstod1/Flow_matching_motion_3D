# Learned autofocus: source and comparison protocol

The benchmark implements Thies et al., *A gradient-based approach to fast and
accurate head motion compensation in cone-beam CT* ([paper](https://arxiv.org/abs/2401.09283)),
on this repository's shared CQ500 measurements.

| Component | Source |
|---|---|
| `vendor/geometry_gradients_CT/` | [mareikethies/geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT), commit `f6b2b2cc6b801e4c575106b554df70ac5ce90c92`; Apache-2.0, with its license retained |
| `vendor/moco_diff_likelihood/akima_spline.py` | [mareikethies/moco_diff_likelihood](https://github.com/mareikethies/moco_diff_likelihood), copied 2026-07-28 from the authors' 2D pipeline; the original copy did not record a commit SHA |
| `qmnet.py`, `data.py`, `motion.py`, `recon.py`, `vif.py` | Our implementation of the quality network, sampling, optimizer inputs and metrics from the paper |
| `fast_backprojector.py` | Our accelerated implementation of the released kernel arithmetic, with a different reduction order |

The vendored files and their legal notices are retained. The released geometry
code supplies the backprojector and geometry gradients, not a complete 3D training
pipeline. The Akima interpolator comes from a 2D sibling project; it does not
establish an exact replication of the authors' unreleased 3D training code.

The quality network is trained on our 150 training patients using motion-free
projections reconstructed at perturbed geometries, supervised by voxelwise VIF*
with L1 loss. Training uses clipped half-normal per-component amplitudes bounded
by nominal widths 15 mm / 20 degrees, batch 16, Adam at .001 and 20,000 iterations.
The reported checkpoint is selected by the lowest mean validation L1 over fixed
draws for all 50 validation patients (archived checkpoint: iteration 19,500).
Validation RPE is a diagnostic and does not select the checkpoint.

At evaluation, the method estimates 30 Akima control points per component with
100 gradient-descent iterations (`s0=100`, decay .97). Pose fitting uses a 128³
volume at 2 mm and final reconstruction uses 256³ at 1 mm. All methods share the
same 360-view scans and nominal test sampling widths 10 mm / 10 degrees. These
acquisition and amplitude settings define our comparison, rather than the
original paper's own benchmark cohort.

Use `FM3D_THIES_VENDOR_BP=1` and `scripts/bench_thies_original.py` for the authors'
original CUDA kernels and the manuscript's runtime convention. The default
`bench_thies_estimate.py` path uses the accelerated kernels; floating-point
reduction order can differ, and its elapsed time must be identified separately.
`backend.json` records which implementation was selected. `gate_bench_thies.py`
includes a numerical comparison between the two kernel paths.

The native benchmark reports several reference/ROI conventions. For the manuscript,
rescore `out_vol` against the ground-truth CT with the common measured-FOV mask and
rigid alignment, and use zero-centered trajectory RPE. The additional VIF and
body-ROI scores are not substitutes for those table columns. Instructions and
optional CUDA dependencies are in [docs/baselines.md](../../docs/baselines.md).
