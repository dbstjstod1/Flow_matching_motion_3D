# Rigid-Motion Correction in Head CBCT with a 3D Flow-Matching Prior on the Geometry Bridge

Blind rigid patient-motion correction for head cone-beam CT: the volume and the per-view 6-DoF
poses are recovered jointly from a single motion-corrupted scan. A 3D flow-matching prior is
trained on the **geometry bridge**, the family of FDK reconstructions of the same scan with the
motion attenuated linearly to zero, and a predictor-corrector loop alternates the frozen prior, a
hash-encoded motion estimator and a conjugate-gradient data-consistency step along that bridge.

This repository contains the implementation used for the paper (S. Yun and S. Cho, KAIST).
An arXiv link will be added on release.

## Method in one screen

```
training      x_t = FDK( A(x; P_nom T((1-t) theta)), P_nom ),  t ~ U[0,1]
              dx_t/dt = FDK( -(dA/dP)[x; P((1-t) theta)] . P_nom Tdot theta, P_nom )   (closed form)
              loss = || v_phi(x_t, t) - dx_t/dt ||^2

inference     x <- FDK(y, P_nom), theta <- 0
              for k = 1..N:  x_pred <- x + dt v_phi(x, t)                       predict
                             theta  <- argmin || A_{P_nom T(theta)} x_pred - y ||  estimate
                             x      <- CG(theta, y; x_pred), then relaxed TV       correct
              return x (final iterate) and FDK(y, P_nom T(theta))
```

Motion is a per-view rigid transform right-multiplied into the projection matrices,
`P(theta)[v] = P_nom[v] T(theta_v)`, with the rotation stored as an axis-angle vector so that
`(1-t) theta` is a geodesic. The volume is never warped; the geometry derivative `dA/dP` and the
pose gradient of the estimator are exact derivatives of the same projector.

## Layout

| path | what it is |
|---|---|
| `fm3d/geometry_3d.py` | cone-beam geometry (`ConeBeam3DConfig.thies()`: SID 785 / SDD 1200 mm, 700x500 panel @ 0.64 mm, 360 views), projection matrices, measured-region mask |
| `fm3d/rigid_motion.py` | 6-DoF poses -> `(V,4,4)` transforms, Akima motion sampler, bridge geometry `P(s)` and `dP/ds`, reprojection error, SE(3) gauge utilities |
| `fm3d/leap_projector.py`, `fm3d/triton_leap_grad.py`, `fm3d/projector_3d.py`, `fm3d/filters.py` | the operator: LEAP modular-beam forward (Joseph, pinned) and backprojection, FDK, and our exact geometry derivatives of the same kernels (estimator gradient, bridge tangent) |
| `fm3d/dataset_cq500.py` | CQ500 indexing, Thies' series selection and 150/50/rest split, native-grid (612^3 @ 0.42 mm) simulation, static-FDK memo |
| `fm3d/unet_3d.py`, `fm3d/prior_patch.py` | the 3D U-Net velocity net and its patch-wise evaluation with the global-context conditioning channels |
| `fm3d/motion_estimation.py`, `fm3d/motion_net.py` | the motion estimator: hash-encoded MLP over the view index, projection-domain losses |
| `fm3d/tv.py`, `fm3d/reg_metric.py` | TV denoiser; rigid-align-then-score metrics (PSNR/SSIM after removing the SE(3) gauge) |
| `scripts/prep_cq500.py` | index CQ500, apply the selection, print the split |
| `scripts/train_fm3d.py` | train the prior on the geometry bridge (`--bridge linear` = the ablation arm) |
| `scripts/val_fm3d.py` | prior-only ODE validation of a checkpoint |
| `scripts/run_posterior3d.py` | the inference loop (Algorithm 1), one patient |
| `scripts/run_cohort_ours.sh` | the 30-patient test cohort driver |
| `scripts/render_posterior3d.py`, `scripts/rpe_report.py`, `scripts/cmp_arms.py`, `scripts/roi_body_metrics.py` | deferred metric rendering, RPE readout, paired cross-arm comparison, body-ROI re-scoring |
| `scripts/gate_*.py` | self-checking gates (geometry, operator tangent, both bridges' endpoints, conditioning channels, RPE units, Thies baseline wiring) |
| `bench/thies/` + `scripts/bench_thies_*.py`, `scripts/cmp_thies_vs_ours.py` | the learned-autofocus baseline (Thies et al., TMI 2025) on our data; vendored upstream code in `bench/thies/vendor/` |
| `baselines/jrm_adm/` + `scripts/export_cohort_for_jrm.py`, `scripts/jrm_theta_convert.py`, `scripts/score_jrm_native.py` | glue for running the released JRM-ADM code on our cohort and scoring it in our convention |
| `third_party/leap/` | the one patch we apply to LEAP (pin the modular-beam forward to the Joseph kernel) and how to build it |

## Setup

Tested with Python 3.11, PyTorch 2.8 (CUDA 12.8), Triton 3.4, two RTX A6000 (48 GB).

1. Python packages: `pip install -r requirements.txt`, then
   [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`)
   for the hash-encoded motion estimator.
2. **LEAP with our patch** (the projector). Clone [LLNL/LEAP](https://github.com/LLNL/LEAP) at
   commit `0c8846f`, apply `third_party/leap/leap_joseph_pin.patch`, build, and install the
   resulting `libleapct.so` next to `leapctype.py` in site-packages; see
   `third_party/leap/FM3D_PATCH.md`. `fm3d/leap_projector.py` refuses to run on an unpatched
   library.
3. **CQ500** ([Qure.ai](http://headctstudy.qure.ai/dataset), CC BY-NC-SA 4.0): unpack the DICOM
   tree under `data/CQ500/` and run

   ```
   python scripts/prep_cq500.py --root data/CQ500
   ```

   which applies the thin-slice selection and the sequential patient-level split
   (150 train / 50 val / rest test) and reports what was dropped.

Gates that need neither data nor a checkpoint:

```
python scripts/gate_geometry.py             # motion enters the geometry correctly, bridge is monotone
python scripts/gate_leap_forward_tangent.py # exact geometry derivative vs a float64 autograd jvp
python scripts/gate_context_unet.py         # conditioning channels and tile blending (CPU)
python scripts/gate_rpe.py                  # reprojection error in Thies' units, gauge split
```

## Training the prior

```
python scripts/train_fm3d.py --out logs/fm3d_databridge
```

The defaults are the run used in the paper: 256^3 @ 1 mm volumes, 32^3 patches with the four
conditioning channels (in_ch = 5), batch 64, a rolling cache of 8 whole-volume bridge draws
refreshed every 12 steps, 500k iterations, AdamW with a cosine schedule 1e-4 -> 1e-6, EMA 0.999,
fp16 AMP, projections simulated on the native 612^3 grid, training amplitudes up to 15 mm / 20 deg
peak-to-peak (per-DoF, Thies' protocol). One A6000, about 3 days.

`--bridge linear` trains the bridge ablation of the paper (pixel-linear path between the same two
endpoints, constant velocity target). `gate_bridge_data.py` and `gate_bridge_linear.py` check both
bridges' endpoints and tangent wiring on real data.

Prior-only validation of a checkpoint (the inline validation of the trainer, standalone):

```
python scripts/val_fm3d.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth --patients 3
```

## Inference

One patient of the test split, the paper's setting (Akima motion, 10 mm / 10 deg peak-to-peak,
N = 50 flow steps, 200 estimator iterations per step on 24 random views, coarse-to-fine until
t = 0.5, five CG iterations, TV weight 0.3):

```
python scripts/run_posterior3d.py --ckpt logs/fm3d_databridge/ckpt_iter500000.pth \
    --split test --run 0 --seed 1000 --out data/test30/p00
```

About 9 minutes per patient on one A6000. `result.pt` holds the final iterate `x_t`, the
learning-free `FDK(theta_hat)`, the recovered trajectory and aligned PSNR/SSIM against the ground
truth and against the motion-free FDK; `final.png` is the montage. With the default
`--metric_mode defer` the per-step montages are produced afterwards by
`python scripts/render_posterior3d.py --out data/test30/p00`.

The 30-patient cohort of the paper is `(split=test, run=i, seed=1000+i)`, i = 0..29:

```
scripts/run_cohort_ours.sh data/test30
python scripts/rpe_report.py data/test30/p*/result.pt
```

Every comparison method is run on the same triples, so the comparisons are paired.

## Baselines

* **Learned autofocus (Thies et al., IEEE TMI 2025).** `scripts/bench_thies_train_qm.py` trains
  the quality-metric network on our 150 training patients under their protocol;
  `scripts/bench_thies_estimate.py` runs their 100-iteration descent on one cohort triple;
  `scripts/cmp_thies_vs_ours.py` pairs the two cohorts. The backprojector and its geometry
  gradient are the authors' released code (`bench/thies/vendor/`, Apache-2.0); see
  `bench/thies/PROVENANCE.md`.
* **JRM-ADM (De Paepe et al., IEEE TRPMS 2025).** Run with the authors' released code as
  published. `scripts/export_cohort_for_jrm.py` writes our cohort measurements in their format,
  `baselines/jrm_adm/` holds the driver and the prior-retraining script that go into their
  repository, and `scripts/score_jrm_native.py` scores their outputs in our convention.

## Evaluation conventions

* Blind motion correction has an exact SE(3) gauge: a global rigid transform of the object and
  the orbit leaves the sinogram unchanged. Volumes are rigidly aligned to the ground truth before
  PSNR/SSIM, and the reprojection error is computed after removing the mean pose offset.
* PSNR/SSIM are read within the scanner's measured field of view.
* All amplitudes in this code are **peak-to-peak** (spline nodes drawn from U(-A/2, A/2)).

## Citation

```
@article{yun2026geometrybridge,
  title   = {Rigid-Motion Correction in Head Cone-Beam CT with a 3D Flow-Matching Prior on the Geometry Bridge},
  author  = {Yun, Sungho and Cho, Seungryong},
  year    = {2026},
  note    = {arXiv preprint}
}
```
