# `bench/thies` — Thies et al. (TMI 2025) reimplemented as OUR benchmark

Target paper: **M. Thies et al., "A gradient-based approach to fast and accurate head motion
compensation in cone-beam CT", IEEE TMI 2024/2025** (arXiv 2401.09283).
Full local text: `refs/thies_2401.09283_TMI2025.txt` — every line reference below is into it.

The point of this directory is a **like-for-like** baseline: the same patients, the same
geometry, the same simulated sinogram `y`, the same motion draw, scored by the same metrics as
`scripts/run_posterior3d.py`. Only the *method* differs.

---

## 1. What is upstream code (verbatim) vs. what we wrote

### `vendor/geometry_gradients_CT/` — VERBATIM, Apache-2.0
Source: <https://github.com/mareikethies/geometry_gradients_CT> @ `f6b2b2cc6b801e4c575106b554df70ac5ce90c92`
(cloned 2026-08-03). This is the repository the TMI paper itself points at (TMI L334-336) for
"the implementation of GPU-accelerated backprojection and corresponding gradient computation".

**Not one character is edited.** `bench/thies/vendor_import.py` puts the directory on `sys.path`
so their flat `from helper import ...` imports resolve, instead of us rewriting them to packages.

What we actually use from it: `backprojector_cone.DifferentiableConeBeamBackprojector`
(+ `geometry.Geometry`, `helper`). The fan-beam half, `example.py` and `check_gradients.py` are
kept only so the vendored tree is the upstream tree.

> **But their CUDA kernels are not what the benchmark executes.** Since 2026-08-04 the default
> path is `bench/thies/fast_backprojector.py`, a transcription of the same arithmetic in a
> different execution shape — their backward spent 21 of its 21.5 s in serialized atomics. The
> vendored kernels remain the reference that gate check **G9** compares against on every run, and
> `FM3D_THIES_VENDOR_BP=1` reverts to them. Read **§4.5** before quoting a timing or assuming
> "we ran their code".

### `vendor/moco_diff_likelihood/akima_spline.py` — VERBATIM, from their 2D sibling
Source: <https://github.com/mareikethies/moco_diff_likelihood> (already copied to
`refs/thies_moco_diff_likelihood/`, see its own PROVENANCE.md). `interpolate_akima_spline` is a
**torch, differentiable** Akima interpolator. Verified here: max |torch − scipy| = **1.8e-15**
over a 10-node / 360-view draw, and `autograd.grad` through it works.

The TMI repo releases no motion model, so this is the closest thing to their own spline code.
Caveat that must travel with it: that repo is the *2D 3-DoF* pipeline.

### Everything else in `bench/thies/` — OURS, written from the paper text
`recon.py`, `motion.py`, `vif.py`, `qmnet.py` and the three `scripts/bench_thies_*.py`.
The TMI paper releases only the backprojector; the motion model, the VIF map, the quality-metric
network and the optimizer loop are all described in prose and reimplemented here. Every
non-obvious choice is annotated in-place with the paper line it comes from, and every place we
*could not* follow the paper is listed in §4 below.

---

## 2. The paper's spec, as implemented

| item | paper | line | implemented in |
|---|---|---|---|
| geometry | SID 785, SDD 1200, 500×700 panel @ 0.64 mm, **360 views / full 2π** | L455-462 | `ConeBeam3DConfig.thies()` (already ours) |
| filtering | "a ramp filter and a cosine filter are applied" | L476 | `recon.ThiesConeRecon` — cosine pre-weight + **plain `ramlak`** ramp |
| backprojection | Eq. 3: `I(p) = Σ_j d_j(g(s_j(p)))` — an **unweighted sum** | L308-312 | vendored kernel (which likewise has no weight) |
| motion enters as | `P*_j = P_j · T_j(x)`, i.e. inverse object motion on the geometry | L207-215 | `motion.ThiesSplineMotion` → `fm3d.rigid_motion.params_to_Pmot` |
| motion model | Akima splines, **10 nodes to simulate, 30 nodes to estimate**, evenly spaced, one node at each end | L288-292 | `motion.ThiesSplineMotion(n_nodes=30)` |
| quality target | `VIF* = 1 − K·VIF(I_dist, I_ref)`, spatially resolved per Shao et al. | L326-334 | `vif.vif_star_map_3d` |
| net training data | the **MOTION-FREE** projections reconstructed with the **PERTURBED** matrices ("The filtered projection data is reconstructed from these perturbed matrices") | L490-493 | `data.QMSampleSource.sample` — **fixed 2026-08-06, see §4.9; everything trained before that date is VOID** |
| quality net | 3D U-Net, ReLU, feature maps **8·l for l = 1..4**, final 1×1 conv, no activation | L334-341 | `qmnet.QualityMetricUNet3D` |
| net training | L1 loss, Adam, **lr 1e-3, batch 16**, 128³ in and out | L341-343 | `scripts/bench_thies_train_qm.py` |
| net input norm | "fixed, **sample-independent** offset and slope" to ≈[0,1] | L493-496 | `recon.MU_LO/MU_HI` affine |
| optimizer | plain GD, `x⁽⁰⁾ = 0`, `x⁽ⁿ⁺¹⁾ = x⁽ⁿ⁾ − s(n)·df/dx` | Eq. 6, L366-372 | `scripts/bench_thies_estimate.py` |
| schedule | `s(n) = s0·tⁿ`, **s0 = 100, t = 0.97, 100 iterations** | L376-380 | same, as CLI defaults |
| estimation grid | **128³ @ 2 mm** | L505-507 | `--est_shape 128 --est_voxel_mm 2.0` |
| final recon grid | **256³ @ 1 mm** | L507-508 | `--out_shape 256 --out_voxel_mm 1.0` |
| data | CQ500, thin-slice + slice-count filter → 320 patients, **sequential** patient-level split 150 / 50 / 120 | L382-394 | `fm3d.dataset_cq500.select_series` + `split_patients` (already ours, written from this same passage) |
| display window | −1200 … +1500 HU | Fig. 6 caption, L606 | `recon.HU_WINDOW` |
| noise / scatter | **none** — "dedicated experiments with controlled levels of noise or scatter" is future work | L891-894 | our `gen.simulate()` is likewise noiseless |

---

## 3. THE AMPLITUDE — deliberately 2× the paper (the user's call, 2026-08-03)

Thies' own amplitudes (peak-to-peak; see `refs/thies_moco_diff_likelihood/PROVENANCE.md` for the
proof that "amplitude" means peak-to-peak in their released sampler):

* **train** the quality net at max 10 mm / 15°, per-DoF unequal (L481-489)
* **evaluate** at 5 mm / 5° (L500-503)

We run this benchmark at **our** amplitudes instead, so the number is comparable to
`scripts/run_posterior3d.py`:

| | Thies | **this bench (= ours)** | ratio |
|---|---|---|---|
| quality-net TRAINING | 10 mm / 15° p2p, `amp_mode="thies_hn"` | **15 mm / 20° p2p, `amp_mode="thies_hn"`** | 1.5× / 1.33× |
| EVALUATION | 5 mm / 5° p2p, `amp_mode="fixed"` | **10 mm / 10° p2p, `amp_mode="fixed"`** | **2× / 2×** |

`amp_mode="thies_hn"` (2026-08-06) is THEIR released per-DoF sampler — a clipped half-normal,
`a_d = A_d·min(|N(0, u_d)|, 1)`, `u_d~U(0,1)` — transcribed from
`refs/thies_moco_diff_likelihood/autofocus_data_set.py`. Its mass near zero is the paper's
"motion patterns that perturb the data only slightly" clause: severity < 0.5 in 12.5% of draws,
against 1.5% under the older `"thies"` U(0,1) reading (which remains what OUR PRIOR was trained
with; its meaning is frozen).

Those are exactly the values on our deployed prior
(`logs/fm3d_cq500_leap/ckpt_iter500000.pth` → `args["train_trans_mm"]=15`,
`train_rot_deg=20`, `trans_mm=10`, `rot_deg=10`, `motion_amp="thies"`, `amp_units="p2p"`), and
they are the CLI defaults of both bench scripts. Pass `--thies_amp` to either script to fall back
to the paper's own numbers if you ever want the published operating point.

**Consequence to state in any writeup:** this baseline is being run ~2× harder than the paper,
so a score below the paper's is expected and is *not* evidence of a broken reimplementation.
Use `--thies_amp` if you need to check the reimplementation against the published figures.

---

### 3b. What the paper does not specify at all

Distinct from §4 (places we read the paper and had to choose between readings), these are places
the paper is simply **silent**, so there is no "Thies protocol" to follow:

| gap | what the paper says | what we do |
|---|---|---|
| quality-net training length | nothing. L341-343 gives loss / Adam / lr 1e-3 / batch 16 and stops. **The "lr 1e-4, Adam, 500 epochs" at L527-528 is their re-implementation of the Huang et al. BASELINE, not their own net** -- an easy line to misattribute. | `--iters 5000` x batch 16 = 80k simulated scans, judged by the val curve |
| use of the 50-patient val split | defines it (L390-394), never mentions it again -- no early stopping, no selection rule | sweep **all 50** every `--val_every`, **one fixed motion draw per patient** (`VAL_SEED0 + i`), keep `qmnet_best.pth` |
| logging | nothing (no paper states this) | tensorboard under `<out>/tb`, same `--no_tb` flag as `scripts/train_fm3d.py` |
| final evaluation cohort | **"the same 30 patients from the test set"** (L500-503), with one motion pattern per patient held constant across every method — NOT all 120 | matched: `scripts/drivers/drive_test30.sh` runs our loop over test patients 0..29 with `--seed 1000+i`, and the bench must be run over the SAME (split, run, seed) triples. Note our selection keeps 343 patients (theirs 320), so our test split is 143, not 120; the first 30 are what both methods score. |

Fixed val draws rather than resampled ones is the load-bearing choice: resampling makes the val
curve jitter by more than the training signal, so a "best" checkpoint chosen off it is noise.
Neither the budget nor the selection rule can favour the baseline unfairly -- both make it
stronger, which is the direction a baseline should err in.

## 4. Where we could NOT follow the paper (read before quoting any number)

1. **No 1/w² distance weight, no angular weight.** Their Eq. 3 *and* their released kernel
   (`backprojector_cone.py:71`) backproject an unweighted sum. Standard FDK has a `1/w²`
   distance weight; our own `fdk_conebeam_3d_batched` has that plus a Voronoi angular weight
   (`geometry_3d.view_angular_weights`, worth +1.14 dB under motion). We keep the vendored
   kernel faithful and expose `ThiesConeRecon(distance_weight=...)` as an explicit A/B knob,
   defaulting to **False = theirs**. Do not silently turn it on and still call it "Thies".
2. **Absolute scale.** Their recon has no physical normalization; the paper only says the volume
   is affinely mapped to ≈[0,1] with fixed constants. We multiply by
   `Δβ · SDD/(2·SOD)`, which is our `_fdk_physical_norm` corrected for the missing `1/w²` at the
   on-axis value `w = SOD`, so the output is ≈ μ [1/mm] and directly comparable to our GT and to
   our FDK. This is OURS, not theirs; `--raw` disables it. `gate_bench_thies.py` check G3 pins
   the central-region agreement against our static FDK.
3. **Rotation parameterization.** The paper says "three rotational and three translational
   components" and never says Euler vs. axis-angle. We use **our** axis-angle convention
   (`fm3d.rigid_motion.rigid_motion_matrices`) so the estimated `theta` is the *same object* as
   our estimator's and RPE is computed by the same code. Their 2D sibling has a single in-plane
   angle, which does not disambiguate this.
4. **Feature-map widths — RESOLVED 2026-08-05, against our first reading.** The sentence is
   "The number of feature maps per level is **8^l** with l = 1, …, 4 levels" (II-B.3, p.1102).
   Every text extraction we had — the arXiv dump in `refs/`, and `pdftotext` on the IEEE PDF —
   flattens the superscript to "8l", and we deployed the literal 8·l = **(8, 16, 24, 32)**.
   Two things overturn that: the published PDF typesets it as an **exponent** (so 8·l is out;
   and 8^l itself would be (8, 64, 512, 4096), which nobody trains), and the cited backbone
   [36] is plant-seg, of which **the author keeps her own fork** at
   `github.com/mareikethies/pytorch-3dunet`, where
   `number_of_features_per_level(8, 4) = [init * 2**k] = (8, 16, 32, 64)`.
   The default is now **(8, 16, 32, 64)** — 0.350 M parameters against the 0.170 M we ran
   before. **Our first reading under-provisioned the baseline by ~2×**, which is the one
   direction a baseline must never err in, so every Thies number produced before 2026-08-05 is
   void and was deleted. `--f_maps 8 16 24 32` restores the old reading for an ablation.
5. **VIF map.** Sheikh & Bovik's VIF [34] localized per Shao et al. [35], which we do not have.
   We implement the pixel-domain VIF (VIF-P) at 4 scales in 3D, resolved so that the map **sums
   to** the scalar VIF — which is precisely the property the paper needs ("scaled by the number
   of voxels K such that an average operation yields values in [0,1]", L328-330). See
   `vif.py`'s docstring for the exact decomposition.
6. **Truncation correction / Parker weights are NOT applied** — correctly: the paper applies
   those only to the real 200° C-arm scans (L727-729), not to the simulated 2π experiments.
7. **The simulation grid is NOT a protocol difference — do not re-raise it.** It was claimed once
   (2026-08-05, in conversation) that Thies forward-projects the *reconstruction* grid and so
   commits an inverse crime while we do not. That is wrong. The paper says the source is "the
   reconstructed **MDCT volumes**" (III), and after filtering those are **0.625 mm slice /
   0.38–0.58 mm in-plane (mean 0.472)** — finer than *every* reconstruction grid in the paper
   (2 mm for estimation, 1 mm for the final volume). Ours is 0.41867 mm isotropic
   (= du·SOD/SDD, 612³). Same regime; ours is marginally finer in-plane and clearly finer
   axially. The only residual difference is that **we** linearly resample the DICOM onto an
   isotropic grid before projecting, i.e. we pay one extra interpolation, which if anything makes
   our simulated `y` slightly smoother — not an advantage. **So if a 5/5 run misses the paper's
   0.61 mm RPE, the simulation grid is not the explanation.** Look at §4.8 first.

### 4.8. Our patient set is NOT the paper's, and it bounds what Table I can be compared to

The paper's filtering ("scans reconstructed with a small slice thickness", then "exclude those
which have considerably fewer or more slices than the average sample") yields **320 scans → 150 /
50 / 120**. Ours yields **343 → 150 / 50 / 143**:

```
[cq500] 343 patients selected | train 150 val 50 test 143
```

The threshold behind "considerably" is never published, so our `count_tol` is ours (see
`fm3d/dataset_cq500.py`'s header, which flags exactly this). Because the split is **sequential by
patient index**, 23 extra admitted patients do not simply land at the end — any of them falling
inside the first 200 shifts the train/val/test boundaries, so our split membership is almost
certainly not theirs.

Consequences, and they differ by which comparison is being made:

* **The head-to-head (ours vs the baseline) is unaffected.** Both methods run on OUR split, the
  same 30 test patients, the same (run, seed) triples, the same sinograms. That is what
  `scripts/cmp_thies_vs_ours.py` asserts and it remains exact.
* **Comparing our numbers to the paper's Table I is weaker than "the same operating point".**
  Even at the paper's own 5 mm / 5° amplitude it is a different 30 patients. Treat a 5/5 run as
  evidence that the reimplementation lands in the right *range* (their Init SSIM 0.83 → 0.94,
  RPE 3.00 → 0.61 mm), not as a reproduction to be matched digit for digit.

### 4.5. The kernels we run are a re-expression of the vendored ones (2026-08-04)

**`vendor/` is still byte-identical to their release and nothing in it was edited.** But the
baseline no longer *executes* those CUDA kernels: `bench/thies/fast_backprojector.py` runs the
same arithmetic in a different execution shape, and `ThiesConeRecon(fast=None)` — the default —
takes it. This is a PERFORMANCE-ONLY deviation and it is recorded here because it is the one
place where "we ran their code" became "we ran their formula".

**Why.** Their `backward_loop` issues twelve `cuda.atomic.add` **per voxel per view** into the
same twelve scalars, with the view loop unrolled into 360 separate kernel launches:

```
128³ threads × 12 components × 360 views = 9.06e9 atomic adds onto 12 addresses
```

Measured on an A6000 (128³ / 360 views / 500×700 panel): forward **0.33 s**, backward
**21.13 s** — and 21.13 s / 9.06e9 = 2.33 ns per atomic, i.e. the entire kernel time *is* the
serialized read-modify-write queue. At Eq. 6's 100 iterations that is **36 min per patient**, so
the 30-patient cohort would have cost ~18 h of GPU purely in atomic contention.

**What changed, exactly.** Three things, all execution shape:

| | vendored | fast |
|---|---|---|
| forward accumulation | `atomic.add` to the thread's own voxel, once per view | register accumulator, one store |
| backward accumulation | 12 atomics per thread per view | shared-memory **tree reduction**, 12 atomics per *block* per view |
| backward launches | 360 (one per view) | 1 |
| divisions per voxel-view | 14 (`/w` ×8, `/w²` ×4, `u/w`, `v/w`) | one reciprocal, folded |

**What did NOT change.** The expressions are transcribed character for character, including the
two conventions a "clean" rewrite would silently repair and thereby stop being their operator:

* `helper.interpolate2d_cuda` floors with `int()`, which **truncates toward zero**, so a sample
  position in (−1, 0) gives index 0 with a *negative* delta — a linear extrapolation off the
  panel edge, not a clamp and not a zero. A `torch.nn.functional.grid_sample` port was written,
  measured **8.3e-2 off** for exactly this reason, and discarded.
* the volume axis map (`point2 ← axis 0`, …) and the `torch.gradient(sinogram, dim=(2,1))`
  detector derivative are theirs, untouched.

Float addition is not associative, so the outputs agree to *reduction-order* noise, not bitwise.

**Measured (2026-08-04, idle A6000, akima 10 mm / 10° p2p motion geometry):**

| | 128³ @ 2 mm (estimation grid) | 256³ @ 1 mm (output grid) |
|---|---|---|
| forward | 405.7 → 314.9 ms (1.3×) | 2717 → 2086 ms (1.3×) |
| forward + backward | 19 403.7 → **750.0 ms (25.9×)** | 111 565 → **3016 ms (37.0×)** |
| 100 Eq. 6 iterations | 32.3 min → **1.25 min** | — |
| value agreement | rel L2 2.9e-7 | 2.9e-7 |
| dI/dP agreement | rel L2 2.4e-5, direction cos 1.0000001 | 6.1e-5, cos 1.0000001 |

**Gate.** `scripts/gate_bench_thies.py` **G9** runs both kernels on a motion geometry every time
and pins value, dI/dP magnitude and — the one that matters — dI/dP *direction*, since that is the
quantity Eq. 6 descends. **G6** (the finite-difference check on dI/dP) is unchanged and now
exercises whichever backend is selected. `FM3D_THIES_VENDOR_BP=1`, or
`ThiesConeRecon(fast=False)`, reverts every caller to the vendored kernels.

### 4.6. The VIF* target is computed in float32, mean-centred (2026-08-04)

`vif.py` used to run entirely in float64 because the textbook `s1sq = conv(x·x) − μ²` subtracts
two numbers of order μ² — and Sheikh's `sigma_nsq = 2` is calibrated on 0–255 imagery, so
μ² ≈ 6.5e4 against a local variance of O(1) in smooth head regions. That is a catastrophic
cancellation costing ~5 significant digits, and float64 on a GA102 runs at 1/64 rate: **412.8 ms
per 128³ pair, i.e. 33% of a `bench_thies_train_qm.py` step at batch 16.**

`_vif_terms` now subtracts the volume mean first, `s1sq = conv((x−c)²) − (μ−c)²`. The identity is
exact in real arithmetic for any constant `c` (the kernel is normalized to sum 1, so the shift
passes through the convolution and the decimation unchanged), both second moments become
O(variance), and float32 becomes safe. **412.8 → 7.3 ms (33×)**; the step target cost drops from
33% to ~1.5%.

Gate **G4** still pins `sum(map) == scalar` (now 1.1e-7, bar 1e-5) and new gate **G10** pins the
float32 map against the float64 one in two regimes — noise (rel 1.2e-7) and the binding
*smooth head-like* case (rel 3.8e-4, where the global centring only partly removes the local
cancellation). `vif_map_3d(..., dtype=torch.float64)` restores the old precision.

### 4.7. The stage-1 run has ONE seam, at iter 2000 (2026-08-04)

> **SUPERSEDED by §4.9 (2026-08-06): the whole `logs/bench_thies_qm` run — both sides of this
> seam — is VOID (trained on the wrong pair). Kept for the record of what the seam was.**

`logs/bench_thies_qm` is **not** a single trajectory from a single code version, and it is **not**
bit-continuous across the seam. Iterations 0–2000 ran the pre-2026-08-04 code; 2000 onward runs
everything in §4.5–4.6. Three things changed under it:

| change | effect on the run |
|---|---|
| VIF* float64 → float32, mean-centred (§4.6) | training **target** shifts rel 3.8e-4 |
| vendored → fast backprojector (§4.5) | network **input** volume shifts rel 2.9e-7 |
| `draw_batch` cross-batch prefetch | RNG consumption order moves → a **different patient/motion sequence** |

**Why this does not warrant restarting from scratch**, and the argument is not bit-identity:

* The target perturbation is ~1.5e-4 absolute on values ≈0.38. The *sample-to-sample* spread of
  the same quantity is ≈0.03 (the log's `target mean` runs 0.32–0.42, because every sample draws
  a fresh motion). **The change is ~80× smaller than the difference between two consecutive
  training samples.** The network is fitting a generator, not a fixed dataset, so there is
  nothing it could have memorized that moved under it.
* The RNG reorder changes *which* i.i.d. draws arrive in what order, not their distribution.
  A stochastic-gradient run has no claim on a particular draw sequence.

**Two caveats that DO follow, and must travel with any number quoted from this run:**

1. **The val curve has a hairline discontinuity at iter 2000.** Validation motions are fixed
   (`VAL_SEED0 + i`) but their *targets* are now computed in float32, so pre- and post-seam val
   L1 differ by ≤1.5e-4 — under 0.13% of the ≈0.116 val L1, and far under the 0.0056 that
   separated the last two checkpoints, but not zero. `qmnet_best.pth` as of the seam
   (val 0.11593 @ iter 2000) was selected under the OLD target.
2. **`--seed 0` no longer reproduces this run end to end.** It reproduces 0–2000 under the old
   code, or 2000→ under the new code from `qmnet_iter002000.pth`, but not one continuous
   trajectory. A clean single-version run needs a fresh launch; nothing about the *method*
   requires one.

### 4.9. 2026-08-06 — the stage-1 pair was the WRONG pair; the second run is VOID too

**The bug.** `QMSampleSource.sample` simulated `y` with the perturbed matrices AND backprojected
with the same perturbed matrices. That is a CONSISTENT pair: the motion cancels between
simulation and reconstruction, so the "motion-affected" input the net trained on was in fact the
CONVERGED point of Eq. 6 — an oracle reconstruction with mild residual artifacts — not a
motion-corrupted volume. The paper's construction (III, p.1103, confirmed against the published
IEEE PDF in `docs/`) perturbs the **backprojection matrices only**: *"The filtered projection
data is reconstructed from these perturbed matrices."* The filtered projection data is the
motion-free circular scan.

**Measured, before the fix** (2 patients, TRAIN_AMP):

| | consistent pair (what the net saw) | paper pair / Eq. 6 x=0 (what it should see) |
|---|---|---|
| PSNR vs static recon | 34.8 / 41.3 dB | 27.0 / 24.2 dB |
| VIF* mean | 0.39 / 0.21 | **0.75 / 0.79** |

Over the 7000 logged iterations of `logs/bench_thies_qm`, the batch target mean stayed in
0.30–0.44 (never above 0.44); at stage 2 the frozen net emitted f ≈ 0.45–0.48 at x=0 —
saturated at its training range's edge — against a true VIF* of ~0.8, and reached RPE
5.33 → 2.68 mm where the paper reports 3.00 → 0.61 mm.

**Voided by this finding:** the ENTIRE `logs/bench_thies_qm` run (both sides of the §4.7 seam),
every `qmnet_*.pth` in it, and every stage-2 number scored with them — including all of
`data/qm_stopcrit/*` (the stop-criterion sweep measured a net trained on the wrong pair).
The `data/bench_thies_cache` static recons remain VALID (the reference side never changed).

**Fixed the same day**, plus the amplitude-shape correction (`amp_mode="thies_hn"`, §3). Gate
**G11** now reconstructs the same filtered static sinogram both ways and asserts the paper pair's
VIF* is high while the consistent pair's is at least 2× lower — the construction cannot silently
regress. Retraining goes to a NEW directory so no seam ambiguity attaches to the corrected run.

### 4.9b. Full re-verification pass (2026-08-06, after §4.9) — what was checked and what came out

A second line-by-line pass over the vendored kernels, the released 2D repo, and the PUBLISHED
PDF (not the txt dump), at the user's request. Confirmed one-to-one, at code level:

* `fast_backprojector` ↔ `backprojector_cone.py`/`helper.py`: all twelve backward expressions,
  the `int()`-truncating 4-tap interpolation, `torch.gradient(sino, dim=(2,1))`, the axis map
  (`point2`←axis0), and Eq. 4/5 of the PDF (p.1101). G9 pins the numerics every gate run.
* motion enters as **P·T** (right multiplication): their released `rigid_2d` does
  `einsum('ijn,jkn->ikn', P, T)`; ours is `apply_rigid_motion = P_nom @ T_obj`. Same side.
* zero-centering: theirs subtracts the mean of the **interpolated per-view curve**
  (`rigid_2d` with the resampled values), not of the nodes; `akima_motion` does exactly that
  (`s - s.mean()` on the per-view spline).
* the training pair: their released `autofocus_data_set.load_data` backprojects
  `filtered_projections.tif` — data from DISK, never re-simulated — with `proj_mat_perturbed`.
  Static data + perturbed matrices, i.e. the §4.9 fix is their released construction verbatim.
* `amp_mode="thies_hn"` ↔ their `(rand(n)−0.5) · min(|N(0, A·rand(1))|, A)`: same distribution,
  per-DoF, p2p convention; their radians/degrees plumbing bug NOT propagated.
* RPE: `fm3d.rigid_motion.reprojection_error` = 300 fixed points (Fibonacci lattice, the paper
  does not specify the arrangement), radii 25/50/100 mm, detector-domain mm, recovered vs target
  geometry — the PDF's own definition (p.1104). Fp64 end to end since today.
* Eq. 6 loop, grids (128³@2mm est / 256³@1mm out), s0/decay/iters, x⁽⁰⁾=0, frozen net,
  map-then-average: all as published.

Remaining KNOWN deviations (all documented above, none silent): ramp discretization (|f|
frequency-sampled; their pyronn build unpublished — cancels inside the VIF pair), the [0,1]
window constants (theirs unpublished), VIF-P vs Shao localization (§4.10), the U-Net block
detail (paper says only "3D conv + ReLU"; the cited plant-seg fork's default block is
GroupNorm8-conv-ReLU with encoder mid-channel halving — ours is plain conv+ReLU at full width,
`--norm group` exists for the A/B), axis-angle rotations, and the deliberate 2× amplitude.

### 4.11. In-training RPE probe (2026-08-06) — the convergence criterion, in tensorboard

`bench_thies_train_qm.py --rpe_every N` (default 500) runs the paper's own Eq. 6 (100 GD steps,
s0=100, t=0.97) on `--rpe_patients` (default 3) FIXED val scans against the current net and logs
`rpe/mean`, `rpe/zero_centred_mean`, `rpe/p{i}`, `rpe/init_mean` to TB. The probe instances are
(val patient i, seed **2000+i**, 10/10 p2p `fixed`) — bit-identical to
`bench_thies_estimate --split val --run i --seed 2000+i`, so a TB point and a stage-2 run measure
the same problem. Val L1 remains what selects `qmnet_best.pth`; RPE is what decides the budget.
The gradient is taken with `autograd.grad(f, mot.x)` so nothing accumulates into the net's
parameters. Setup keeps 3 filtered sinograms on CPU (504 MB each); one probe ≈ 2–3 min.

**Result metrics were extended the same day** (for the fm3d head-to-head): `result.json` now
carries fp64 RPE (+`rpe_init`), per-DoF MAE in the PDF's Fig. 4 axes (in-plane tx/ty/rz,
out-of-plane tz/rx/ry; mm and degrees), and `rmse_hu_*` / `vif_aligned` per reference block —
the paper's Table I axes (RMSE/SSIM/VIF after rigid registration). `aligned_metrics` itself
gained `rmse_raw`/`rmse_aligned` (native mu units). `cmp_thies_vs_ours.py` recomputes BOTH
sides' RPE in fp64 from the saved thetas rather than trusting a json.

### 4.12. The filtered-sinogram RAM cache (2026-08-07) — performance-only, bit-exact

A consequence of §4.9: stage-1's sinogram is now the MOTION-FREE scan, a per-patient constant,
so `QMSampleSource` caches the filtered sinogram in host RAM (`_gfilt`, 504 MB/patient fp32;
150 train + 50 val ≈ 100 GB against 220 GB available) and gates `prefetch` so a cache hit loads
no native volume. Measured: warm sample **0.84 → 0.19 s** (miss unchanged); the cached tensor is
the same deterministic kernel output, verified **bitwise identical** hit vs miss, so the
mid-run redeploy (killed at iter 3800, resumed from `qmnet_last.pth` with its saved RNG state)
is an exact continuation, not a seam. One real bug fixed on the way: the checkpoint's RNG state
is loaded with `map_location=dev` and must come back to CPU before `Generator.set_state`.

### 4.10. Cross-check against the published Table I, and two things it settled (2026-08-06)

The published PDF (`docs/IEEE Xplore Full-Text PDF_.pdf`, TMI 44(2), p.1104 — the txt dump in
`refs/` drops this table) gives the **Init** row at 5 mm / 5°, 30 test patients, metrics on
256³ @ 1 mm after rigid registration: **RMSE 120.75 HU | SSIM 0.83 | VIF 0.48**. Ours, measured
at the same operating point (test patients 0–2, x=0, 128³ @ 2 mm, unregistered):

| | RMSE [HU] | SSIM | VIF |
|---|---|---|---|
| paper Init (5/5) | 120.75 | 0.830 | 0.480 |
| ours x=0 (5/5) | **118.92** | 0.873 | **0.328** |
| ours x=0 (10/10, deployed) | 179.70 | 0.733 | 0.189 |

Two conclusions:

1. **RMSE agrees to 1.5%** (different 30 patients, different grid, no registration — and still
   1.5%): the simulation + geometry + reconstruction chain sits on the paper's operating point.
   Any future gap to the paper is not the recon chain.
2. **Our VIF runs ~0.15 LOW at the same state.** `vif.py` is a VIF-P decomposition (§4.5 of the
   spec list; we do not have Shao et al.), and the probe grid/registration differ, so the exact
   offset is indicative — but absolute VIF values are NOT comparable to the paper's. Monotone
   transformations do not move Eq. 6's minimizer, so this is a *reporting* constraint, not a
   fairness problem. Never quote our VIF against their 0.70.

**Step size is tunable by THEIR OWN protocol.** p.1106: for the clinical scans *"we adjust the
step size for the gradient descent to s0 = 10"* — the authors themselves recalibrate s0 when the
objective's scale changes. Our objective (retrained net, our VIF calibration, 2× amplitude) is
not their objective, so an s0 sweep after stage 1 is protocol-conformant, not a departure. The
CLI default stays 100 (the paper's simulation setting); `bench_thies_estimate.py` already prints
the first-step magnitude to read before trusting any run.

---

## 5. Environment

`numba-cuda` is required by the vendored kernel and was installed into the `flow_matching` conda
env on 2026-08-03:

```
pip install numba-cuda "cuda-bindings==12.9.*" "nvidia-cuda-nvcc-cu12==12.2.*"
# -> numba 0.66.0, numba-cuda 0.30.4, llvmlite 0.48.0, cuda-bindings 12.9.7, nvcc 12.2.140
```

`nvidia-cuda-nvcc-cu12` is **pinned to 12.2** on purpose: the box runs driver 535.154.05
(CUDA 12.2), and libnvvm from a 12.8 toolkit would emit a PTX ISA this driver refuses to JIT.
Torch keeps its own 12.8 CUDA libs; the dry run confirmed nothing torch depends on is touched
(no numpy, cudart, nvrtc or cublas change).

---

## 6. RUNBOOK

### 0. Gate first (seconds, ~0.6 GB, safe on a busy GPU)

```
python scripts/gate_bench_thies.py            # binned panel / 120 views -- wiring only
python scripts/gate_bench_thies.py --full     # deployed 700x500 @ 360 views, ~5 GB
```

Measured 2026-08-03, **ALL GREEN** on the reduced config:

| check | result |
|---|---|
| G1 their torch Akima vs scipy | max abs **5.1e-15**, autograd works |
| G2 30-node estimator fits a 10-node truth | RPE **0.073 mm** (their method reaches 0.61) |
| G3 mm-domain P -> pixel-domain P | max **6.1e-5 px**; isocentre lands on the panel centre |
| G4 VIF map sums to the scalar | rel **1.5e-16**; VIF(ref,ref) = 1.0000 |
| G5 vendored backprojection vs our FDK | corr **0.99974**, ls-scale **0.989** (central 50 mm) |
| G6 analytic dI/dP vs central FD | best rel **8.6e-3** (bar is 1e-1, see the docstring) |
| G7 x=0 is the identity / oracle helps | exact 0.0; **18.54 -> 26.23 dB** with theta_true |
| G8 the Eq. 6 loop runs end to end | f finite, gradient reaches the 30x6 parameters |

G6's eps sweep is non-monotonic on purpose: below ~1e-5 the float32 cancellation in the
difference of two ~1e5 sums dominates, so the LARGEST eps in the table is the trustworthy one.

### 1. Train the quality metric (stage 1, long)

```
setsid nohup /home/mirlab/anaconda3/envs/flow_matching/bin/python \
  scripts/bench_thies_train_qm.py --out logs/bench_thies_qm2 --device cuda \
  </dev/null > logs/bench_thies_qm2.log 2>&1 &
```

(`logs/bench_thies_qm` is the VOID first run — §4.9. Do not resume from it or write into it.)

Defaults are the paper's everywhere except the amplitude, which is **ours** (15 mm / 20 deg p2p,
per-DoF unequal). `--amp thies` gives the published 10 mm / 15 deg.

**Measured cost, and the data-loading trap that cost 43% of it** (A6000, batch 16, 2026-08-04):

| configuration | s/iter | GPU duty cycle |
|---|---|---|
| no prefetch (as first written) | **36.4** | **45%** |
| `fine_workers=1`, prefetch depth 1 | 22.9 | 87% |
| `fine_workers=2`, prefetch depth 2 (**deployed**) | **20.7** | **93%** |

`CQ500Generator.volume_fine` keeps a **single-slot** RAM cache. The trainer refreshes one draw
every 12 steps so it hits; this sampler draws a random patient every sample, so it misses ~always
— cache hit 43 ms vs **cache miss 1.231 s**, fully synchronous with the GPU idle. `prefetch_fine`
already existed (the trainer shipped it in commit 3fafd9a for the identical symptom); the bench
simply never called it. Wiring it plus a second loader thread took the run from 36.4 to 20.7
s/iter, i.e. **~17 h off a 5000-iteration run**.

> **2026-08-07: the whole table above is the PRE-CACHE regime.** The §4.12 filtered-sinogram RAM
> cache (possible only because §4.9 made the sinogram a per-patient constant) supersedes it:
> **~4 s/iter warm** (~0.19 s/sample), native volumes load only on each patient's first touch,
> and ~100 GB host RAM is the price. 10000 iters ≈ 11–12 h.

Beware when profiling this: timing `gen.simulate(0, ...)` in a loop measures a permanent cache
HIT and reports ~0.9 s/sample, hiding the entire 1.231 s load. Draw random patients, or watch the
duty cycle. Likewise a first smoke looks ~5x slow — that is numba JIT plus the static-recon cache
being built, not the steady rate.

At 20.7 s/iter, `--iters 5000` = **~29 h**. `--iters` is a CEILING, not a target: `validate()`
sweeps all 50 val patients every 500 steps and keeps `qmnet_best.pth`, so the run can be killed
the moment the val curve flattens with nothing lost.

### 2. Estimate motion (stage 2, ~1.5 min per patient)

> Cost note: this was **~36 min per patient** until 2026-08-04, all of it atomic contention in
> the vendored backward. With the fast kernels (§4.5) the 100 Eq. 6 iterations take ~75 s, so a
> 30-patient cohort is well under an hour instead of ~18 h. An old timing in a log or a notebook
> is from the vendored path.


```
python scripts/bench_thies_estimate.py \
  --qm logs/bench_thies_qm/qmnet_best.pth \
  --run 0 --out data/bench_thies_val0
```

This calls `run_posterior3d.build_world` with the same `--ckpt/--run/--seed/--trans_mm/--rot_deg`
our own loop uses, so the baseline is handed the byte-identical sinogram. It writes
`final.png` (input / Thies output / their static reference / GT), `result.json` (RPE raw and
gauge-quotiented, plus aligned PSNR/SSIM against BOTH the GT volume and the motion-free Thies
reconstruction) and `result.pt`.

**Check the `[step size]` line the first run prints.** Thies' `s0 = 100` is calibrated on their
objective's magnitude; if the first step moves `x` by ~1e-4 the 100 iterations are a no-op that
will still finish and still print a plausible RPE. The script warns, but read it anyway.

### 3. Head-to-head

Same patient, same seed, same amplitude, our method:

```
python scripts/run_posterior3d.py --ckpt logs/fm3d_cq500_leap/ckpt_iter500000.pth \
  --out data/post500k_val0 --run 0
```

Compare `data/bench_thies_val0/result.json` against that run. Quote the **vs_thies_static**
numbers when comparing to the paper's SSIM 0.94 and the **vs_gt** numbers when comparing to
anything else in this repo -- the two references rank reconstructions differently.
