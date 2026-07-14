# Flow matching for rigid motion correction in 3D head-and-neck CBCT

The 3D extension of `Flow_matching_motion`. Blind rigid patient-motion correction: recover the
volume *and* the per-view 6-DoF motion from a single motion-corrupted cone-beam scan, with a
flow-matching image prior trained on the **geometry bridge**.

## Where this comes from

Three of the lab's projects already hold the pieces, and they turn out to share one algebra:
motion is a right-multiplied object transform on the projection matrix.

| project | what it contributes |
|---|---|
| **Flowmatching-4DCT** | the 3D cone-beam substrate: differentiable ray-march projector (+ Triton kernel), FDK with half-fan / Wang / Ohnesorge, the barrel `measured_region_mask`, 3D UNet, patch-blended prior, 3D TV. Its "4DCT" name is about the *data*; the operator is cone-beam CBCT throughout. |
| **Flow_matching_motion** (2D) | the method: geometry-bridge training, the PnP-TV predictor-corrector posterior loop, band-limited motion estimators, gauge-aware evaluation. |
| **AI_GEOCAL / DTNGC** | how a rigid motion enters a cone-beam `P`: `E = E0 @ T_obj`, `P = K @ E`, pivot at the volume centre; and the projection-domain LNCC objective. |

`fm3d/{geometry_3d,projector_3d,triton_raymarch,tv,unet_3d,unet,prior_patch}.py` are taken from
Flowmatching-4DCT. `fm3d/filters.py` is its ramp filter + FDK scale constant, split out so this
repo carries no 2D fan-beam geometry. Everything else is new.

## The motion model

`theta: (V, 6) = [tx, ty, tz mm | wx, wy, wz rad]`, rotation as an **axis-angle vector**, and the
per-view transform is right-multiplied into the nominal geometry:

```python
P_moved = P_nom @ T(theta)          # (V,3,4) @ (V,4,4)      fm3d/rigid_motion.py
y       = A(x_true; P_moved)        # simulate
x_corr  = FDK(y;     P_moved)       # correct
```

The volume is never warped: a motion update is a `(V,4,4)` matmul, and `d(sinogram)/d(theta)`
flows analytically through the projector's recovery of the rays from `P`.

**Axis-angle, not Euler** — and that is forced, not stylistic. The geometry bridge trains on
partially-corrected geometries `x_t = FDK(y, P_nom @ T(t*theta))`, so `t*theta` must trace the
shortest path from "uncorrected" to "corrected". `exp(t*skew(w))` is the SO(3) geodesic; scaling
Euler angles is not. (AI_Geocal uses `Rz@Ry@Rx` in degrees, which is fine there — it never
interpolates.) Getting this wrong would not crash; it would quietly curve the bridge so the
training velocity no longer points where the inference ODE travels.

## Status

**Milestone 1 (geometry + projector + gates) — done, all gates pass.**

```
python scripts/gate_geometry.py       # ~4 s on one A6000, no dataset, no checkpoint
```

| gate | what it pins down |
|---|---|
| G1 | so(3): `exp`/`log` roundtrip, orthonormality, a finite gradient at `w=0` (the estimator starts there), and `exp((a+b)w) = exp(aw)exp(bw)` — the bridge's licence to interpolate `s*theta` |
| G2 | `theta = 0` leaves `P_nom` bit-for-bit unchanged |
| G3 | `P @ T` projects the object *moved by* `T`, checked against exact array ops (integer-voxel roll; 90-deg rotate by transpose+flip). A sign error here does not crash — it mirrors the reconstruction |
| G4a | autograd `d(loss)/d(theta)` vs central differences, on a smoothed phantom |
| G4b | the descent direction stops moving when the ray sampling is refined |
| G5 | static FDK 30.98 dB → motion 20.76 dB → corrected with the true `theta` 30.51 dB |
| G6 | the bridge path is monotone: 20.76 / 22.13 / 23.90 / 26.79 / 30.51 dB at `t = 0 … 1` |

Montages land in `data/gates/`. **Read them.** G5/G6 clear a numeric threshold, but the failure
that matters — a mirrored or sheared reconstruction that still scores well — is only visible by
eye.

### Two things milestone 1 turned up

**The Triton projector had no adjoint w.r.t. `P`** — *fixed, see below.* `_RayMarch.backward`
returned `gvol, None, None, …`: the exact adjoint in the *volume*, dropping the gradient in the
ray constants, which is exactly where `Pmat` enters. 4DCT never noticed, because it put motion in
a DVF that warps the volume and kept `P` fixed. Here `d(loss)/dP` *is* the motion estimator.

It failed silently — autograd reads a `None` as a zero, so a graph that also differentiates the
volume returns a healthy volume gradient beside a silently-zero `d(loss)/d(theta)`, and the
estimator never moves, converged-looking and with nothing in the log.

**Finite differences on a sharp phantom are a bad gradient test.** Trilinear interpolation makes
the projection only C0 in the sample coordinates, so on sharp edges the analytic gradient (exact,
for the *discretized* operator) and a finite difference (which secants across the kinks, i.e.
approximates the *continuous* operator) legitimately disagree — 15% on one component, and it does
not converge as you shrink the step. Smooth the phantom and they agree to 0.2%. The descent
direction itself is fine: refining `n_samples` 384 → 768 moves it by 0.1% (translation) and 0.6%
(rotation). What is unstable is any single *small* component, which is only a warning about how
to test, not about the code.

**Milestone 2 (everything else) — implemented; smoke test passes on real data.**

```
python scripts/smoke_slab.py                          # ~5 min, no checkpoint needed
python scripts/train_fm3d.py --iters 20000 --amp      # ~4 h on one A6000
python scripts/run_posterior3d.py --ckpt logs/fm3d_a/ckpt_last.pth
```

| module | what it is |
|---|---|
| `fm3d/dataset_slab.py` | virtual slab volumes from the AAPM slice archive (see below) |
| `fm3d/motion_estimation.py` | the three estimators behind one contract, and the projection-domain data terms (`l2si`, **`lncc`**, `ncc`, `ramp`, `l1`) |
| `fm3d/motion_net.py` | `MotionNet6DoF` — AI_Geocal's architecture at the 2D project's **band-limited** hash settings |
| `fm3d/reg_metric.py` | rigid-align-then-score. The headline metric; see the SE(3) gauge below |
| `fm3d/prior_patch.py` | patch-wise prior evaluation, Hann blending, and the global-context channels (milestone 4) |
| `scripts/train_fm3d.py` | geometry-bridge training, 64³ patches, rolling bridge cache |
| `scripts/run_posterior3d.py` | PnP-TV predictor-corrector |

### The data, for now

The real head-and-neck CBCT is not here yet, so the 2D project's AAPM archive (1626 loose
512×512 HU slices) is stacked back into volumes. Adjacent slices in it *are* adjacent anatomy —
but two things had to be measured before that was true:

- **The archive interleaves slices from other levels.** img1367 is near the vertex, img1368 is
  the skull base, img1369 is back at the vertex. Stacking by index puts a skull-base slice inside
  the brain and the coronal reslice comes out streaked. 123 of 1626 are found by the property
  that *deleting* them reconnects their neighbours, and dropped. Without this step no threshold
  works at all: a low one shatters the archive into 5-slice fragments, a high one admits the
  interlopers.
- **What remains is many patients concatenated.** Splitting on consecutive-slice RMS leaves
  **10 runs of ~100 slices** (262 slabs), which reslice cleanly in coronal and sagittal.

The head measures 358 px across, so the pixel size is **~0.5 mm** — not the 1.0 mm the 2D project
assumed, which would make the head 358 mm wide and unable to fit any real CBCT's 26 cm FOV. We
2×-downsample to an isotropic 1 mm grid, **256×256×64**. `dz = 1.0 mm` is an **assumption** (the
archive carries no slice thickness); it changes only how much cone angle a slab subtends.

### Smoke results (oracle: the estimator is handed the true image)

| | aligned PSNR | aligned SSIM |
|---|---|---|
| static FDK (no motion) | 32.59 | 0.713 |
| uncorrected | 22.84 | 0.556 |
| true `theta` | 32.74 | 0.645 |
| **FDK(`theta_hat`), best estimator** | **31.74** | **0.639** |

The estimator reaches **0.56° / 0.91 mm** (gauge-free) and its reconstruction is visually
indistinguishable from the true-`theta` one. That is the ceiling the blind loop is chasing.

**LNCC edges out `l2si`** — 0.91 vs 1.00 mm at 150 iters, and 1.90 vs 3.71 mm at 60, so it also
converges faster. Which is the answer AI_Geocal already had.

`direct` scores terribly in that table and **the comparison is confounded**: stochastic view
subsampling (24 of 360 views per iteration) starves free per-view parameters, which are updated
~1/15 as often as shared ones. It says `direct` is starved, not that it is hopeless.

**Milestone 3 (the Triton adjoint w.r.t. `P`) — done.**

```
python scripts/gate_triton_adjoint.py          # ~1 min
```

`_bwd_kernel` now carries the gradient in the ray constants as well as in the volume. It comes
from the trilinear interpolant's spatial derivative, which the corner weights hand over for free:

```
f      = sum_corners  v * wx * wy * wz            wx = fx (cx=1) or 1-fx (cx=0)
df/dpx = sum_corners  v * (+1 if cx else -1) * wy * wz
```

and with `p_k = A + B*(k+1/2)`:  `d out/d step = sum_k f`,  `d out/d A = step * sum_k df/dp`,
`d out/d B = step * sum_k df/dp * (k+1/2)`. Out-of-bounds corners contribute `v = 0` to both sums,
so `padding_mode='zeros'` is reproduced in the gradient exactly as in the value. `NEED_VOL` /
`NEED_RAY` are compile-time flags — the ray half has to *load* the eight corner values (the volume
half only scatters into them), and the training path, which differentiates the volume with `P`
fixed, must not pay for it.

Gated against `grid_sample`, which autograd differentiates correctly by construction:

| | grid_sample | Triton |
|---|---|---|
| `d(loss)/d(theta)` | reference | **cos = 1.000000**, rel 7e-6 (trans) / 7e-5 (rot) |
| `d(loss)/d(volume)` | reference | cos = 1.000000, rel 2e-5 |
| fwd+bwd through theta | 177 ms, **10.02 GiB** | **58 ms**, **0.62 GiB** |

End-to-end motion estimation: **58 s → 13 s (4.3×)**, and the same-seed accuracy is unchanged
(0.57° / 0.99 mm vs 0.57° / 0.96 mm).

**The 16× memory saving is the bigger prize, and not for the reason you would guess.** Spending it
on more views per iteration buys almost nothing — 360 views × 150 iters costs 15× the time of
24 × 150 and improves rotation from 0.58° to 0.52°, while 24 views × 450 iters is *better*
(0.48°) at a fifth of that cost. Stochastic view subsampling is simply very efficient; iterate
more, don't look at more views. The headroom is what will let the real CBCT run at full size
(512³ volume, 1024×768 panel), which `grid_sample` cannot fit at all.

**Milestone 4 (global context for the patch prior) — implemented, gated.**

```
python scripts/gate_context_unet.py            # ~20 s, CPU, no data, no checkpoint
```

The patch prior had a hole in it. A 64³ patch of a head does not know whether it is orbit or
posterior fossa, nor what the rest of the slab looks like — so it can only learn *local*
structure, and Hann blending hides the seams without fixing that.
[*Local Patches Meet Global Context*](https://arxiv.org/abs/2512.18161) (arXiv:2512.18161) is the
follow-up to the very DiffusionBlend++ that `prior_patch.py` was built on, and it measures the
hole: removing the global-context channel takes FID from **40.8 to 112.1**. Their CT numbers,
LIDC 256³ 8-view: DiffusionBlend 30.43 dB → **33.06 dB**, and **2.75× faster**.

The fix is pure input conditioning — **no architecture change**, only `in_conv` grows:

| ch | content |
|---|---|
| 0 | the patch of `x_t` (what we had) |
| 1 | **the whole `x_t`, resampled onto the patch grid** — the global context |
| 2–4 | the patch voxels' **absolute** (z, y, x) in the volume, normalized to (−1, 1) |

The velocity output stays single-channel and predicts channel 0, so the FM parameterization,
the bridge and the loss are all untouched. `--context none` restores the old prior exactly.
The context channel is rebuilt from the *evolving* `x_t` at every ODE step, which is what makes
inference see the same channel training did. Ported from Flowmatching-4DCT, which had already
implemented and gated it.

`--patch_offsets K` additionally blends `K−1` randomly *shifted* tile grids per step — the FM
analogue of the paper's recurrent noising (K=2 was optimal there). Default 1.

Every failure mode here is bookkeeping — a mis-sliced channel, a coordinate map built from the
*tile* index instead of the *volume* index, a jittered grid that stops covering the border — so
the gate is exact-by-construction rather than statistical. The sharp one is [4]: a probe net that
returns its own coordinate channel must reproduce the analytic coordinate map through the blend,
which can only happen if overlapping tiles agree exactly wherever they overlap.

**Milestone 5 (CQ500 + the field's standard geometry) — implemented and gated; the data itself is not downloaded yet.**

```
python scripts/gate_cq500.py           # ~1 min, no data, no checkpoint
```

The head-motion-compensation literature has standardized on **CQ500** (Qure.ai / CARING: 491
non-contrast head CT scans, DICOM, CC BY-NC-SA 4.0) and on one simulation geometry. CQ500 is
diagnostic *MDCT*, not CBCT: everyone takes its volumes as the clean ground truth and
forward-projects them into a cone beam, which is exactly what we were already doing with AAPM
slabs. `fm3d/dataset_cq500.py` retires three of `dataset_slab.py`'s debts at once — `dz` stops
being a guess (DICOM carries the spacing), the interloper-slice and patient-boundary heuristics
are gone, and we get whole heads instead of 64-slice slabs.

`ConeBeam3DConfig.thies()` — **SID 785 / SDD 1200 / 500×700 panel @ 0.64 mm / 360 views**, shared
by [Thies et al.](https://arxiv.org/abs/2401.09283) (IEEE TMI 2025, and two companions) and
[JRM-ADM](https://arxiv.org/abs/2504.14033). Derived: FOV 288.1 mm, axial coverage 209.3 mm,
M = 1.529. `ConeBeam3DConfig.jrm_adm()` is the same scanner at 0.5 mm pitch and 120 views.

**Which axis of "500 × 700" is lateral is never stated in any of those papers**, and it is not a
detail: `nu = 700` gives a 288 mm FOV that contains a head, `nu = 500` gives 207 mm and truncates
one at every view. Gate [1] pins it. Note also that Thies' own 256³ @ 1 mm evaluation box is
*taller* than the 209 mm the panel sees — 47 mm of it is never measured, which is physical, and
why `measured_region_mask` stays in every metric.

Selection follows Thies Sec. III: thin-slice filter → slice-count outlier cut → **sequential,
patient-level** 150 / 50 / rest split, no RNG. Their two unstated thresholds ("considerably fewer
or more slices"; which of a patient's several thin series to take) are ours and are marked as such
in the module docstring.

The gate does not wait for the download: it **synthesizes a CQ500-shaped DICOM tree** (real DICOM
through SimpleITK, with thick series and slice-count outliers deliberately planted) and runs the
actual indexing, selection, split and projector code over it. It caught a real bug on its first
run — in itself, not the library: its head phantom was larger than the volume it was written into,
so the "head" was pure brain with no skull and no implant, and the HU-clipping check passed on
nothing.

## Next

- Train the prior, then run the blind posterior loop and see how far under the ~31.5 dB oracle
  ceiling it lands. (Deferred: the user is supplying the real data.)
- **`basis` recovers rotation badly** — 10–12° RMSE on both backends at the same seed, against
  0.57° for `net`. An earlier 5.94° was an unseeded lucky run. Probably `n_ctrl=20` is too coarse
  for the `mixed` profile (which contains a step and a jerk), or the lr is wrong for a basis whose
  columns are not unit-norm. Worth one afternoon.
- Swap in the real head-and-neck CBCT when it arrives: `dataset_slab.py` is the only file that
  should need to change, plus a `ConeBeam3DConfig` preset for the real scanner geometry.

## Environment

`conda activate flow_matching` (torch 2.8.0+cu128, triton 3.4.0). 2x RTX A6000.
