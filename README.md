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

**The Triton projector has no adjoint w.r.t. `P`.** `_RayMarch.backward` returns
`gvol, None, None, …`: it is the exact adjoint in the *volume* and it drops the gradient in the
ray constants, which is exactly where `Pmat` enters. 4DCT never noticed, because it put motion in
a DVF that warps the volume and kept `P` fixed. Here `d(loss)/dP` *is* the motion estimator.

This fails silently — autograd reads a `None` as a zero, so a graph that also differentiates the
volume returns a healthy volume gradient beside a silently-zero `d(loss)/d(theta)`, and the
estimator never moves. `_use_triton` now refuses the Triton path whenever `Pmat.requires_grad`
and falls back to `grid_sample` (loudly, once). Teaching the Triton kernel to carry `dA`/`dBk`
would put the estimator's inner loop back on the fast path; until then motion estimation runs on
`grid_sample`.

**Finite differences on a sharp phantom are a bad gradient test.** Trilinear interpolation makes
the projection only C0 in the sample coordinates, so on sharp edges the analytic gradient (exact,
for the *discretized* operator) and a finite difference (which secants across the kinks, i.e.
approximates the *continuous* operator) legitimately disagree — 15% on one component, and it does
not converge as you shrink the step. Smooth the phantom and they agree to 0.2%. The descent
direction itself is fine: refining `n_samples` 384 → 768 moves it by 0.1% (translation) and 0.6%
(rotation). What is unstable is any single *small* component, which is only a warning about how
to test, not about the code.

## Next

2. **6-DoF motion estimation.** Port the estimator contract (`refine_global` / `current_params` /
   `render_sinogram`) and the band-limited parameterizations (B-spline basis, `hashbl`) that the
   2D project converged on; add projection-domain **LNCC** (AI_Geocal's objective, MONAI-free —
   the 2D repo already has a self-contained `lncc`) beside `l2si`.
3. **Geometry-bridge training** of a 3D prior (patch-based, `prior_patch` blending at inference).
4. **Posterior loop**: PnP-TV predictor-corrector, the configuration the 2D project settled on.
5. **Gauge-aware evaluation.** Blind motion correction has an exact SE(3) gauge — global pose is
   unobservable — so raw PSNR is pose-contaminated. Metrics go through rigid-align-then-SSIM.

The real head-and-neck CBCT data is not here yet; the pipeline is being stood up on a synthetic
head phantom (`fm3d/phantom.py`), which is scaffolding for the gates and **not** a quality
benchmark.

## Environment

`conda activate flow_matching` (torch 2.8.0+cu128, triton 3.4.0). 2x RTX A6000.
