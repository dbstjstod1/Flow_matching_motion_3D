# Results & checkpoint ledger — snapshot at the SF-operator transition (2026-07-28)

Written the evening the project switched to the single SF operator and relaunched training.
Everything below this line was produced with the RAY-MARCH operator unless marked otherwise;
post-transition numbers carry a ~5e-3 operator difference against it until the SF-trained
checkpoint lands.

## Checkpoints

| path | what | status |
|---|---|---|
| `logs/fm3d_cq500/ckpt_iter500000.pth` | **the deployed prior**: 500k iters from scratch (2026-07-19, seed 0), cosine lr 1e-4→1e-6, fp16 AMP, akima 5 mm/5°, patch 32 / base 32 / batch 64 / cache 8 / in_ch 5 (global context), anchor=static. Final val (50-step prior ODE, 3 patients): **aligned 25.42 dB / SSIM 0.78**. | RAY-trained. Keep until the SF ckpt beats it. |
| `logs/fm3d_cq500/ckpt_iter{005k..495k}.pth` (101 files, 16 GB) | every-5k snapshots of the same run (incl. the 180k constant-lr plateau the cosine extension recovered from) | prunable to every-50k if disk pressure ever bites; do NOT prune 180k (the resume point that validated cosine-vs-constant) |
| `logs/fm3d_cq500/train.log`, `tb/`, `val/` | training curves; the late train-loss "rise" is cache-adaptation bias, judge by val only | keep |
| `logs/fm3d_cq500_sf/` | **the SF-operator retrain, relaunched 2026-07-28 evening** — 500k, cosine 1e-4→1e-6, seed 0, val 50 steps × 3 patients every 10k, SF operator, **and the Thies TRAINING amplitude protocol** (`--motion_amp thies --train_trans_mm 10 --train_rot_deg 15`) | running |
| `logs/fm3d_cq500_sf_fixedamp_aborted/` | the first SF relaunch, killed at ~5k when the amplitude protocol changed underneath it. Same recipe but training motion at a fixed 5 mm / 5 deg. | dead; keep only as the loss-scale counterparty (it-50 loss 0.385 vs the new run's ~1.3) |

## Posterior-loop results, in the order the numbers evolved

All at akima 5 mm/5° (the Thies-comparable setting) unless noted. References: vs-GT volume
(ours) and vs-static-FDK (only the latter is comparable to Thies' 0.94 SSIM). Static-FDK
ceiling vs GT ≈ 33.4 dB / 0.825 on val 0.

| stage | config | headline (val 0, seed 3 unless noted) | lives in |
|---|---|---|---|
| baseline at akima55 | thies_v{0,1,2} | vs-sFDK SSIM **0.763** (Thies 0.94) | `data/runs/akima55/` |
| estimator budget night (07-27) | **c2f winner** (coarse 2 mm→fine, PER 400) | x_t 34.25→**37.88 dB** (mixed3mm era numbers in the night table) | `data/runs/akima55/c2f_v*`, `est_sweep/` |
| kernel retune (07-28 am) | BLOCK 128 + relaxed atomics | run 63→**~51 min**, accuracy-identical | memory: kernel-launch-retune |
| unmatched CG (07-28 noon) | `cg_adjoint voxel` (now retired) | CG 28.6→4.6 s; run ~33 min; −0.5 dB vs matched at N=6 | `scratchpad ab_voxel/ab_matched` (temp) |
| SF estimator A/B (07-28 pm) | SF v2 corner kernel | **quality parity with ray** (OUT 30.64 vs 30.61, x_t 33.15 vs 33.07, N=6), coarse step 22.4→**14.3 s** | gate + A/B logs; memory: sf-projector |
| **single operator (07-28 eve)** | SF everywhere (est + matched CG + y) | N=6 smoke: OUT 30.18 / x_t 32.73 — the ~0.45 dB dip IS the ray-trained-prior mismatch the retrain removes | `fm3d/triton_sf.py`; memory: sf-single-operator |
| legacy archive (pre-akima55) | 5-way dc_op A/B (cg winner), fair_cg, night_A/B/C, tv/admm/asd sweeps | dcop table: cg 31.76/0.798, x_t 36.13/0.957 (mixed 3mm/2° — NOT the headline setting) | `data/runs/legacy_mixed3mm/` |

## Deployed inference configuration (what the retrain will be evaluated under)

`run_posterior3d.py` defaults: N=50, c2f estimator (coarse 1/2 until t=0.5), **PER 200**, fullband
NGP lr 3e-3, loss l2, views/iter 24, dc_op cg (cg_iters 5), TV (kappa 0.3,
step 0.015, iters 5), fp16 prior batch 64, metric_mode defer (+`render_posterior3d.py`).
Step times on LEAP with the 500k prior: coarse ~10.7 s, fine ~27.3 s → N=50 ≈ **9.3 min/patient**
(was 15.8 min at PER 400). The SF-era "coarse ~15 s / fine ~54 s → ~29 min" line is retired.

### PER 400 → 200 and the coarse schedule (2026-08-03, first sweep on a FINISHED prior)

3-patient blind A/B, 500k ckpt, val 0/1/2, akima **10/10 p2p**, x_t aligned vs GT (the deliverable):

| config | s/patient | x_t dB / SSIM | rot |
|---|---|---|---|
| fine the whole way, PER 400 | 1321 | 38.52 / 0.9858 | 0.122° |
| c2f, PER 400 (previous default) | 948 | 38.53 / 0.9869 | 0.100° |
| **c2f, PER 200 (deployed)** | **558** | 38.60 / 0.9853 | 0.142° |
| *(--theta_oracle, x_t ceiling)* | *223* | *40.71 / 0.9893* | *0* |

- **The estimator is 76–84% of a step** (949 s blind vs 223 s oracle), not the ~15% the old
  ray-march-era note claimed. PER is the only real lever on inference time.
- **Running fine throughout buys nothing** (+39% time, SSIM −0.0011 vs c2f, 3/3 patients).
- **PER 100 does not converge** — rot still descending at step 49; SSIM −0.0096 = 5× the bar.
  PER 50 collapses (−3.3 dB). `--per_sched ramp` is refuted on both quality and wall clock.
- **Rerun noise is config-dependent**: 0.0018 SSIM at PER 400, **0.0047 at PER 200**. The
  −0.00165 SSIM deficit of PER 200 is therefore *undetectable*, not *absent*; the 3/3 sign
  agreement and the real rot degradation (0.100→0.142°) are the residual risk. Validated at
  10/10 p2p only — θ costs the OUTPUT 0.1 dB but x_t **1.68 dB**, so a harder regime may break
  PER 200 first. Montages indistinguishable by eye.

## FDK normalization (fitted scale DELETED 2026-07-28)

Not a tuning knob: it is the FDK normalization constant `SOD*SDD/2 = 471,000`, which the code
factors out of `fdk_conebeam_3d_batched` and supplies via `scale`. Reconstructing a uniform
water cylinder with that constant alone — no fitting — returns mu = 0.01996 vs a true 0.02
(**-0.18%**). The stored least-squares values (477,165 ray / 475,820 SF) sit +1.31% / +1.02%
above it, which is discretization, and the full operator swap moved it only 0.28%. The fit is
DELETED 2026-07-28: the FDK self-normalizes via `_fdk_physical_norm`; the old ckpt (477,165) is
+1.31% off it, the retrain runs on the constant. See the fbp-scale memory.

## Motion amplitude: TRAIN and EVAL are different protocols (2026-07-28)

Verified against `refs/thies_2401.09283_TMI2025.txt`, both clauses verbatim:

* **Training** (II-B, L478–489): 10 nodes/spline, **maximal** amplitude **10 mm / 15°**, resampled
  per use, *"unequal amplitude across the different motion parameters"*, plus patterns that
  *"perturb the data only slightly"*, all splines individually zero-centred.
* **Evaluation** (IV, L501–506): a fixed **5 mm / 5°** pattern per patient, held constant across
  methods. This is the number our SSIM is comparable to.

### UNITS: EVERY AMPLITUDE IN THIS REPO IS PEAK-TO-PEAK (converted 2026-07-28)

Their released sampler (`refs/thies_moco_diff_likelihood/`) draws nodes as
`(rand(n) − 0.5) · amplitude`, so **their "5 mm / 5°" is nodes in ±2.5**. Confirmed against the
paper's own *"initial median RPE of around 3 mm"*: their draw gives 3.05 mm.

Our arguments used to mean the ± node bound — half the number they mean now. **On 2026-07-28 the
whole repo was converted to peak-to-peak** (`akima_motion` draws `uniform(−amp/2, +amp/2)`) and
every call site doubled. This was a RELABELLING: `gate_motion_amp` check 9 asserts
`make_motion(2A) ≡ pre-switch make_motion(±A)` bit-for-bit across all six motion kinds
(worst |diff| 0.00e+00), and the relaunched training reproduced the previous run's loss to 5
decimals at matching iterations.

| | p2p (the only units now) | initial RPE |
|---|---|---|
| Thies evaluation | 5 / 5 | 3.05 mm |
| Thies training max | 10 / 15 | 7.46 mm |
| **our evaluation** (kept at 2× on purpose) | **10 / 10** | 6.10 mm |
| **our training max** (deployed) | **15 / 20** | — |

**Reading amplitudes back out of a stored run:** use `rigid_motion.amp_from_run_args(d)`, never
`d["trans_mm"]`. Runs saved before the switch lack `amp_units="p2p"` and their stored numbers are
half; the helper doubles them. The trainer and `run_posterior3d` now stamp `amp_units`.

**Our evaluation is therefore at 2× Thies' amplitude, deliberately kept** — every result we have
is on the harder problem, and it is evidence the method holds there. The training max was then
chosen to reproduce his train/eval *relationship* at our point rather than copy his absolute
numbers: `P(a_dof ≥ eval amplitude)` = **0.33 / 0.50** for us against his own **0.29 / 0.42**.
(Copying his absolute 10/15 p2p instead would drop translation coverage to 0.035 — training
*below* the test point.) The run header prints both conventions.

### The sampler

Until 2026-07-28 we ran a fixed 5/5 (node ±) for BOTH train and eval. Now `--motion_amp thies`
(default) makes the training amplitudes per-DoF `a_d = A_d · u_d`, `u_d ~ U(0,1)`;
`run_validation` still draws the fixed 5/5 with its fixed seeds, so the val curve is unbroken
back to the ray era.

**We use a uniform `u_d`, not their clipped half-normal** `min(|N(0, A_d·u)|, A_d)`. Theirs is
not a principled taper — measured, it is 43% of mass below `0.2·max` plus a **9.4% point mass
exactly at the max** (the clipping atom). The low half is the *"perturb only slightly"* clause our
bridge already covers pathwise; the ceiling atom is coverage uniform supplies anyway
(`P(a ≥ 0.9 max) = 10%`). So the half-normal buys nothing here.
`amp_mode="fixed"` is bit-identical to the pre-change sampler (it consumes nothing extra from the
RNG stream) — verified on 5 seeds including val's 1000+i. Gate: `scripts/gate_motion_amp.py` (8/8).

**We adopt two of the three training clauses, not three.** The *"perturb the data only slightly"*
injection is **not implemented at all** (it existed briefly as a `p_slight` knob on 2026-07-28 and
was deleted the same day, once the bridge equivalence below was worked out):
* It is **not** redundant with the per-DoF draw. Severity is a `max` over six independent
  uniforms, so it concentrates near 1 — measured over 4000 draws: median severity 0.86, **0.00%**
  below 20% severity, 0.10% below 30%. Thies' "as well as" is doing real work for a network that
  sees one static corrupted volume per sample.
* It **is** redundant with our geometry bridge. `bridge_P_and_dP` reconstructs at `P(s·θ)` against
  data formed at `P(θ)`, and the axis-angle scaling shares its generator, so the residual is
  exactly `(1−s)·θ`; with t uniform, every draw sweeps residual amplitude uniformly over `[0,|θ|]`.
  And the sampler is **exactly linear in the amplitude** (`uniform(-cA,cA) = c·uniform(-A,A)` on
  the same stream; Akima interpolation and mean-subtraction are both linear), so `(1−s)·Akima(A)`
  is `Akima((1−s)A)` **pathwise** — gate 4 asserts it to 9.5e-07 (float32 rounding), not merely
  distributionally.
* Worse than redundant: it would put mild images at LOW s, while low s in the posterior loop always
  carries LARGE residual motion (t advances with the correction, and our estimator lags — c2f is
  deliberately coarse until t=0.5). Revisit only if the estimator ever leads the ODE.

Two consequences worth writing down:
* **Loss scale jumped ~3.5× again** (it-50: 0.385 → ~1.3) and is now far noisier draw-to-draw
  (0.49 … 1.70 within 200 iters). Both are arithmetic: loss ∝ amplitude², `E[u²] = 1/3`, so the
  translation term scales by `(1/3)(10/5)² = 1.33` and rotation by `(1/3)(15/5)² = 3.0`, and the
  per-draw amplitude is now random instead of constant. **Judge by val only** — the same rule the
  SF switch and the late-loss-rise trap both taught.
* The Akima model's *"maximal amplitude"* bounds the NODE draws, not the trace: the spline
  overshoots between nodes and zero-centring shifts it, so realized peaks average 0.96× the
  nominal bound and reach ~1.5× in the tail. This has always been true of our 5/5 evaluation
  runs too (a "5 mm" pattern can peak near 7 mm).

## Footprint window: DERIVED, not a constant (2026-07-28, late)

`triton_sf`'s `fp`/`fpv` were hardcoded `6`/`4`. The adequate window depends on the geometry —
`u-width = sqrt(dx²+dy²)·M/du`, `v-width = dz·M/dv`, `M = SDD/w`, and the kernel needs
`ceil(width)+1` cells because its scan can start a cell early. At the voxel nearest the source
(M_max 1.99 for SOD 785 / SDD 1200 over a 256³ @ 1 mm box) that is **6 in u and 5 in v** — so the
old default was **truncating the v-footprint**:

| | forward max-rel | dP vs untruncated |
|---|---|---|
| FP/FPV 6/4 (old) | **5.6e-4** (coarse grid 8.2e-4) | cos 0.99955, 3.2e-2 rel-L2 |
| 6/5 (derived) | 1.0e-6 = float noise | **cos 1.000000000** (exact) |

Truncation is one-sided (it only removes mass) and sat on near-source voxels, so it was a
systematic bias in `y`, the bridge target and the CG solution. `_footprint_window()` now derives
and caches it (2.1 µs/call). Cost on an idle GPU: est iter **103.5 → 115.0 ms, +11%**. Traps:
FP must be rounded up to EVEN (it is the innermost unrolled loop; odd measured ~3.5× slower), the
cache key must not touch `P`'s values (a `P.sum()` key cost a host sync per call and could
re-trigger Triton's JIT), and wall-clock checks are meaningless under GPU contention (the same
gate read 276 ms while training held the card at 93%).

Same sweep, no other operator-dependent constant found: the `0.05·d` width floors and `1e-6`
taper floor scale with the voxel, and the fitted FDK scale is already gone (above).
`--est_n_samples` is now documented as INERT under SF.

## The crosshatch streaks, and the FDK ramp window (2026-07-29)

The user found grid texture in the SF-era static-FDK validation panel that the ray era did not
show. **Diagnosed as the VOXEL BASIS, not a bug** — SF is defined for the cube (piecewise-constant)
basis; the retired ray-march/gridsample forward integrated a trilinear tent. Swapping only the
forward proves it: gridsample **bilinear** (tent) is clean, gridsample **nearest** (cube) shows the
same texture, worse. A piecewise-constant object has real energy above the sampling Nyquist (its
voxel faces) and our detector resolves it (0.64 mm pitch = 0.42 mm at isocenter vs 1 mm voxels), so
an unapodized ramp reconstructs the voxel grid. Ruled out first: the FPV truncation (3e-5 effect)
and angular gain error (SF's per-view mass is flat to 5e-4, better than gridsample's 1.2e-2).

**Fix: `window` default `ramlak` → `hann`** in both `fdk_conebeam_3d_batched` and the tangent twin
(derivation in `projector_3d._RAMP_WINDOW_NOTE`; diagnostic `scripts/diag_static_fdk.py`):

| forward / filter | excess sd in homogeneous brain | rmse vs GT | bone-edge sharpness |
|---|---|---|---|
| SF + ramlak (was) | 13.2 HU | 1.334e-3 | 99% of GT |
| SF + shepp | 10.3 HU | 1.309e-3 | 97% |
| **SF + hann (deployed)** | **1.4 HU** | **1.301e-3** | **90%** |
| SF + rtkhann 0.8 | 0.0 HU | 1.318e-3 | 87% |
| tent (ray era) + ramlak | 3.4 HU | 1.292e-3 | 90% |

`hann` reproduces the retired tent+ramlak regime on both axes — same sharpness, lower texture,
equal rmse. The band-limiting moved from implicit (the forward's basis) to explicit (the filter).

**This also retires the "SF train loss is 4× and that is unavoidable" entry above**: the same
high-frequency content drove it, and at `hann` the it-50 loss is ~0.09–0.13, at or below the ray
era's 0.112. Training was stopped at it 2800 and relaunched once the filter was fixed, because the
static FDK is the prior's TRAINING TARGET.

**Refuted, do not retry:** widening the SF footprint to the tent's support. Non-integer widths beat
against the voxel pitch (443 HU at BW 1.4); the exact double width reaches only 9.2 HU while
costing 30% of the edge sharpness. It is the tent's piecewise-*cubic* shape that matters, and doing
it properly reintroduces the width parameterization that made SF v1's d/dP diverge.

**Collateral:** `gate_anchored_bridge`'s cone-attribution sub-check is now pinned to `ramlak` — hann
is a second, spatially uniform contributor to the static-vs-GT deficit and flips the midplane gap
from +2.13 to −0.41 dB. Never measure an attribution through an apodized reconstruction.

## Operator lineage (why the retrain)

ray-march+scatter (exact pair, scatter 3.9 s floor) → +unmatched voxel CG (toolkit standard,
−0.5 dB) → **SF corner-parameterized matched pair with d/dP** (gather speed both directions,
θ-grad autograd-exact, gates 5/5) → single operator, all others deleted (gridsample kept as
gate reference). The 500k prior predates the switch → retrain `logs/fm3d_cq500_sf`.
