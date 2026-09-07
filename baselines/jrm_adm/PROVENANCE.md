# JRM-ADM — official code (verbatim clone, 2026-08-11)

Source: <https://github.com/antoinedepaepe/jrm-adm> (branch `main`, shallow clone), the repo for
**De Paepe, Bousse, Phung-Ngoc, Mellak, Visvikis, "Adaptive Diffusion Models for Sparse-View
Motion-Corrected Head Cone-Beam CT", IEEE TRPMS 2025** (arXiv:2504.14033). Adopted as our
second published benchmark (user's call, 2026-08-11): the only other published method in our
exact niche — diffusion prior + blind rigid 6-DoF spline motion + 3D head CBCT + CQ500.
`weights/model_state_dict.pth` (222 MB) is their released pretrained W3DM prior, from
<https://huggingface.co/antoinedepaepe/adm-jrm-w3dm>. NOT in our git (see .gitignore of the clone).

## THE AMPLITUDE CONVENTION (checked first — the Thies 2x incident must not repeat)

`src/utils/creator_utils.py:125-129`:

```python
rand = torch.rand(1, n_control_points)
out  = amplitude * (2 * rand - 1)          # -> U(-A, +A) node values
```

with `motion_amplitude_degree = 5, motion_amplitude_translation = 5` (config). So their
"5 mm / 5 deg" is the **± BOUND** — node values span **10 mm / 10 deg peak-to-peak**, i.e.
**exactly OUR eval amplitude convention (10/10 p2p)** and 2x Thies' (whose "5/5" is p2p, nodes
±2.5). Their eval motion is directly comparable to ours with NO amplitude rescaling; only the
interpolant differs (cubic B-spline, `torch_cubic_spline_grids.CubicBSplineGrid1d`, vs our
Akima) plus per-DoF equal amplitudes (all 3 translations AND all 3 rotations drawn at ±5;
Thies-2D used unequal per-DoF stds).

## Protocol (config/adm_jrm.yaml + generate_data.py), and deltas vs ours

| item | theirs | ours |
|---|---|---|
| volume | 160x192x192 (@ ~1 mm, head.npy demo) | 256^3 @ 1 mm |
| detector | 700(u) x 500(v) @ **0.5 mm** | 500x700 @ **0.64 mm** (Thies) |
| SID/SDD | 785 / 785+415 = 1200 | same |
| views | **120 simulated over 2pi** ("base"), reconstructed from a subsample `n_angles` (paper 60/20; **config ships 40**) | 360, full-view |
| motion | 6-DoF, cubic B-spline, `n_control_points` (**config 30**; paper text says 20), nodes U(-5,+5) mm/deg | Akima 10 nodes, 10/10 p2p eval |
| noise | Poisson, I = 5e5, `b_tilde = log(I/yi)` clamped at 0/inf (`dataloaders.add_noise_sino`) | none in our standard cohort |
| intensity | [-1000, 2000] HU -> [-1,1]; mu_water(80 keV) = 0.0193/mm | our to_net |
| prior | W3DM: 8-channel Haar-wavelet-domain 3D DDPM (in/out_ch=8, base 64, image_size 192 -> operates at 96^3 x 8ch), T=1000, DDIM 100 steps, eta=0 | FM on data bridge |
| splits | 263/15/18 of 296 CQ500 volumes (paper); **split lists NOT in the repo**, repo ships ONE demo volume `data/test_volume/head.npy` | our 150/50/143 |

## The loop (src/sampler/adaptative_diffusion_sampler.py) — the "antithesis" structure

Per DDIM step (100 total, t from ~1000 down to 1):
1. `x0_model = model(xt, t)` — **Tweedie/x0-hat prediction**, the role our FM prior step plays
   (user's observation, 2026-08-11: their consistency step is anchored by x0-hat).
2. `prox_step`: 15 (first 10 steps) / 10 RMSprop iterations on x minimizing
   `WLS(A_theta x, b; yi) + gamma * ||x - x0_model||^2` — **gamma = 0 for the first 10 DDIM
   steps** (pure data fit; hardcoded), then gamma = 1e4.
3. `blind_step`: 10 / 5 RMSprop iterations on the 6x`n_control_points` spline control points,
   WLS sinogram loss, chunked over views (angle_batch_size 20). Optimizer state is REBUILT
   after every solve (`_reset_model_and_optimizer`) — RMSprop moments do not persist across
   DDIM steps.
4. DDIM update from (xt, x0 after prox, x0_no_step before prox) — `sample_one_step`.

Hardcoded schedule inside the sampler (NOT in the config): x_lr 0.5 -> 0.001 at idx 1 ->
0.0005 at idx 50; motion_lr [0.1,0.1] -> [0.01,0.01] at idx 1 -> [0.001,0.001] at idx 50;
the final returned volume is `x0_model` (the last prior prediction), clipped, NOT the last
prox output.

Their own code NOTE (run_jrm_adm.py, end) concedes the SE(3) gauge: the reconstruction "can be
in any position", and motion curves need realignment before comparison — consistent with our
gauge memory; RPE/zc handling must be applied to their outputs like ours.

## Local deltas from the upstream clone

- `config/adm_jrm.yaml`: `n_angles 40 -> 60` (2026-08-11) — the shipped 40 matches NO row of
  the paper's Table I; 60 is the main-table protocol. Everything else untouched.
- `weights/model_state_dict.pth` added (downloaded per README, not in upstream git).
- OURS, not upstream: `score_demo.py` (A1 scoring), `train_w3dm.py` (A2 retraining — upstream
  released inference only; header documents every assumption reconstructed from their
  inference code), `data/train_volumes/` (150 exported volumes from OUR train split, via
  `scripts/export_w3dm_train.py` in the main repo: raw HU, vertex-anchored/bone-COM crop to
  160x192x192, z-FLIPPED to their superior-first order), `weights_retrain/` (the retrained
  prior; `model_state_dict.pth` there is drop-in for the yaml's model_path).
- UPSTREAM BUG to remember: `UNetModel.to()` (w3dm.py:1255) returns None — never chain
  `create_model().to(dev)`; their own scripts call it as a statement, which is why it survives.
- Our volumes carry the CT table/headrest; their volumes are patient-only. Self-consistent for
  the retrained arm; a stated domain gap if the RELEASED weights are run on our exports.

## Repro path (A1)

- env: `jrm_adm` conda env, `carterbox-torch-radon` (conda-forge) + requirements.txt
  (torch-cubic-spline-grids, PyWavelets, torchio, nibabel...).
- `python generate_data.py` (writes data/simulated/gts_and_measurements.pt for the demo head)
  then `python run_jrm_adm.py` (writes data/recon/adm_jrm_{n}_view.pt).
- Table I (paper, n=18): n_a=60 FDK 18.36/0.34 -> JRM-ADM_js 32.02/0.95, JRM-ADM 30.71/0.94;
  n_a=20 -> JRM-ADM 28.88/0.92. The repo has ONE volume, so the A1 gate is
  pipeline-completes + improvement in the Table-I ballpark, not an 18-patient mean match.
  Set `n_angles: 60` (config ships 40, which matches NO row of Table I).
- Their metrics are vs GT warped to the central-view frame (their eval frame) — replicate
  that or apply our rigid-align scoring; note which one any number came from.

## A1 RESULT (2026-08-13): repro gate PASSED

Demo volume (`head.npy`), n_angles=60, their released weights, end-to-end on GPU0 shared with
our trainer (~30-40 min wall; paper: ~17 min on a dedicated RTX 6000 Ada). Scored with OUR
gauge-aware metric (fm3d.reg_metric.aligned_metrics, mu space, rigid-align 300 iters):

    PSNR 31.78 dB / SSIM 0.966 aligned  (raw 23.93 / 0.871; gauge offset 0.85 mm / 2.10 deg)
    paper Table I (n=18 mean): JRM-ADM 30.71 / 0.94, jumpstart 32.02 / 0.95

Squarely in the Table-I ballpark => their pipeline + weights are verified in our hands.
TRAPS FOUND: (1) score in a FIXED frame and it reads ~20 dB / 0.72 -- the recon sits at an
arbitrary SE(3) gauge pose, rigid-align before ANY number leaves this repo (their own paper
NOTE says the same); `score_demo.py`'s fixed-frame convention understates it, kept only as a
record of that trap. (2) launch with `python -u` -- their inner loops print constantly and a
buffered redirect shows a silent 0% tqdm forever (the 08-11 run died unlogged behind exactly
that). (3) torch-radon fork's ConeBeam has no `filter_sinogram` (their FDK row is not
reproducible from this repo alone; De Paepe computed it elsewhere).

## A2c: the operator bridge is MEASURED (2026-08-17)

`scripts/diag_jrm_operator_xcheck.py` (two-stage, one per env; same mu volume through both
static operators at their detector): our LEAP forward vs their torch-radon ConeBeam --
**no u/v flips, no direction reversal, angle origin offset 270 deg, scale 1.00034, all-view
rel error 0.0016.** So our y feeds their solver in OUR view order with
`angles = our_betas + 90 deg` and nothing else. Port harness: `scripts/export_cohort_for_jrm.py`
(stage 1, our env: test30 (split=test, run=i, seed=1000+i) -> data/ours_cohort/p*.pt with
b/yi/angles/theta_true; noiseless yi = 5e5*exp(-b)) + `run_on_ours.py` (stage 2, their env:
their solver stack untouched, recon on THEIR 160x192x192 grid = the retrained prior's native
grid, angle_batch 36 | 360). theta_true rides through both for cmp_thies_vs_ours-style pairing.

## Adaptation decisions still open (A2)

- full-view: our 360-view protocol vs their 120-base/60-sub — port by feeding our simulated y
  (LEAP, native grid) converted to their tensor layout, keeping their solver stack intact.
- their released weights were trained on THEIR 263-volume split — overlap with our test30 is
  unknown; for the paper either retrain W3DM on our train split or bound the leakage risk.
- noise: our cohort is noiseless; run their solver with yi = I*exp(-b) weights consistent with
  whatever y we feed (noiseless limit is fine — weights become exp(-b) up to scale).
