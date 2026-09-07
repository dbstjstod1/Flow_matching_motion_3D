# arXiv manuscript (extended SPIE submission + benchmark front)

**v2 (2026-08-31, user-directed prose revision).** `main.tex` was rewritten end to end for
natural, journal-style prose after a calibration read of Thies TMI'25 and JRM-ADM: no
aphoristic topic sentences or numbered-observation scaffolding, justifications inlined,
figure/table narration ("As can be seen in..."), each number interpreted once, Discussion
carries implications not restatement. Substantive changes beyond prose: (1) **Table 1's
Thies row now uses the unified scoring convention** (measured-FOV mask, from
`data/roi_body_metrics.json`): 29.22±1.11 / 0.727±0.032 — under one convention the
FDK-class "tie" of v1 becomes a small consistent edge for ours (+0.026 SSIM 23/30
p=1.5e-4; +2.6 dB 30/30), and the v1 "statistically indistinguishable (p=0.30)" claim
(mixed-convention artifact) is GONE — do not resurrect it; (2) noiseless-simulation
statement in III.A + scatter/beam-hardening in the limitations; (3) U-Net spec (base 32,
mults 1/2/4, 2 res blocks, 9.9M params, in_ch=5) and κ=0.3 / tv_step 0.015 / 5 TV iters /
5 CG iters in II.D; (4) p16 PSNR analysis moved from the Fig. 5 caption into IV.C text;
(5) naming unified to "geometry bridge" and "final iterate" (figure labels regenerated);
(6) conference note moved to a footnote (SPIE MI 2027). **The pre-revision version is
preserved as `main_v1.tex` / `main_v1.pdf` / `main_v1.docx`.**

**Polish pass (2026-08-31, post-review, both outputs rebuilt)**: fixed "the $x_t$ column"
(Table 1 has no such column), the inverted "degrades with prior quality", and the TV
notation clash (correct step now uses $D_{\mathrm{TV}}$); the unsourced "81-second"
runtime became the paper's own "under two minutes" (Thies text says only that; the exact
number lives in their Fig. 8 bar chart); Ko et al. moved out of the autofocus family
sentence; III.A now states the Thies thin-slice selection behind the 343 patients;
removed the "worth" tic, "stated openly", "before the test cohort was touched",
"Two features... First/Second" scaffolding, "binding constraint", the duplicate
\cite{thies25} in III.B, and abstract's "improves every patient" (now "every case").

**Introduction rewrite (2026-09-03, user-supplied draft, polished and spliced)**: five
paragraphs in the user's order (problem; autofocus family with its metric-dependence
critique; generative-prior loops, JRM-ADM, and the DPS re-noising detour; the geometry
bridge as the correction-aligned alternative; a numbered contribution list (1)/(2) plus
the evaluation sentence). Citations mapped onto the existing bib keys only, no new
references; Fig. 1 caption and the SPIE footnote unchanged. Both outputs rebuilt. The abstract was
rewritten the same day to carry the same argument (autofocus metric-dependence, the
DPS re-noising detour, the bridge as correction-aligned trajectory); its numbers are
unchanged.

Two deliverables, one source:

- `main.tex` → `main.pdf` — arXiv preprint (single-column, TMI-style I/II.A section layout).

      /home/mirlab/anaconda3/envs/texbuild/bin/tectonic main.tex

- `main.docx` — editable Word version, generated FROM `main.tex`:

      /home/mirlab/anaconda3/envs/flow_matching/bin/python make_docx.py

  (`make_docx.py` resolves \cite/\ref to literal numbers, numbers the captions, swaps the
  figures to PNG, runs pandoc 3 from the `texbuild` env, then post-fixes the docx with
  python-docx: 1-in margins, Times, black headings, fixed table grids with CENTERED data
  cells. Its REF/SECNUM maps are maintained by hand — adding/reordering a
  section/table/figure in main.tex requires updating them; a stale \ref fails loudly.
  Proof-render with `soffice --headless --convert-to pdf main.docx --outdir proof`.)

Both engines live outside the repo env: tectonic 0.17 and pandoc 3.10 in the `texbuild`
conda env (installed 2026-08-19; first tectonic run needs network for packages).

## What this is

The extended version of `docs/spie/SPIE_manuscript_submission.pdf` (2026-08-06), same method
and title, restructured to the field's section layout (I Intro / II Methods / III
Experiments / IV Results / V Discussion / VI Conclusion), with the benchmark front completed
2026-08-11 ~ 08-19. Per user direction (08-19): no reimplementation table (folded into two
sentences in III.C), no em dashes, motion accuracy gets its own table (Table 2) arguing
FDK(θ̂) ≈ FDK(θ_true), and Fig. 4 uses a patient (p14) where the physics bridge clearly
beats the linear bridge.

## Where every number comes from (regenerated/verified 2026-08-19)

| item | source |
|---|---|
| Table 1 (image quality, all methods) | `scripts/cmp_arms.py` (3 arms) + `cmp_thies_vs_ours.py` (Thies row) + SPIE staging (uncorrected, motion-free FDK rows) + `vs_oracle_fdk.json` (oracle row, loop operator) |
| Table 2 RPE columns | `cmp_thies_vs_ours.py` (Thies, ours) + fp64 recompute from stored theta (arms) |
| Table 2 ceiling-referenced SSIM | `vs_oracle_fdk_vanilla.json` per arm — ours+Thies existed; linbridge/w3dm generated 08-19 via `cmp_fdk_vs_oracle_ceiling.py --ours data/{linbridge,w3dm}_test30` (logs/ceiling_*.log) |
| III.C reimplementation sentences | `data/bench_thies_test30_amp55/p*/result.json` aggregate vs Thies TMI Table I (verified in `docs/IEEE Xplore Full-Text PDF_.pdf`) |
| Fig. 2 (method: bridge strip + inference flowchart) | `python scripts/fig_method_arxiv.py` (no GPU; panel (a) reads the preserved bridge strip `data/fig_assets/bridgeB_p00.npz`, 2026-08-05; replaced the SPIE figure 2026-09-03 on user request, old files in `docs/arxiv/old/`; panel (a) shows MID-CORONAL slices (user's call: aligned with Fig. 3's coronal row: `v[:, 128][::-1]`, uncropped 256x256, vertex up); typography unified to 8.2/6.5/6.0 pt; the flow axis carries no text labels, the bridge equation sits directly under it, user's call) |
| Fig. 6 (per-DoF motion trajectories, p14) + Table 3 (per-DoF MAE, 30 patients) | `python scripts/fig_motion_arxiv.py --tag p14` (CPU; zero-mean gauge via `zero_centre_gauge`, no GT; prints the Table 3 numbers) |
| Fig. 1 (manifold concept) | `python scripts/fig_manifold_arxiv.py` (pure schematic, no GPU) |
| Figs. 3 and 5 (two-case qualitative: ours / cross-method incl. the pixel-linear column) | `python scripts/fig_paper_arxiv.py` (defaults `--ours p00,p16 --bench p14,p16`; needs GPU; regenerated 2026-09-03 on GPU1 with METHOD-NAME labels; Fig. 5 = uncorrected / Differentiable autofocus / (JRM-ADM, pending) / Proposed, pixel-linear bridge / Proposed / GT; `fig_ablation.*` in figs/arxiv is a leftover of the reverted IV.C split) |

Fig 4 is the SPIE figure (`figs/spie/`); Fig 2 was redrawn 2026-09-03 (see table), staged here as
PDF (LaTeX) and PNG (docx). Fig. 1 is the manifold/workflow concept figure added 2026-08-20
(user request): (a) clean-image generative prior vs (b) the geometry bridge as the loop's
own state family, with black analytic-tangent arrows and a blue/green
predict / estimate+correct arrow walk. Figs. 3 and 5 show TWO patients each (p16 added on
user request, 08-20; p16 is the W3DM arm's best case, 1/30, stated in the caption). Slices are UNCROPPED at
the native 256x256 aspect (user's call, 08-21; the old head-bounding-box crops are gone).
Figure
metric labels: stored cohort metrics where a run stores them (`final`, `final_xt`), and the
loop's own convention (mask=measured region, iters=200) for panels no run stores
(uncorrected, FDK(theta_true), Thies) -- ONE convention per figure.

**Body-ROI re-scoring + gauge audit (08-21, user request)**: `scripts/roi_body_metrics.py`
→ `data/roi_body_metrics.json` (30 patients, GT>-500HU closing+fill+largest-CC ∩ measured;
alignment unchanged). Cohort: fm 32.17±1.75/0.9749, linear 30.70/0.9654, w3dm 24.28/0.8645,
thies 25.24/0.8804 — every conclusion preserved, x_t beats Thies 30/30 on BOTH metrics in
the body ROI; quoted in IV.C. Gauge audit (III.B): alignment converged (iters 200→800 and
meas→body ROI change scores <0.01 dB/1e-4 SSIM) and the image-domain gauge matches the
theta-domain SE(3) fit (1.1mm/0.8° vs 1.5mm/0.7° on p16).

**The p16 PSNR inversion (W3DM 35.62 vs ours 35.40, SSIM 0.961 vs 0.976) is REAL and
explained, not a bug** (user challenged it, 08-21; measured): blurring our x_t LOWERS its
PSNR (35.40 -> 32.25 at sigma 0.8), so it is not MSE's smoothing bias; the 0.22 dB sits at
high-contrast bone interfaces of the skull-base anatomy (38% of our MSE in the 5% of voxels
> 500 HU; bone RMSE 153 vs 142 HU) while soft tissue is identical (41 vs 42 HU) and air is
better for us (19 vs 25). Stated in the Fig. 5 caption. Note the standing
convention split elsewhere: our stored cohort metrics are masked, while the Thies-side json
and `cmp_thies_vs_ours` are unmasked; measured 08-20 on 3 patients, the mask moves BOTH
methods the same direction per patient, so the paired Table-1 tie is conservative, not
inflated.

**Algorithm 1 (2026-09-03, user request)**: II.C was renamed "Predictor-corrector inference loop"
(the old "Blind posterior loop" claimed a posterior the method never samples) and now
carries an `algorithm`/`algpseudocode` block placed with `[H]` so it sits inline in II.C
(user's call) right after the opening paragraph; `make_docx.py` converts it
into a bold "Algorithm 1" line plus a one-column table for Word (pandoc has no
algorithmic support), and also resolves `\eqref` now (it used to leak as `[eq:objective]`
into the docx). The float parameters (topfraction 0.9 etc.) in the preamble keep the taller
Fig. 2 from pushing every later float to the end of the paper. Fig. 2 is now a `[H]`
block DIRECTLY UNDER the II.B heading (user: "prettiest right below the B heading"), width 0.80, short caption, so
that Algorithm 1 fits on the same page; `\interfootnotelinepenalty=10000` stops the SPIE
footnote from splitting across pages 2-3, which had been forcing a page break before II.B.

**Prose-tightening pass (2026-09-03, user-directed: "wrong / misleading / too much technical
detail")**: removed the finite-difference-vs-exact-tangent measurement (2% / 2.8e-6), the
"float precision" remark, the 1.1%-RMS / 1-3 dB bridge-mismatch numbers, the Joseph-kernel
pinning sentence, the gauge-audit minutiae (200->800 iters, <0.01 dB), the RPE point-set
numbers, the FDK weighting parenthetical, the Thies optimizer settings (30 nodes / 100 iters /
2 mm), the "plateau" wording of the renoise sweep, the +0.35 dB / four-fold amplification
claim, and the 38%-of-MSE decomposition; introduced the abbreviations A_theta / FDK(theta)
in II.A. **The native JRM-ADM paragraph no longer says "within 0.5 dB"** (true for p00 only):
it now reads "trails ours by 0.5 to 2.9 dB in FDK(theta_hat), about 6 dB in the final
output" (p00-02: 0.54/2.85/1.04 dB and 33.01/31.42/31.71 vs 36.55/38.29/38.72 dB). Update
again when the 30-patient native cohort is scored. Follow-up (same day, user): the
finite-difference sentence and the whole bridge-mismatch paragraph ("One property of the
bridge should be noted...") are gone from II.B; in their place a paragraph on WHY the bridge
is the correction path (the estimator fits the motion on the prior-improved image, so the
prior's velocity assists the estimator directly) and a substantive description of the
U-Net's conditioning channels (downsampled whole volume, absolute z/y/x maps).

**III.C Thies paragraph rewritten (2026-09-03, user)**: the manuscript now states that the
baseline runs **the authors' publicly released implementation** (user's correction: "we took
their original code from GitHub as is"), with the quality network retrained on our 150
patients; the "reimplemented because unreleased" claim and the reproduction numbers
(SSIM 0.85->0.93, RPE 1.16 vs 0.61) are GONE from III.C and from the limitations. The
paragraph now describes the METHOD instead: frozen VIF-regressing quality network as the
objective, spline nodes optimized by gradient descent through a differentiable
backprojector, no projection-domain data term, output = analytic reconstruction at the
estimated poses.

**III.C JRM-ADM paragraph rewritten the same way (2026-09-03, user)**: describes the
method first (W3DM wavelet-domain diffusion prior; per DDIM step: clean prediction ->
WLS volume update regularized toward it -> B-spline motion update by gradient descent on
the same projection residual -> re-noise; source `refs/jrm-adm/PROVENANCE.md` "The loop"),
then states that the authors' released implementation is used with the prior retrained on
our 150 patients, then the prior-swap arm (unchanged) and a pointer to the native run.

**DECISION 2026-09-03 (user): the JRM-ADM comparison is the NATIVE run.** III.C now
describes JRM-ADM as run with the authors' released code as published (prior retrained on
our 150 patients; described as a DPS-style algorithm without re-noise/denoise wording;
designed for sparse view, applied to full view unchanged "since full-view data only
provide it with more measurements"). The protocol adaptations (224^3 grid at 1 mm because
their 160x192x192 truncates our measured FOV; prior weight rescaled to keep their
data-to-prior balance under 360 views; output placed in our frame; theta converted to our
convention for FDK(theta_hat)) are DELIBERATELY NOT in the manuscript (user: too technical)
and live here and in refs/jrm-adm/PROVENANCE.md. The prior-swap
arm ("W3DM prior in our loop", SDEdit adapter + InDI step) is GONE from III.C. **Results
(Tables 1-2, IV.B last paragraph, IV.C, Discussion) still describe the prior-swap arm and
must be rewritten once the native 30-patient cohort is scored (running, ~09-06).** The
`sdedit22` bib entry is now uncited and should be dropped with that rewrite.

**Table 1 reference rows removed (2026-09-03, user)**: "FDK at true motion" (33.34/0.773)
and "motion-free FDK" (33.11/0.832) are no longer in Table 1. The abstract, IV.A, the
Discussion and the Conclusion still say the final iterate "surpasses even the motion-free
FDK reference" (+3.5 dB / +0.146 SSIM) -- those numbers now have no table to point to and
must be re-anchored or dropped in the results rewrite.

**Table 2 raw-RPE column dropped (2026-09-03, user)**: Table 2 is now Method | RPE (mm,
mean pose offset removed) | SSIM vs FDK(theta_true). III.B defines RPE with the offset
removed only; IV.B's raw-RPE sentence is gone and replaced by the paired tally on the
offset-removed RPE: ours < Thies on 30/30, p = 1.7e-6 (computed 2026-09-03 from
`data/bench_thies_test30/paired_vs_ours.json`, fields o_rpe_g / t_rpe_g, scipy wilcoxon).

**Table 2 MERGED INTO Table 1 (2026-09-03, user)**: Table 1 is now Method | PSNR | SSIM |
RPE (mm, mean pose offset removed); the separate motion table and its "SSIM vs
FDK(theta_true)" column are GONE (user: SSIM already shown above; the oracle-referenced
comparison is not reported any more -- this closes the earlier question about the
FDK(theta_true) sentence in III.B). IV.B now reads the RPE column of Table 1 and points to
the FDK(theta_hat) row as the learning-free witness. Abstract: the 0.980-vs-0.845 oracle
SSIM sentence was replaced by "0.28 mm against 2.26 mm for the autofocus baseline".
`make_docx.py` REF map: tab:motion removed. 12 pages now.

**Results RESTRUCTURED (2026-09-03, user)**: IV.A "Performance of the proposed method"
(Table 1 = OURS ONLY: uncorrected / FDK(theta_hat) / x_t with PSNR, SSIM, RPE; Figs 3-4;
the motion-free-FDK claim now carries its numbers inline, 33.11 dB / 0.832), IV.B
"Comparison with existing methods" (Table 2 = Thies / De Paepe et al. (JRM-ADM) [row
EMPTY until the native cohort is scored] / ours, plus the pixel-linear ablation row under a
midrule; Fig 5; the native 3-of-30 paragraph), IV.C "Effect of the bridge" (linear-bridge
ablation only). The W3DM-in-our-loop arm is gone from the tables and the text; Fig 5 still
has its column (caption says it is preliminary) and MUST be regenerated with the native
JRM-ADM result. Naming unified (checked 2026-09-03): Thies et al. give their method NO
acronym ("differentiable autofocus"/"our proposed method" in the TMI text), so we write
"Thies et al. [2]" and describe it as differentiable autofocus with a learned quality
metric; "JRM-ADM" IS the authors' name (joint reconstruction and motion estimation
adaptive diffusion model, arXiv 2504.14033), written "De Paepe et al. [7] (JRM-ADM)".
make_docx.py maps updated (sec:results-cmp, tab:cmp, new subsection titles).

**Naming + placement (2026-09-03, user)**: method NAMES label every table/figure row and
column (never author names): "Differentiable autofocus [2]" (Thies et al. have no
acronym), "JRM-ADM [7]", "Proposed, FDK(theta_hat)", "Proposed, x_t (final iterate)",
"Proposed, pixel-linear bridge (ablation)". Author names remain only in prose. User's layout style (09-03): EACH Results subsection opens with its head-image figure,
then its table, then plots, THEN the prose. Final structure (after a brief IV.C detour the
user reverted): IV.A = Fig. 3 (ours qualitative) -> Table 1 -> Fig. 4 (cohort plots) ->
text; IV.B "Comparison with other methods" = Fig. 5 (uncorrected / Learned autofocus /
(JRM-ADM pending) / Proposed / GT; the pixel-linear column was REMOVED again 09-03, the
JRM-ADM column takes its place later; the linear arm appears only in Fig. 6) -> Table 2
(ablation row directly BELOW Proposed, FDK(theta_hat), no separator) -> comparison text -> bridge-ablation paragraph.
No IV.C. All `[H]`, `placeins` FloatBarriers at IV.B and V; figure widths enlarged on user request (09-03): Fig. 3 0.92, Fig. 4 0.78, Fig. 5 0.93
("no need to fit everything on one page"); the resulting part-page gaps at the end of IV.A
and after Table 2 are accepted. 13 pages.

**NATIVE JRM-ADM COHORT COMPLETE AND FILLED IN (2026-09-06, 30/30)**: scored by
`scripts/score_jrm_native.py` -> `data/jrm_native_test30/scores.json` (unified convention:
x_est -> mu*0.02/0.0193, 224^3 zero-padded to 256^3, aligned_metrics mask=meas iters=200;
theta via jrm_theta_convert, fp64 RPE on the zero-centred gauge; FDK(theta_JRM) with our
vanilla operator). Cohort: output 29.64+-2.64 dB / 0.912+-0.038, FDK(theta_JRM)
29.45/0.715, RPE 0.84+-0.55 mm (raw 2.19), per-DoF MAE t 2.17/1.97/0.06 mm, r
0.09/0.07/0.68 deg, runtime 274 min/patient. Paired vs ours: x_t +7.02 dB / +0.066 SSIM
(30/30, p=1.9e-9); FDK(theta_hat) +2.35 dB (28/30, p=1.6e-7) / +0.038 SSIM (29/30).
`docs/arxiv/fill_jrm.py` fills these into main.tex from the frozen template
`main_pre_jrm.tex` (abstract, Table 2 row, Table 3 row, both captions, the IV.B native
paragraph, the motion-MAE paragraph's JRM sentence, Discussion, Conclusion) -- re-run it,
never hand-edit those spots. Fig. 5 has the native JRM-ADM column (fig_paper_arxiv.py src
"jrm"), Fig. 6 the JRM-ADM curve (fig_motion_arxiv.py). The in-plane translation MAE of
JRM-ADM (~2 mm > uncorrected) with a good RPE is the along-beam drift that a detector-plane
RPE does not see; stated in IV.B.

**Native JRM-ADM cohort COMPLETE (2026-09-06 19:00, 30/30)** — scored with
`scripts/score_jrm_native.py` (GPU1; unified convention: rigid-aligned, mask = measured
FOV, iters 200; theta via jrm_theta_convert, zero-mean gauge; fp64 RPE) into
`data/jrm_native_test30/scores.json`. Cohort: output 29.64+-2.64 dB / 0.912+-0.038,
FDK(theta_JRM) 29.45 / 0.715, RPE 0.84+-0.55 mm (raw 2.19), runtime 274 min; per-DoF MAE
t 2.17/1.97/0.06 mm, r 0.09/0.07/0.68 deg. Paired vs ours: x_t +7.02 dB / +0.066 SSIM
(30/30, p=1.9e-9); FDK(theta_hat) +2.35 dB (28/30, p=1.6e-7) / +0.038 SSIM (29/30).
Manuscript filled by `fill_jrm.py` from the frozen template `main_pre_jrm.tex`
(abstract, Tables 2-3 rows, captions, IV.B native paragraph, Discussion, Conclusion); Fig. 5
JRM-ADM column and Fig. 6 JRM-ADM curve regenerated (`fig_paper_arxiv.py` src "jrm",
`fig_motion_arxiv.py` ARMS). **fill_jrm.py is now GUARDED (sys.exit) — main.tex has been
hand-edited since (2026-09-07: IV.C "Computational cost" with Table 4; soft-tissue sentence
in the JRM-ADM paragraph); edit main.tex directly from here on.** Table 4 sources: ours
9.3 min / 80% estimator (memory: inference-cost ledger), JRM-ADM 274 min (scores.json),
autofocus "<2 min" = the authors' statement (the 81 s in paired_vs_ours.json is a stored
constant, NOT a measurement — do not quote it); JRM iteration counts from
refs/jrm-adm/PROVENANCE.md ("The loop": 100 DDIM steps x (15/10 volume + 10/5 motion RMSprop)).
Table 4's third column is ONE unit for all rows (user, 09-07): full-view projector-gradient
evaluations, one unit = one pass over 360 views; autofocus ~100 (one backprojection per GD
iteration), JRM-ADM ~1,600 (10x15+90x10 volume + 10x10+90x5 motion), ours ~900 (10,000
Adam iterations x 24/360 = 670 + 250 CG). The count ratio is <2x; the 30x runtime gap is
mostly per-evaluation cost of their projector (chunked views, 224^3 + wavelet prior per
step) — stated in the text; do not attribute the 30x to the iteration count alone.

## Known TODOs before submission

- ~~The native JRM-ADM transplant paragraph cites p00–02 only~~ DONE 2026-09-06 (30/30).
- The abstract / IV.A / Discussion / Conclusion still claim the final iterate surpasses the
  motion-free FDK (33.11 dB / 0.832, numbers now inline in IV.A only) -- user has not decided
  whether to keep that claim.
- ~~Ref [12] (arXiv:2512.18161) author list~~ CONFIRMED 2026-08-31 against the arXiv
  abstract page: Taewon Yang, Jason Hu, Jeffrey A. Fessler, Liyue Shen — the manuscript's
  "T. Yang, J. Hu, J. A. Fessler, and L. Shen" is correct as printed.
- Author emails / funding / acknowledgments not included.
- ~~Fig. 5 JRM-ADM column, Table 2 / Table 3 rows, Fig. 6 curve~~ DONE 2026-09-06.
- **Per-DoF motion-parameter table (planned, user 2026-09-03)**: III.B now announces a
  view-by-view comparison of theta_hat vs theta per DoF (MAE over views and patients,
  translations and rotations separately, global rigid gauge removed). The table itself is
  not yet in the manuscript; the numbers must be computed from the stored theta of each arm
  (NOT from p00 alone, see the earlier Table-2 MAE mistake).

**Proofreading pass (2026-09-07, user request; both outputs rebuilt, 14 pages)**: code URL added at
the end of II.D (`\url{https://github.com/dbstjstod1/Flow_matching_motion_3D}`, repository still
private); Fig. 3 caption now names the fourth column (FDK at the true motion, which IV.A cites);
Fig. 5 caption corrected (the figure is two patients x {axial, coronal}, not a mid-brain/skull-base
pair of rows); "compares ... with the autofocus baseline" -> "with the two comparison methods";
Table 3 "No compensation" -> "Uncorrected"; the 80%-estimator clause and the 4.6 h sentence
removed from IV.A/IV.B (they live in IV.C / Table 4); JRM-ADM per-DoF sentence made consistent
with Table 3 (r_z 0.68 vs 0.23 is not "about as well"); "matches" -> "is comparable to" for the
pixel-linear arm; Discussion states the 2x amplitude behind the autofocus RPE gap; the uncited
`sdedit22` bibitem dropped ({19} -> {17}). Repository restructured the same day: `main` is now the
compact public tree (separate worktree `~/Desktop/Flow_matching_motion_3D_public`), `dev` keeps the
full history including this directory.

**Native-JRM audit + Codex review pass (2026-09-07, user request)**: every JRM-ADM statement
(III.C, Tables 2-4, IV.B, IV.C, Discussion, abstract) was checked to describe the NATIVE run
(authors' code as published). Two residues of the retired prior-swap arm were fixed: (i) the
IV.B conclusion "the difference lies mainly in the prior" (valid only when the loop was shared)
is now "the pipelines differ in prior, estimator and solver alike, so this comparison does not
attribute the gap to a single component; the bridge ablation isolates the path"; (ii) the
body-ROI sentence, whose 08-21 audit covered the prior-swap W3DM arm, was re-measured with the
native JRM-ADM output added to `scripts/roi_body_metrics.py` (GPU1, 30 patients; the 08-21 json
is preserved as `data/roi_body_metrics_0821.json`): body-ROI fm 32.17/0.975, linear 30.70/0.965,
jrm 25.06/0.910, thies 25.24/0.880, w3dm 24.28/0.865; fm vs jrm +7.10 dB (29/30, p=3.7e-9) /
+0.066 SSIM (30/30); fm vs linear +1.47 dB (23/30, p=7.9e-5). The two baselines' PSNR order
flips inside the body ROI (thies 25.24 > jrm 25.06, a near-tie), so the sentence now claims only
"the ordering and significance of all comparisons involving the final iterate" and was moved to
the end of IV.B. Also: III.C "we apply it to our data without modification" was inaccurate (224^3
grid, prior weight rescaled) and now states both adaptations in one clause.
Codex (OpenAI) editorial review of the 16:30 version, found misplaced in the 4DCT sibling's
docs, now lives in `codex_review/` (plan `arxiv_editorial_plan_20260907.md`, full rewrite
`arxiv_by_codex/main_by_codex.tex` + PDF/DOCX). Adopted from it: "bilinearly" -> "linear in
the volume but nonlinear in the poses" (P0 factual); the DPS critique no longer calls each
clean prediction a "stochastic sample" (JRM-ADM samples with DDIM eta = 0,
`refs/jrm-adm/config/adm_jrm.yaml:41`; now "a generative extrapolation from a partially noised
state"); Algorithm 1 writes the data step as CG_5 warm-started at x_pred; Table 3 bold = column
minima (pixel-linear wins t_x, t_y); Table 4 header "Network evaluations" with a caption
definition and "(reported)" on the autofocus runtime; "one motion realization per patient" in
the limitations; Discussion "competitive" -> "accurate" for the JRM-ADM motion estimate. NOT
adopted (user's prose, or user decisions on record): Codex's abstract/introduction rewrite, the
removal of the "first" claim, re-adding the Wilcoxon sentence, the Eq. 1 -> data-fidelity-only
reformulation, and the motion-free-FDK conclusion rewording (still pending the user's call).
