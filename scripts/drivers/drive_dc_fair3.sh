#!/usr/bin/env bash
# THE FAIR 3-WAY: adj vs sart vs cg, all three run with the BEST motion estimator, all schedules
# OFF, equal update count. Decides which data step is actually best before anything else is
# tuned (user's call, 2026-07-24).
#
# WHY THE EARLIER COMPARISONS WERE NOT FAIR. Each operator carries its OWN built-in relaxation
# schedule, and they differ wildly over the 50-step ODE:
#     adj    alpha*(1-t)^2        -> 1% of its initial size by the end (measured: dc/fm 2.8x at
#                                    step 0, 0.8x at 30, 0.0x from 45 -- the data step switches
#                                    ITSELF OFF for the last 20% of the run)
#     sart   beta *= 0.99/step    -> 60%
#     cg     none                 -> 100%
# So the original 5-way A/B compared "adj that fades out" against "cg at full strength", and the
# follow-up compared 1 adj step against 5 CG iterations. Two confounds, both favouring cg.
#
# WHAT IS EQUALIZED HERE
#   * schedules OFF   --alpha_p 0  /  --sart_beta_red 1.0  /  cg has none
#   * update count    5 per ODE step  (--pnp_k 5  /  --cg_iters 5)
#   * ESTIMATOR       fullband encoder + lr 1e-3 for ALL THREE (user, 2026-07-24) -- the oracle
#                     sweep's winner (rot 0.035 vs hashbl+1e-2's 0.173 deg at equal cost). The
#                     verdict should be reached at the configuration that will actually ship,
#                     not at a superseded one. This is why the cg arm is RE-RUN rather than
#                     reusing data/cg_v0 (hashbl+1e-2) -- mixing estimators across arms would
#                     confound the data-step comparison with the encoder change.
#   * everything else val 0, seed 3, l2si, kappa 0, N=50, PER=50, views/iter 24.
#
# WHAT IS *NOT* EQUAL, AND MUST BE READ WITH THE RESULT: cost per update differs.
#     adj   1 fwd + 1 adj per step        -> ~5 pairs/ODE step
#     cg    1 fwd + 1 adj per iteration   -> ~6 pairs/ODE step
#     sart  2 fwd + 2 adj per step        -> ~10 pairs/ODE step  (it recomputes the row-sum and
#           the column weight V every call, though both depend only on P, which is FIXED inside
#           one ODE step -- an implementation inefficiency, not an intrinsic cost)
# So sart gets ~2x the compute. If it still loses, the verdict is safe; if it wins, redo with
# cached weights. Wall-clock per run is in the log for this reason.
#
# HISTORICAL POINTS at the OLD estimator (hashbl+1e-2), same scan -- context, not comparanda:
#     cg_iters=5, no schedule        data/cg_v0            31.66/0.798, x_t 35.96/0.956, rot 0.15
#     adj K=5 WITH the decay         data/isocost_A_adj_k5 31.39/0.793, x_t 33.65/0.936, rot 0.29
#     adj K=1 / sart K=1             data/dcop_A_adj / data/dcop_B_sart
#
#   GPU=0 bash scripts/drivers/drive_dc_fair3.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --run 0 --seed 3 --loss l2si --kappa 0 --est_band fullband"

echo "=== [1] cg   iters=5 (no schedule by nature) + fullband  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op cg --cg_iters 5 --out data/fair_cg

echo "=== [2] adj  K=5, NO decay (--alpha_p 0) + fullband  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op adj --pnp_k 5 --alpha_p 0 --out data/fair_adj

echo "=== [3] sart K=5, NO decay (--sart_beta_red 1.0) + fullband  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op sart --pnp_k 5 --sart_beta_red 1.0 \
    --out data/fair_sart

echo "=== FAIR 3-WAY DONE  $(date) ==="
