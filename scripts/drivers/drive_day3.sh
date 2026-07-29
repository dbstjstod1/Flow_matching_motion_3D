#!/usr/bin/env bash
# Day 3, GPU 0, sequential. Two independent questions, both on CQ500 val 0 / seed 3 so every run
# here is directly comparable to data/cg_v0 (cg_iters 5, kappa 0, hashbl+1e-2 estimator).
#
# PART A -- THE FALSIFICATION TEST of "the fix had to be SPECTRAL" (user's hypothesis).
#   Landweber/gradient iteration DOES reach the least-squares solution, high frequencies
#   included; it just needs ~kappa iterations where CG needs ~sqrt(kappa). The 5-way A/B never
#   controlled for that -- it gave adj ONE data step per ODE step and cg FIVE CG iterations. With
#   --pnp_k 5 --kappa 0 the loop repeats the data step 5x per ODE step, each with a FRESHLY
#   COMPUTED gradient, matching cg_iters=5's forward+adjoint count. If adj catches up, our story
#   is iteration count, not spectrum. (Non-trivial either way: adj's step is a FIXED-SIZE
#   normalized alpha*||z||*unit(g), which cannot converge as-is -- 2D measured 84% of the
#   accumulated path cancelling -- and recomputing the gradient between repeats is exactly what
#   might turn that oscillation into convergence.)
#   Runs A/B keep the OLD estimator (hashbl + lr 1e-2) on purpose: changing the estimator at the
#   same time would confound the data-step comparison with the Part-B change.
#
# PART B -- deploy check of the new estimator default (fullband encoder + lr 1e-3), which the
#   oracle sweep put 5x ahead of hashbl+1e-2 (rot 0.035 vs 0.173 deg) at equal cost. The oracle
#   hands the estimator the GT; the loop hands it 23-36 dB images, where wide bandwidth has more
#   room to fit artefacts -- so this must be confirmed in-loop before it is trusted. Everything
#   else matches cg_v0, so the ONLY difference is the estimator.
#
#   GPU=0 bash scripts/drivers/drive_day3.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --run 0 --seed 3 --loss l2si --kappa 0"

echo "=== [A] adj x5 data steps/ODE step (iso-cost vs cg_iters=5)  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op adj --pnp_k 5 \
    --est_band hashbl --out data/isocost_A_adj_k5

echo "=== [B] sart x5 data steps/ODE step  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op sart --pnp_k 5 \
    --est_band hashbl --out data/isocost_B_sart_k5

echo "=== [C] cg + NEW estimator (fullband + lr 1e-3), val 0  $(date) ==="
$PY scripts/run_posterior3d.py $BASE --dc_op cg --est_band fullband \
    --out data/fullband_v0

echo "=== [D] cg + NEW estimator, val 1 seed 7 (the hard-theta patient)  $(date) ==="
$PY scripts/run_posterior3d.py --ckpt $CK --split val --run 1 --seed 7 --loss l2si --kappa 0 \
    --dc_op cg --est_band fullband --out data/fullband_v1

echo "=== DAY-3 DONE  $(date) ==="
