#!/usr/bin/env bash
# ISO-COST FALSIFICATION TEST of the "spectral preconditioning" claim (user's hypothesis,
# 2026-07-24): would adj / sart have caught up if we had simply RUN THEM MORE TIMES per ODE step?
#
# THE CLAIM UNDER TEST. Landweber (gradient) iteration DOES converge to the least-squares
# solution -- high frequencies included -- it just needs ~kappa iterations where CG needs
# ~sqrt(kappa). So "spectral vs diagonal" and "few vs many iterations" are two different
# explanations of the same measurement, and the 5-way A/B could not separate them: it gave adj
# ONE step per ODE step and cg FIVE CG iterations. That is not a controlled comparison.
#
# THE CONTROL. `--pnp_k K` repeats the data step K times per ODE step (with --kappa 0 there is no
# TV in between, so it is K pure data steps, each with a FRESHLY COMPUTED gradient). At K=5 the
# forward+adjoint count matches cg_iters=5. Same theta, same prior, same everything else.
#
#   F  adj  pnp_k 5   iso-cost with cg_iters=5
#   G  sart pnp_k 5   iso-cost with cg_iters=5
#   (compare against data/dcop_A_adj (K=1), data/dcop_B_sart (K=1), data/cg_v0 (cg_iters 5))
#
# NOTE ON alpha. adj's step is alpha*||z||*unit(g) with alpha = 0.02*(1-t)^2. Five of them is the
# same TOTAL displacement as the old single alpha=0.1 step that was measured to ping-pong -- but
# NOT the same thing, because the gradient is recomputed between them, which is exactly what can
# turn oscillation into convergence. That difference is the experiment.
#
# IF F CATCHES UP WITH cg, the "the fix had to be spectral" story is WRONG and it was iteration
# count all along. Worth knowing before it goes in a paper.
#
#   GPU=0 bash scripts/drivers/drive_isocost_dc.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt $CK --split val --run 0 --seed 3 --loss l2si --kappa 0 --pnp_k 5"

if [ -n "${WAIT_PID:-}" ]; then
    echo "waiting for pid $WAIT_PID ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    sleep 20
fi

echo "=== [F] adj  x5 data steps/ODE step  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op adj  --out data/isocost_F_adj_k5

echo "=== [G] sart x5 data steps/ODE step  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op sart --out data/isocost_G_sart_k5

echo "=== ISO-COST DC TEST DONE  $(date) ==="
