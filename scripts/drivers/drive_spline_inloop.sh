#!/usr/bin/env bash
# DOES A SMOOTHER TRAJECTORY MAKE A BETTER RECONSTRUCTION? (user's hypothesis, 2026-07-27)
#
# The bench cannot answer this: it scores theta, never an image. What it DID establish is that the
# question is now well posed, because on the reference the loop actually hands the estimator (a
# carried x_t, `--ref xt`) the two candidates are TIED on accuracy while differing 5x in
# smoothness:
#
#   config                     rot deg   roughness vs truth   high-freq energy
#   mlp fullband + Adam 3e-3    0.380          x10.7               1.6%
#   bspl30      + Adam 1e-2     0.409          x2.0                1.2%
#
# (On a GT reference the MLP is 2.4x more accurate -- 0.068 vs 0.165 -- so this is specifically a
# statement about the noisy references a real loop provides. Thies' FULL method, spline + plain
# GD, is far behind either: 2.794 deg. And the band limit itself has a clear optimum at exactly
# his 30 control points: none 7.93 > 60 ctrl 1.19 > 30 ctrl 0.165 > 10 ctrl 1.90.)
#
# So: same loop, same budget, same everything -- only the motion model changes. If the smoother
# trajectory reconstructs better despite equal theta error, the hypothesis holds and the estimator
# should be swapped. All three patients directly, because the deliverable differences at stake are
# small enough that one patient could not settle it.
#
# Compare against data/runs/akima55/c2f_v{0,1,2} (the same config with the MLP):
#   mean x_t vs GT 36.60 / 0.9799 | OUT vs sFDK 34.12 / 0.7726 | rot 0.308 | RPE zc 0.287
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_spline_inloop.sh > logs/spline_inloop.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the deployed operating point, with ONLY the estimator swapped
BASE="--ckpt $CK --split val --n_steps 50 --views_per_iter 24 --theta_avg 1 \
      --dc_op cg --cg_iters 5 --kappa 0.3 --loss l2si \
      --per 400 --est_coarse 2 --est_coarse_until 0.5"
SPL="--estimator basis --n_ctrl 30 --lr 1e-2"

while pgrep -f "drive_q_estimator.sh" > /dev/null; do sleep 60; done
while pgrep -f "exp_est_sweep.py|exp_fdk_vs_cg" > /dev/null; do sleep 60; done
echo "=== spline-vs-MLP in-loop  $(date) ==="

for pv in "0 3" "1 7" "2 11"; do
    set -- $pv
    out=data/runs/akima55/bspl30_c2f_v$1
    [ -f "$out/result.pt" ] && { echo "SKIP bspl30_c2f_v$1"; continue; }
    echo "=== bspl30_c2f_v$1  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE $SPL --run "$1" --seed "$2" --out "$out"
done
echo "=== SPLINE IN-LOOP DONE  $(date) ==="
