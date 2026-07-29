#!/usr/bin/env bash
# Phase 3, 03:00. Two probes then the 3-patient confirmation, so the morning has BOTH a defensible
# recommended config and a map of where the frontier is.
#
# STANDINGS on val 0 (x_t vs GT = the MAIN deliverable | OUT vs sFDK = EVIDENCE | rot | RPE zc):
#   thies_v0  baseline   34.25 / 0.9488 | 33.22 / 0.7764 | 0.456 | 0.478   49.0 s/step
#   l2si3e3   lr 3e-3    35.85 / 0.9695 | 34.27 / 0.7862 | 0.259 | 0.308   49.0
#   c2per400  coarse     36.75 / 0.9796 | 34.64 / 0.7885 | 0.181 | 0.213   54.0
#   per200    fine       37.59 / 0.9802 | 35.08 / 0.7930 | 0.176 | 0.190   69.8
#   oracle theta         40.45 / 0.9889 | 36.27 / 0.7935 | 0     | 0
#
# THE TWO THINGS WE DO NOT KNOW.
#  (a) IS PER=200 THE KNEE? Every step from 50 -> 200 paid off; nothing says 400 will not. This is
#      the single number that decides the recommendation, so it goes first.
#  (b) WHY DOES COARSE LOSE x_t WHILE MATCHING theta? c2per400 reached rot 0.181 against per200's
#      0.176 -- a tie -- yet its x_t is 0.84 dB lower. A theta fitted on the 2 mm grid appears to
#      carry a small bias with respect to the FINE forward model that the CG data step then bakes
#      into x_t; rot_rmse is too coarse a summary to see it (RPE zc does: 0.213 vs 0.190).
#      `--est_coarse_until` tests the fix directly: take the coarse grid's cheap early descent
#      (it beat per200 at every step to t=0.5 -- 0.36/0.29/0.34/0.30 vs 0.61/0.48/0.44/0.35) and
#      hand the endpoint to the fine grid. The estimator is a net over the VIEW INDEX, so the
#      switch carries its weights and Adam moments across intact.
#
# Then per200 on val 1 and val 2 for a 3-patient mean against the baseline's. If a probe wins by
# more than noise, re-point the tail at it before it runs.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_night_phase3.sh > logs/night_phase3.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --n_steps 50 --views_per_iter 24 \
      --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3 --loss l2si --lr 3e-3"

while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 30; done
echo "=== phase 3 starts  $(date) ==="

go () {  # go <tag> <run> <seed> <args...>
    local tag=$1 p=$2 s=$3; shift 3
    local out=data/runs/akima55/$tag
    if [ -f "$out/result.pt" ]; then echo "SKIP $tag"; return; fi
    echo "=== $tag  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --run "$p" --seed "$s" --out "$out" "$@"
}

go per400_v0 0 3  --per 400                                     # (a) is 200 the knee?
go c2f_v0    0 3  --per 400 --est_coarse 2 --est_coarse_until 0.5   # (b) coarse->fine
go per200_v1 1 7  --per 200
go per200_v2 2 11 --per 200
echo "=== PHASE 3 DONE  $(date) ==="
