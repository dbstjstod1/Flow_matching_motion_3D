#!/usr/bin/env bash
# Phase 4, 04:55. The budget knee has been found; what is left is STRUCTURE.
#
# STANDINGS on val 0 (x_t vs GT = MAIN | OUT vs sFDK = EVIDENCE | rot | RPE zc | s/step):
#   thies_v0  baseline  34.25/0.9488 | 33.22/0.7764 | 0.456 | 0.478 | 49.0
#   l2si3e3   lr 3e-3   35.85/0.9695 | 34.27/0.7862 | 0.259 | 0.308 | 49.0
#   c2per400  coarse    36.75/0.9796 | 34.64/0.7885 | 0.181 | 0.213 | 54.0
#   per200              37.59/0.9802 | 35.08/0.7930 | 0.176 | 0.190 | 69.8
#   per400              38.16/0.9796 | 35.15/0.7927 | 0.125 | 0.181 | 99.3   <- KNEE
#   oracle theta        40.45/0.9889 | 36.27/0.7935 | 0     | 0     | 41.7
#
# THE KNEE, AND WHAT IT MEANS. PER 50->200 bought +1.74 dB of x_t; 200->400 bought +0.57 dB and
# LOST 0.0006 SSIM, for the same +42% wall clock. Meanwhile theta kept improving (0.176 -> 0.125).
# So by per200 the loop is no longer theta-limited -- yet the oracle, whose only advantage is
# having the right geometry FROM STEP 0 (it starts from the same cold FDK), is still 2.3 dB above
# per400. That difference is PATH DEPENDENCE: the early steps apply a badly-wrong geometry to the
# data step and bake it into x_t, and no amount of late-step accuracy undoes it.
#
#   pass2iso   THE ISO-COST TEST OF THAT CLAIM: 2 passes x 25 steps = the same 50 ODE steps and
#              the same PER=200, i.e. EXACTLY per200's price. The estimator (net weights and Adam
#              moments) carries across the restart, so pass 2 re-runs the early steps with the
#              theta pass 1 ended on. If path dependence is real this wins for free.
#   per200_v1/v2  the 3-patient confirmation of the safe recommendation.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_night_phase4.sh > logs/night_phase4.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="--ckpt $CK --split val --est_band fullband --views_per_iter 24 \
      --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3 --loss l2si --lr 3e-3"

while pgrep -f "drive_night_phase3.sh" > /dev/null; do sleep 30; done
while pgrep -f "run_posterior3d.py" > /dev/null; do sleep 30; done
echo "=== phase 4 starts  $(date) ==="

go () {  # go <tag> <run> <seed> <args...>
    local tag=$1 p=$2 s=$3; shift 3
    local out=data/runs/akima55/$tag
    if [ -f "$out/result.pt" ]; then echo "SKIP $tag"; return; fi
    echo "=== $tag  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --run "$p" --seed "$s" --out "$out" "$@"
}

go pass2iso_v0 0 3  --n_steps 25 --passes 2 --per 200
go per200_v1   1 7  --n_steps 50 --per 200
go per200_v2   2 11 --n_steps 50 --per 200
echo "=== PHASE 4 DONE  $(date) ==="
