#!/usr/bin/env bash
# PER DIET SWEEP (2026-07-28). How far can the estimator budget be cut before the deliverables
# move? Everything else is the deployed c2f configuration, so `data/runs/akima55/c2f_v{0,1,2}`
# (PER 400, same defaults, verified from snaps/meta.pt) IS the top row of this sweep and is not
# re-run.
#
# WHY THIS AXIS FIRST. A fine estimator iteration costs 115 ms and a coarse one 32 ms (measured
# 2026-07-28 after the kernel retune), so at PER 400 the estimator is 46 s of a 78 s fine step
# and 13 s of a 45.5 s coarse step -- by far the largest single item left. The night sweep put
# the knee at PER~200 for a fine-only run (200->400 bought +0.57 dB of x_t and LOST 0.0006 SSIM),
# but that was never measured for c2f, where the coarse phase has already done most of the
# descent by the time the fine grid takes over.
#
# READ THE RESULT WITH THE NON-DETERMINISM IN MIND: the loop is not run-to-run reproducible
# (~15-20% relative on rot), so a single patient cannot resolve <5% in rot or <0.003 SSIM.
# Judge on the 3-patient mean, and on all four cells (scripts/cmp_runs.py).
#
# LIKELY FOLLOW-UP if PER 200 is neutral but 100 is not: split the knob per phase (coarse stays
# 400 -- it is 3.6x cheaper per iteration and buys the early descent -- fine drops). 78% of the
# wall-clock saving lives in the fine phase alone.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_per_diet.sh </dev/null &> data/runs/akima55/drive_per_diet.log &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# the deployed c2f config MINUS --per, which is what this sweep varies
BASE="--ckpt $CK --split val --motion_kind akima --trans_mm 5 --rot_deg 5 \
      --n_steps 50 --est_coarse 2 --est_coarse_until 0.5 --lr 3e-3 --loss l2si \
      --est_band fullband --views_per_iter 24 --theta_avg 1 \
      --dc_op cg --cg_iters 5 --kappa 0.3"

run () {  # run <out> <args...>
    local out=$1; shift
    if [ -f "data/runs/akima55/$out/result.pt" ]; then echo "SKIP $out (done)"; return; fi
    echo "=== $out  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --out "data/runs/akima55/$out" "$@"
}

# PER 200 first: it is the arm most likely to be adopted, so a truncated night still lands it.
for v in 0 1 2; do
    case $v in 0) s=3;; 1) s=7;; 2) s=11;; esac
    run "per200_c2f_v$v" --per 200 --run $v --seed $s
done

for v in 0 1 2; do
    case $v in 0) s=3;; 1) s=7;; 2) s=11;; esac
    run "per100_c2f_v$v" --per 100 --run $v --seed $s
done

echo "=== DONE $(date) ==="
echo "compare:  $PY scripts/cmp_runs.py data/runs/akima55/{c2f,per200_c2f,per100_c2f}_v{0,1,2}"
