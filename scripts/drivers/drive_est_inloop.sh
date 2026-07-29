#!/usr/bin/env bash
# In-loop confirmation of the akima55 estimator sweep's top two (user's call, 2026-07-26).
#
# WHY THIS IS NOW THE HIGHEST-VALUE RUN. `--theta_oracle` measured what theta is worth to the
# MAIN deliverable: x_t goes 34.25 dB / 0.9488 (blind) -> 40.45 / 0.9889 (oracle theta) vs GT,
# i.e. +6.20 dB. On the evidence channel FDK(theta_hat) the same perfect theta is worth only
# +3.05 dB / +0.017 SSIM, because FDK itself is that volume's ceiling. So estimator accuracy is
# the #1 lever for x_t, and the question is whether the oracle sweep's rot win TRANSFERS in-loop
# (the 2026-07-25 views48 result did not -- an oracle win that vanished inside the loop).
#
# The two candidates, each at ITS OWN best lr from the GT leg (2500 iters = the loop's own
# N50 x PER50 budget). The loss axis is a TIE once lr is free; what moved was lr:
#     lncc @ 1e-3  0.087 deg      l2si @ 3e-3  0.093       l2si @ 1e-3  0.159 (DEPLOYED)
# so this also separates "lr was mistuned" from "lncc is better", which the deployed config
# confounds. Baseline for all rows: data/runs/akima55/thies_v{0,1,2} (same patients/seeds).
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_est_inloop.sh > logs/est_inloop.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=data/runs/akima55/est_sweep
BASE="--ckpt $CK --split val --est_band fullband --n_steps 50 --per 50 \
      --views_per_iter 24 --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3"

while pgrep -f "run_posterior3d.py .*--theta_oracle" > /dev/null; do sleep 30; done
echo "=== in-loop confirmation starts  $(date) ==="

run () {  # run <tag> <estimator args...>
    local tag=$1; shift
    local args=("$@") pv p s out
    for pv in "0 3" "1 7" "2 11"; do
        p=${pv% *}; s=${pv#* }
        out=data/runs/akima55/${tag}_v$p
        if [ -f "$out/result.pt" ]; then echo "SKIP ${tag}_v$p"; continue; fi
        echo "=== ${tag}_v$p  $(date) ==="
        $PY scripts/run_posterior3d.py $BASE --run "$p" --seed "$s" --out "$out" "${args[@]}"
    done
}

run l2si3e3 --loss l2si --lr 3e-3
run lncc1e3 --loss lncc --lncc_win 9 --lr 1e-3

echo "=== IN-LOOP DONE  $(date) ==="
# the deferred legs, in the order they became less urgent
for ref in cold xt; do
    echo "=== est sweep ref=$ref  $(date) ==="
    $PY scripts/exp_est_sweep.py --suite akima55 --ref "$ref" \
        --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
done
for c in 2 4; do
    echo "=== coarse=$c ref=gt  $(date) ==="
    $PY scripts/exp_est_sweep.py --suite akima55 --ref gt --coarse "$c" \
        --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
done
echo "=== ALL DONE  $(date) ==="
