#!/usr/bin/env bash
# Re-prioritised night queue (user, 2026-07-26): **x_t IS THE MAIN DELIVERABLE; FDK(theta_hat) is
# EVIDENCE that the estimated Pmat is accurate.** Ideally both are best.
#
# What changed. `scripts/exp_theta_transfer.py` measured that a PERFECT estimator would move the
# FDK(theta_hat) output by only +0.017 SSIM (0.7764 -> 0.7935 vs the static FDK) though 3.05 dB of
# PSNR -- the FDK operator, not theta, is that volume's ceiling at akima 5/5. That prices theta on
# the EVIDENCE channel. It says nothing about x_t, which is not an FDK and is not bounded by one
# (x_t already scores 0.8048 vs sFDK, ABOVE the oracle FDK's 0.7935).
#
# So the first thing to buy is the missing half of that price:
#   A  --theta_oracle: the loop with the TRUE trajectory every step. The gap to the blind run is
#      exactly what x_t pays for theta being estimated. val 0 (typical) and val 2 (the worst
#      estimator case, rot 1.338 deg -- if theta hurts x_t anywhere, it is here).
# then the estimator legs the earlier queue never reached, which stay worthwhile because the
# evidence channel is PSNR- and RPE-bound even where it is SSIM-saturated:
#   B  est sweep, --ref cold (what step 0 sees) and --ref xt (what the late loop sees).
#   C  the coarse-grid legs (Thies' 128^3 @ 2 mm), accuracy at equal iterations.
#
#   GPU=0 setsid nohup bash scripts/drivers/drive_xt_first.sh > logs/xt_first.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=data/runs/akima55/est_sweep
BASE="--ckpt $CK --split val --est_band fullband --loss l2si --n_steps 50 --per 50 \
      --views_per_iter 24 --theta_avg 1 --dc_op cg --cg_iters 5 --kappa 0.3"

while pgrep -f "exp_est_sweep.py --suite akima55 --ref gt" > /dev/null; do sleep 30; done
echo "=== queue starts  $(date) ==="

# ---- A: what does x_t pay for theta being estimated? ---------------------------------------
for pv in "0 3" "2 11"; do
    set -- $pv
    out=data/runs/akima55/oracle_v$1
    [ -f "$out/result.pt" ] && { echo "SKIP oracle_v$1"; continue; }
    echo "=== oracle_v$1  $(date) ==="
    $PY scripts/run_posterior3d.py $BASE --theta_oracle --run $1 --seed $2 --out "$out"
done

# ---- B: the estimator legs the first queue never reached ------------------------------------
for ref in cold xt; do
    echo "=== est sweep ref=$ref  $(date) ==="
    $PY scripts/exp_est_sweep.py --suite akima55 --ref "$ref" \
        --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
done

# ---- C: coarse grid, accuracy at equal iterations --------------------------------------------
for c in 2 4; do
    for ref in gt xt; do
        echo "=== coarse=$c ref=$ref  $(date) ==="
        $PY scripts/exp_est_sweep.py --suite akima55 --ref "$ref" --coarse "$c" \
            --iters_per_config 2500 --run 0 --seed 3 --out "$OUT"
    done
done
echo "=== XT-FIRST QUEUE DONE  $(date) ==="
