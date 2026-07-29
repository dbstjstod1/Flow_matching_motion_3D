#!/usr/bin/env bash
# Push the TV step ONE notch softer than the newly-adopted default: tv_step 0.0075 vs 0.015.
#
# WHY THIS IS WORTH A RUN. The 3-patient 2x2 (data/tvv_*) showed 0.03 -> 0.015 buys the carried
# x_t +0.49 dB at zero cost while the OUTPUT FDK(theta_hat) and theta do not move at all -- TV
# never touches theta, so it can only reach the output through the estimator's reference, and
# that channel measured as noise. By eye at that setting x_t is still visibly SMOOTHER than the
# static FDK and the GT (brain parenchyma reads waxy), i.e. TV is still the thing removing
# texture, so the ridge may not be walked out yet.
#
# The 2D sibling measured this same axis (kappa 0.3, iters 15, 10 indices):
#     tv_step   x_t SSIM    FBP-final SSIM
#     0.03      0.883       0.923
#     0.015     0.899       0.935   <- 2D's pick
#     0.0075    0.905       0.922   <- x_t STILL rising, but 2D's OUTPUT fell here
# 2D stopped at 0.015 because its OUTPUT degraded. **That reason does not obviously apply to us**:
# our output is flat in tv_step across three patients. If x_t keeps rising and the output stays
# flat, 0.0075 is a free win; if the output finally drops, we have found the wall and stop.
#
# Everything else = the CURRENT DEPLOYED DEFAULT: dc_op cg, cg_iters 5 (user's call 2026-07-26 --
# 5 stays), est_band fullband, kappa 0.3, tv_iters 5, N=50, PER=50, l2si, theta_avg 1,
# views_per_iter 24, prior fp16, metric_mode defer. Same three patients / seeds as every other
# sweep so the rows line up with data/tvv_*.
#
#   GPU=1 bash scripts/drivers/drive_tv0075.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt $CK --split val --dc_op cg --cg_iters 5 --est_band fullband \
        --kappa 0.3 --tv_iters 5 --tv_step 0.0075 --views_per_iter 24 \
        --loss l2si --n_steps 50 --per 50 --theta_avg 1"

run () {  # run <out> <run> <seed>
    local out=$1 rn=$2 sd=$3
    if [ -f "data/$out/result.pt" ]; then echo "SKIP $out (done)"; return; fi
    echo "=== $out  (val $rn seed $sd  tv_step 0.0075)  $(date) ==="
    $PY scripts/run_posterior3d.py $COMMON --run "$rn" --seed "$sd" --out "data/$out"
}

run tv0075_v0 0 3
run tv0075_v1 1 7
run tv0075_v2 2 11

echo "=== TV 0.0075 3-PATIENT SWEEP DONE  $(date) ==="
