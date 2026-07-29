#!/usr/bin/env bash
# Two UNTESTED axes at the deployed estimator (fullband + lr 1e-3), on ALL THREE val patients:
#   tv_step     0.03 (deployed) vs 0.015 (softer corner of the 2D by-eye ridge, never run in 3D)
#   views/iter  24 (deployed) vs 48 (oracle-sweep tie with fullband, but never validated IN-LOOP)
# 2x2 per patient so the two axes are not confounded, plus the deployed base as the reference.
#
# EVERYTHING ELSE = the CURRENT DEPLOYED DEFAULT (post-2026-07-25 accel code): dc_op cg,
# cg_iters 5, est_band fullband, kappa 0.3, tv_iters 5, N=50, PER=50, l2si, theta_avg 1,
# prior_amp fp16 ON, metric_mode defer, metric_every 5. NB the fp16 prior means these are NOT
# bit-comparable to the old fp32 baselines (fair_cg, night_*) -- but all 12 arms here share it,
# so ARM-vs-ARM is clean; that is the comparison this sweep is for.
#
# ORDER: val 0's four arms FIRST (so the 2x2 verdict is readable in ~3 h), then val 1, then val 2.
# Each patient keeps its own motion seed (3/7/11) as in the night runs, so results line up with
# night_A_v1/v2 and fair_cg.
#
# metric_mode defer => the loop only SNAPSHOTS; montages come from render_posterior3d.py afterward.
# The FINAL metrics (x_final & x_t, vs GT AND vs static FDK) are still computed inline and land in
# result.pt, so cmp_* / best-selection work with NO render step. Render for the eyeball later.
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt $CK --split val --dc_op cg --cg_iters 5 --est_band fullband \
        --kappa 0.3 --tv_iters 5 --loss l2si --n_steps 50 --per 50 --theta_avg 1"

run () {  # run <out> <run> <seed> <tv_step> <views>
    local out=$1 rn=$2 sd=$3 ts=$4 vw=$5
    if [ -f "data/$out/result.pt" ]; then echo "SKIP $out (done)"; return; fi
    echo "=== $out  (val $rn seed $sd  tv_step $ts  views $vw)  $(date) ==="
    $PY scripts/run_posterior3d.py $COMMON --run "$rn" --seed "$sd" \
        --tv_step "$ts" --views_per_iter "$vw" --out "data/$out"
}

for pv in "0 3" "1 7" "2 11"; do
    set -- $pv; rn=$1; sd=$2
    run "tvv_v${rn}_base"  "$rn" "$sd" 0.03  24    # deployed reference (TV on)
    run "tvv_v${rn}_tv015" "$rn" "$sd" 0.015 24    # softer TV
    run "tvv_v${rn}_v48"   "$rn" "$sd" 0.03  48    # more views/iter
    run "tvv_v${rn}_both"  "$rn" "$sd" 0.015 48    # both
done

echo "=== TV/VIEWS 2x2 x 3-PATIENT SWEEP DONE  $(date) ==="
