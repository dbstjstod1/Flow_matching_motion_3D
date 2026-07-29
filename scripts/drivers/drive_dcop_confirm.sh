#!/usr/bin/env bash
# CONFIRMATION of the val-0 result on a SECOND patient and a SECOND motion draw, because the whole
# adj-vs-fdk conclusion currently rests on one volume with one seed. Runs the incumbent (adj) and
# the challenger (fdk) back to back on val patient 1, seed 7 -- everything else identical to the
# A/D pair, so the only free variables are the data step and the scan.
#
# Waits for the E (cg) run to release the GPU first: three 256^3 posterior loops do not fit beside
# each other, and the 2026-07-24 OOM was exactly this kind of contention.
#   WAIT_PID=<pid of the cg python> GPU=0 bash scripts/drivers/drive_dcop_confirm.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-1}
SEED=${SEED:-7}

if [ -n "${WAIT_PID:-}" ]; then
    echo "waiting for pid $WAIT_PID to exit ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "pid $WAIT_PID gone, starting  $(date)"
    sleep 20                                    # let the allocator actually release
fi

COMMON="--ckpt $CK --split val --run $RUN --seed $SEED --loss l2si --kappa 0"

echo "=== [confirm/adj] val $RUN seed $SEED  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op adj --out data/confirm_adj_v${RUN}

echo "=== [confirm/fdk] val $RUN seed $SEED  $(date) ==="
$PY scripts/run_posterior3d.py $COMMON --dc_op fdk --out data/confirm_fdk_v${RUN}

echo "=== CONFIRM DONE  $(date) ==="
