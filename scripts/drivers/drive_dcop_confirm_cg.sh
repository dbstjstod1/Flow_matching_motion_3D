#!/usr/bin/env bash
# Adds the WINNER (cg) to the val-1 confirmation. Separate file rather than an edit to
# drive_dcop_confirm.sh because that script was ALREADY RUNNING when cg won the val-0 A/B, and
# bash re-reads a running script by byte offset -- editing it in place corrupts the rest of the run.
#   WAIT_PID=<pid of drivers/drive_dcop_confirm.sh> GPU=0 bash scripts/drivers/drive_dcop_confirm_cg.sh
set -u
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
CK=logs/fm3d_cq500/ckpt_iter500000.pth
cd /home/mirlab/Desktop/Flow_matching_motion_3D
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${RUN:-1}
SEED=${SEED:-7}

if [ -n "${WAIT_PID:-}" ]; then
    echo "waiting for pid $WAIT_PID (adj+fdk confirmation) to exit ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "pid $WAIT_PID gone, starting  $(date)"
    sleep 20
fi

echo "=== [confirm/cg] val $RUN seed $SEED  $(date) ==="
$PY scripts/run_posterior3d.py --ckpt $CK --split val --run $RUN --seed $SEED \
    --loss l2si --kappa 0 --dc_op cg --out data/confirm_cg_v${RUN}

echo "=== CONFIRM-CG DONE  $(date) ==="
