#!/bin/bash
# ONE GPU, THREE JOBS, IN ORDER. GPU 0 only (user, 2026-08-06).
#
#   1. WAIT for the running quality-metric training (stage 1) to reach iter 10000 and exit
#   2. Thies baseline over the 30-patient cohort -- BOTH amplitudes
#   3. our own FM3D prior, 500k iters, on the NEW `--bridge data` default
#
# (This replaced the 2026-08-03 chain, which waited on the test30 cohort and then launched stage 1
# at --iters 5000. Both of those have happened.)
#
# WHY IT IS A CHAIN AND NOT THREE LAUNCHES. Step 2 must not start before step 1 finishes: it
# freezes `qmnet_best.pth`, and reading that file mid-training freezes whatever checkpoint
# happened to be on disk at that instant. The 5000-iter run was shown to be UNDER-TRAINED -- its
# LAST validation was its best (0.06679) -- which is why it was extended to 10000. Benchmarking
# against a half-trained baseline would repeat exactly the mistake the f_maps correction was made
# to undo (PROVENANCE section 4.4).
#
# Step 3 has no dependency on step 2's numbers; it is sequenced only because one GPU is in play.
#
# GUARD. If step 1 dies early, `qmnet_best.pth` still exists and step 2 would run happily against
# a stale net and print a plausible table. So the chain re-reads the checkpoint's own `iter` and
# refuses to continue below MIN_ITER.
#
#     setsid nohup bash scripts/drivers/drive_chain_thies_qm.sh </dev/null > logs/chain.log 2>&1 &
set -u
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
QM_DIR=logs/bench_thies_qm
MIN_ITER=${MIN_ITER:-9000}
FM3D_OUT=${FM3D_OUT:-logs/fm3d_databridge}

say () { echo "[chain $(date '+%F %T')] $*"; }

# ---- 1. wait for stage 1 -------------------------------------------------------------------
say "waiting for the quality-metric training to exit (target iter 10000)..."
while pgrep -f "bench_thies_train_qm.py --out $QM_DIR" > /dev/null; do sleep 120; done
say "stage 1 process is gone; checking the checkpoint"

IT=$($PY - <<EOF
import torch
try:
    print(int(torch.load("$QM_DIR/qmnet_best.pth", map_location="cpu", weights_only=False)["iter"]))
except Exception:
    print(-1)
EOF
)
say "qmnet_best.pth is from iter $IT (bar: $MIN_ITER)"
if [ "$IT" -lt "$MIN_ITER" ]; then
  say "REFUSING to benchmark against a checkpoint below the bar -- stage 1 must have died."
  say "Fix stage 1, then re-run this chain: steps 2 and 3 skip whatever is already on disk."
  exit 1
fi

# ---- 2. the Thies baseline, both amplitudes ------------------------------------------------
say "step 2: Thies baseline, 30 patients, 10/10 then 5/5"
bash scripts/drivers/drive_thies_test30.sh
say "step 2 done -- 10/10 $(ls data/bench_thies_test30/*/result.json 2>/dev/null | wc -l)/30, 5/5 $(ls data/bench_thies_test30_amp55/*/result.json 2>/dev/null | wc -l)/30"

# ---- 3. our prior, 500k, on the data bridge -------------------------------------------------
# Every argument here is already the DEFAULT -- `--bridge data` has been the default since
# 2026-08-05 and `python scripts/train_fm3d.py --out <dir>` reproduces the deployed recipe
# exactly (a stated invariant of that script). They are spelled out anyway so the log says which
# bridge this run is without anyone having to date the checkpoint against a git log.
say "step 3: FM3D prior, 500k iters, --bridge data -> $FM3D_OUT"
$PY scripts/train_fm3d.py --out "$FM3D_OUT" --bridge data --iters 500000 \
    >> logs/fm3d_databridge.log 2>&1
say "step 3 exited (rc=$?)"
say "chain complete"
