#!/bin/bash
# AFTER THE DATABRIDGE 500k: our full test cohort on the NEW prior, then the Thies 5/5 arm.
# Armed 2026-08-08 (user: "데이터 브릿지 끝나면 우리도 테스트셋 다 추론돌려서 결과 올려두고
# 그다음 5/5 thies스케줄해놔"). GPU0, strictly sequential, one job at a time.
#
#   1. WAIT for the running `train_fm3d.py --out logs/fm3d_databridge` (chain step 3 of
#      drive_chain_thies_qm.sh) to exit, then REFUSE to continue unless ckpt_iter500000.pth
#      exists -- the same died-early guard the QM chain uses.
#   2. OUR 30-patient test cohort on the DATABRIDGE prior -> data/fm3d_test30_databridge.
#      Identical (split=test, run=i, seed=1000+i) triples as data/fm3d_test30 (old prior) and
#      data/bench_thies_test30 (baseline), so all three are pairwise comparable -- the seeds are
#      the contract (see drive_test30.sh's header). Deployed inference defaults throughout.
#      ~9.3 min/patient -> ~4.7 h.
#   3. The Thies baseline at the PAPER's OWN 5/5 amplitude (RUN_AMP55=1 pass of
#      drive_thies_test30.sh; its 10/10 pass skips itself -- results already on disk).
#      This is the "is the reimplementation in the paper's range?" check, ~1.7 h.
#
#     setsid nohup bash scripts/drivers/drive_chain_databridge_eval.sh </dev/null > logs/chain_databridge_eval.log 2>&1 &
set -u
cd /home/mirlab/Desktop/Flow_matching_motion_3D
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
export CUDA_VISIBLE_DEVICES=0
CK=logs/fm3d_databridge/ckpt_iter500000.pth

say () { echo "[chain-dbe $(date '+%F %T')] $*"; }

# ---- 1. wait for the databridge training --------------------------------------------------
say "waiting for train_fm3d (logs/fm3d_databridge) to exit..."
while pgrep -f "train_fm3d.py --out logs/fm3d_databridge" > /dev/null; do sleep 300; done
say "training process is gone; checking the checkpoint"
if [ ! -f "$CK" ]; then
  say "REFUSING: $CK does not exist -- the 500k run must have died early."
  say "Fix/resume the training, then re-run this chain; step 2 skips whatever is on disk."
  exit 1
fi

# ---- 2. our test cohort on the new prior --------------------------------------------------
say "step 2: OUR cohort on the databridge prior -> data/fm3d_test30_databridge"
mkdir -p logs/test30_databridge data/fm3d_test30_databridge
for i in $(seq 0 29); do
  tag=$(printf "p%02d" "$i")
  if [ -f "data/fm3d_test30_databridge/$tag/result.pt" ]; then
    echo "##### $tag :: already done, skipping #####"
    continue
  fi
  echo "##### $tag :: split=test run=$i seed=$((1000 + i)) :: $(date +%H:%M:%S) #####"
  $PY scripts/run_posterior3d.py --ckpt $CK --split test --run "$i" \
      --seed $((1000 + i)) --out "data/fm3d_test30_databridge/$tag" \
      > "logs/test30_databridge/$tag.log" 2>&1
  echo "--- $tag exit $? ---"
  grep -E "FINAL|^step  49" "logs/test30_databridge/$tag.log" | tail -4
done
say "step 2 done -- $(ls data/fm3d_test30_databridge/*/result.pt 2>/dev/null | wc -l)/30"

# ---- 3. the Thies 5/5 arm ------------------------------------------------------------------
say "step 3: Thies baseline at the paper's 5/5 (RUN_AMP55=1)"
RUN_AMP55=1 bash scripts/drivers/drive_thies_test30.sh
say "step 3 done -- 5/5 $(ls data/bench_thies_test30_amp55/*/result.json 2>/dev/null | wc -l)/30"
say "chain complete. Compare: FM3D_TEST30=data/fm3d_test30_databridge python scripts/cmp_thies_vs_ours.py (old-prior cohort stays at data/fm3d_test30)"
