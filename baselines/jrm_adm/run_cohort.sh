#!/bin/bash
# Sequential test30 cohort driver for the JRM-ADM port (A2d, 2026-08-18). OURS, not upstream.
# One patient at a time (each run holds ~26 GB on the 224^3 grid), ~4.6 h/patient measured on
# p00 -> ~5.6 days for the remaining 29. Skips patients whose result already exists, so a
# restart after any death resumes where it stopped. Launch:
#   cd refs/jrm-adm && setsid nohup ./run_cohort.sh </dev/null > ../../logs/jrm_cohort.log 2>&1 &
set -u
PY=/home/mirlab/anaconda3/envs/jrm_adm/bin/python
cd "$(dirname "$0")"
for i in $(seq 0 29); do
    tag=$(printf "p%02d" "$i")
    out="data/recon_ours_v2/${tag}_result.pt"
    if [ -f "$out" ]; then
        echo "== $tag: exists, skip"
        continue
    fi
    echo "== $tag: start $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES=0 $PY -u run_on_ours.py --case "data/ours_cohort/${tag}.pt" \
        --out data/recon_ours_v2 --vol 224 224 224 --gamma 3.3e4 --prior_zflip \
        > "/tmp/jrm_cohort_${tag}.log" 2>&1
    rc=$?
    echo "== $tag: rc=$rc $(date '+%F %T')"
    [ $rc -ne 0 ] && tail -3 "/tmp/jrm_cohort_${tag}.log"
done
echo "COHORT DONE $(date '+%F %T')"
