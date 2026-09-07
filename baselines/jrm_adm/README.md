# JRM-ADM baseline: glue for the authors' released code

JRM-ADM (De Paepe et al., "Adaptive diffusion models for sparse-view motion-corrected head
cone-beam CT", IEEE TRPMS 2025) is run with the authors' code, unchanged:
<https://github.com/antoinedepaepe/jrm-adm>. Nothing of theirs is vendored here. The files in
this directory are OURS and are meant to be dropped into a clone of their repository
(`refs/jrm-adm/` in our layout, which the scripts below assume):

| file | role |
|---|---|
| `PROVENANCE.md` | what we verified about their code (amplitude convention, loop structure, protocol deltas) and every adaptation decision |
| `train_w3dm.py` | retrains their W3DM wavelet-domain prior on our 150 training patients (they released inference only) |
| `run_on_ours.py` | runs their solver stack, byte-identical, on our cohort measurements exported by `scripts/export_cohort_for_jrm.py` |
| `run_cohort.sh` | the 30-patient driver (`--vol 224 224 224 --gamma 3.3e4 --prior_zflip`, see PROVENANCE) |
| `score_demo.py` | scores their demo run in their own Table I convention (reproduction check) |

Pipeline:

```
python scripts/export_cohort_for_jrm.py --n 30 --out refs/jrm-adm/data/ours_cohort   # our env
cd refs/jrm-adm && ./run_cohort.sh                                                  # their env
python scripts/score_jrm_native.py                                                  # our env
```

`scripts/jrm_theta_convert.py` converts their per-view affine motion into our 6-DoF convention
so that FDK(theta_hat) and the reprojection error can be computed with our operator.
