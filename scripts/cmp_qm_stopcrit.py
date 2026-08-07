"""Read the stop-criterion sweep: does more stage-1 training still buy RPE?

    python scripts/cmp_qm_stopcrit.py

Reads whatever `scripts/drivers/drive_qm_stopcrit.sh` has written so far -- it is safe to run
mid-sweep and only compares patients present for EVERY checkpoint, because the whole point is the
paired difference and an unpaired mean over a different patient subset per checkpoint would move
for reasons that have nothing to do with the weights.

HOW TO READ THE OUTPUT
  * The CONTROL row (2000 vs 7000) has to separate. If it does not, RPE is insensitive to this
    net over this range and the sweep is INCONCLUSIVE -- it does not license stopping.
  * The QUESTION row (4000 vs 7000) is the decision. Flat there, with the control alive, means
    the val-L1 gains have stopped reaching the metric and the remaining budget buys nothing.
  * `rpe_at` traces RPE along the 100 GD steps (the estimator prints it every 5). A checkpoint
    can reach the same final RPE by a different path; if the late checkpoints only converge
    FASTER, that is a real gain the final-value comparison hides, and the iteration count is
    fixed at 100 so it would not show up in the benchmark either.

WHY NOT A p-VALUE HERE. n=5 patients. A signed-rank test on 5 pairs cannot go below p=0.0625
even when every pair agrees, so it can never reach 0.05 and quoting it would be theatre. The
honest summary at this n is the paired mean difference, its spread, and how many of the 5 pairs
agree in sign -- all three are printed.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = "data/qm_stopcrit"
CKPTS = (2000, 4000, 7000)
PATIENTS = (0, 1, 2, 3, 4)


def load():
    """-> {ckpt: {patient: record}}, only patients complete across every checkpoint."""
    got: dict[int, dict[int, dict]] = {c: {} for c in CKPTS}
    for c in CKPTS:
        for p in PATIENTS:
            f = os.path.join(ROOT, f"it{c}_v{p}", "result.json")
            if os.path.exists(f):
                got[c][p] = json.load(open(f))
    common = sorted(set.intersection(*(set(got[c]) for c in CKPTS)))
    return got, common


def main():
    got, common = load()
    have = {c: len(got[c]) for c in CKPTS}
    print(f"runs on disk: " + "  ".join(f"it{c}={have[c]}/{len(PATIENTS)}" for c in CKPTS))
    if not common:
        raise SystemExit("no patient is complete across all checkpoints yet")
    print(f"PAIRED over patients {common} (val split, seed 2000+i, 10/10 p2p)\n")

    def col(c, key):
        return np.array([got[c][p]["rpe" if key == "raw" else "rpe_zero_centred"]["rpe_mm"]
                         for p in common])

    print("=" * 72)
    print("final RPE [mm] after the fixed 100 GD steps")
    print("=" * 72)
    print(f"  {'checkpoint':>12s}  {'raw':>18s}  {'zero-centred':>18s}")
    for c in CKPTS:
        r, z = col(c, "raw"), col(c, "zc")
        print(f"  {'it ' + str(c):>12s}  {r.mean():7.3f} +- {r.std(ddof=1):5.3f}  "
              f"{z.mean():7.3f} +- {z.std(ddof=1):5.3f}")

    print()
    print("=" * 72)
    print("PAIRED differences (negative = the later checkpoint is better)")
    print("=" * 72)
    for label, a, b in (("CONTROL   it7000 - it2000", 7000, 2000),
                        ("QUESTION  it7000 - it4000", 7000, 4000),
                        ("          it4000 - it2000", 4000, 2000)):
        for key, name in (("raw", "raw"), ("zc", "zeroc")):
            d = col(a, key) - col(b, key)
            print(f"  {label:26s} {name:6s} {d.mean():+7.3f} +- {d.std(ddof=1):5.3f} mm   "
                  f"({int((d < 0).sum())}/{len(d)} pairs improved)   "
                  f"per-pair [{', '.join(f'{x:+.2f}' for x in d)}]")

    print()
    print("=" * 72)
    print("RPE trajectory along the 100 GD steps (mean over the paired patients)")
    print("=" * 72)
    # The estimator logs RPE every 5 iterations into `history`... except it does not: `history`
    # carries (n, f, step, grad) only, and RPE is printed to stdout. So parse the logs.
    import re
    for c in CKPTS:
        tr = []
        for p in common:
            lg = os.path.join("logs", "qm_stopcrit", f"it{c}_v{p}.log")
            if not os.path.exists(lg):
                tr = []
                break
            v = [(int(m.group(1)), float(m.group(2))) for m in
                 (re.match(r"\s+it\s+(\d+).*RPE ([\d.]+) mm", ln) for ln in open(lg)) if m]
            tr.append(dict(v))
        if not tr:
            print(f"  it {c}: logs missing")
            continue
        ns = sorted(set.intersection(*(set(t) for t in tr)))
        show = [n for n in ns if n % 20 == 0 or n == ns[-1]]
        print(f"  it {c}: " + "  ".join(
            f"n{n}={np.mean([t[n] for t in tr]):.3f}" for n in show))

    print("\nNOTE n=5: no p-value is reported (a 5-pair signed-rank test floors at p=0.0625).")


if __name__ == "__main__":
    main()
