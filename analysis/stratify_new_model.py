#!/usr/bin/env python3
"""Stratify a new model's per-image results by the released shared-failure labels.

    python analysis/stratify_new_model.py results/shared_failure_labels.csv path/to/my_run

`path/to/my_run` must contain `<benchmark>.jsonl` files in the same format as results/main/<run>/ (the output of
eval/run.sh, `samples/` directory).  Prints, per tier, the failure rate on that tier's images and the share of the model's
mean 1 - delta1 error that falls on that tier, as in Table 7 of the paper: the mean error is the average over images within
each dataset followed by the equal-weight average over datasets, and the shares of the four tiers sum to 100%.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TIERS = ['none', 'specific', 'intermediate', 'shared']


def main(labels_csv, run_dir):
    labels = defaultdict(dict)
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            labels[r['dataset']][r['filename']] = r['tier']
    fail = defaultdict(list)          # per dataset: failure rate on the tier's images
    contrib = defaultdict(float)      # per tier: sum over datasets of (sum of 1 - delta1 on the tier) / (images in the dataset)
    total, n_ds = 0.0, 0
    for ds, lab in labels.items():
        p = Path(run_dir) / f'{ds}.jsonl'
        if not p.exists():
            print('missing', p)
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines() if x]
        t = np.array([lab[r['filename']] for r in rows])
        e = np.array([1 - r['depth_metric']['delta1'] for r in rows])
        for tt in TIERS:
            m = t == tt
            if m.any():
                fail[tt].append(float((e[m] > 0.5).mean()))
            contrib[tt] += float(e[m].sum() / len(e))
        total += float(e.mean()); n_ds += 1
    share = {tt: contrib[tt] / total if total > 0 else 0.0 for tt in TIERS}
    print(f"{'tier':14s} {'failure rate':>13s} {'share of 1-d1 error':>20s}")
    for tt in TIERS:
        fr = f"{100*np.mean(fail[tt]):12.1f}%" if fail[tt] else f"{'—':>13s}"
        print(f"{tt:14s} {fr} {100*share[tt]:19.1f}%")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
