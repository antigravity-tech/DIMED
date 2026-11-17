#!/usr/bin/env python3
"""Export the per-image shared-failure labels (Section 5.1) so that a new model can stratify its own errors.

    python analysis/export_shared_labels.py results/shared_failure_labels.csv

One row per image of the 7 MoGe-2 benchmarks: dataset, filename, k (number of the 11 architectures, given
ground-truth intrinsics, whose metric delta1 < 0.5 on that image), tier (0 / specific 1-2 / intermediate 3-5 /
shared >= 6), and the per-architecture failure flags.
"""
import csv
import sys

import numpy as np

from load_results import ARCH, DS7, FAIL, d1, load, names


def tier(k):
    return 'none' if k == 0 else 'specific' if k <= 2 else 'intermediate' if k <= 5 else 'shared'


def main(out):
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'filename', 'k', 'tier'] + [n for n, _, _ in ARCH])
        for ds in DS7:
            fn = names(load(ARCH[0][2], ds))
            flags = []
            for _, _, run_k in ARCH:
                rows = load(run_k, ds)
                assert names(rows) == fn, f'image order differs for {run_k} on {ds}'
                flags.append(d1(rows) < FAIL)
            flags = np.array(flags)
            for i, name in enumerate(fn):
                k = int(flags[:, i].sum())
                w.writerow([ds, name, k, tier(k)] + [int(x) for x in flags[:, i]])
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'shared_failure_labels.csv')
