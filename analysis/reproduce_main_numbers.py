#!/usr/bin/env python3
"""Recompute the main numbers of the paper from the released per-image results.

    python analysis/reproduce_main_numbers.py            # prints Table 3 and the headline numbers
    python analysis/reproduce_main_numbers.py --json out.json

Covers: Table 3 (18 evaluations, 7 benchmarks, datasets weighted equally), the share of metric failures that pass
after scale alignment (Section 4.2), the shared-failure grading k with its tiers (Table 5), the share of the 1 - delta1
error of each model that falls on shared-failure images (Table 7), and the best-of-18 per-image reference (Section 4.1).
"""
import argparse
import json

import numpy as np

from load_results import ARCH, DS7, FAIL, d1, load, rel, runs18, scale, sid1, sirel


def table2():
    rows = []
    for name, cond, run in runs18():
        per = {ds: load(run, ds) for ds in DS7}
        ls = {ds: np.log(scale(r)) for ds, r in per.items()}
        eq = lambda f: float(np.mean([f(ds, r) for ds, r in per.items()]))  # noqa: E731  equal weight over datasets
        rows.append(dict(
            model=name, K=cond,
            metric_d1=eq(lambda ds, r: d1(r).mean()), metric_rel=eq(lambda ds, r: rel(r).mean()),
            fail=eq(lambda ds, r: (d1(r) < FAIL).mean()),
            sa_d1=eq(lambda ds, r: sid1(r).mean()), sa_rel=eq(lambda ds, r: sirel(r).mean()),
            sa_fail=eq(lambda ds, r: (sid1(r) < FAIL).mean()),
            bias=eq(lambda ds, r: (-ls[ds]).mean()), dispersion=eq(lambda ds, r: ls[ds].std()),
            abs_err=eq(lambda ds, r: np.abs(ls[ds]).mean()),
        ))
    return rows


def failures_fixed_by_scale():
    m, s = [], []
    for _, _, run in runs18():
        for ds in DS7:
            m.append(d1(load(run, ds)))
            s.append(sid1(load(run, ds)))
    m, s = np.concatenate(m), np.concatenate(s)
    f = m < FAIL
    return dict(pairs=int(len(m)), metric_failures=int(f.sum()), pass_after_alignment=float((s[f] >= FAIL).mean()),
                structure_intact=float((s[f] >= 0.9).mean()), structure_failure=float((s[f] < FAIL).mean()))


def shared_failures():
    """k = number of the 11 architectures (with K) failing an image; tiers: 0, specific 1-2, intermediate 3-5, shared >= 6."""
    k = {ds: np.sum([d1(load(run_k, ds)) < FAIL for _, _, run_k in ARCH], axis=0) for ds in DS7}
    tiers = {ds: dict(k0=float((v == 0).mean()), specific=float(((v >= 1) & (v <= 2)).mean()),
                      intermediate=float(((v >= 3) & (v <= 5)).mean()), shared=float((v >= 6).mean()),
                      n_shared=int((v >= 6).sum()), n_all11=int((v == 11).sum())) for ds, v in k.items()}
    share_eq = float(np.mean([t['shared'] for t in tiers.values()]))
    err_share = {}
    for name, _, run_k in ARCH:
        e = {ds: 1 - d1(load(run_k, ds)) for ds in DS7}
        on_shared = np.mean([e[ds][k[ds] >= 6].sum() / len(e[ds]) for ds in DS7])
        err_share[name] = float(on_shared / np.mean([e[ds].mean() for ds in DS7]))
    return k, dict(per_dataset=tiers, shared_images=int(sum(t['n_shared'] for t in tiers.values())),
                   shared_share_equal_weight=share_eq, error_share_on_shared=err_share)


def best_of_18():
    out = {}
    for ds in DS7:
        allfail = np.all([d1(load(run, ds)) < FAIL for _, _, run in runs18()], axis=0)
        out[ds] = dict(n=int(allfail.sum()), rate=float(allfail.mean()))
    return dict(per_dataset=out, equal_weight_rate=float(np.mean([v['rate'] for v in out.values()])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', help='also write all numbers to this file')
    a = ap.parse_args()
    t2 = table2()
    print('Table 3 (7 benchmarks, equal weight)')
    print(f"{'model':14s} {'K':4s} {'d1':>6s} {'Rel':>6s} {'Fail':>6s} {'SA d1':>6s} {'SA Rel':>6s} {'SA F':>6s} {'bias':>7s} {'dispersion':>10s} {'abs':>6s}")
    for r in t2:
        print(f"{r['model']:14s} {r['K']:4s} {r['metric_d1']:6.3f} {r['metric_rel']:6.3f} {100*r['fail']:5.1f}% {r['sa_d1']:6.3f} "
              f"{r['sa_rel']:6.3f} {100*r['sa_fail']:5.1f}% {r['bias']:+7.3f} {r['dispersion']:10.3f} {r['abs_err']:6.3f}")
    fx = failures_fixed_by_scale()
    print(f"\nMetric failures: {fx['metric_failures']} of {fx['pairs']} image x evaluation pairs; "
          f"{100*fx['pass_after_alignment']:.1f}% pass after scale alignment, {100*fx['structure_intact']:.1f}% have intact structure (SA d1 >= 0.9)")
    _, sh = shared_failures()
    print(f"Shared-failure images (k >= 6 of 11, with K): {sh['shared_images']}, {100*sh['shared_share_equal_weight']:.1f}% of images (datasets weighted equally)")
    print('Share of 1 - delta1 error on shared-failure images: ' + ', '.join(f"{n} {100*v:.0f}%" for n, v in sh['error_share_on_shared'].items()))
    b = best_of_18()
    print(f"Best of the 18 evaluations per image: failure rate {100*b['equal_weight_rate']:.1f}% (equal weight); "
          + ', '.join(f"{ds} {v['n']}" for ds, v in b['per_dataset'].items() if v['n']))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(table2=t2, failures=fx, shared=sh, best_of_18=b), f, indent=1)


if __name__ == '__main__':
    main()
