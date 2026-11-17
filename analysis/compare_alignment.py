"""Appendix C: compare the paper's alignment scale s* = median(gt / pred) with four others, from the released per-image
results of the 18 main evaluations (fields `scales` and `sa`):

    s_med     median(gt / pred)                    the paper's s* (L1-optimal for the per-pixel log error)
    s_lmean   exp(-mean(ln pred - ln gt))          log-MSE optimal scale (L2), the e^{-mu} of Appendix B
    s_rom     median(gt) / median(pred)            ratio of medians (early self-supervised evaluation)
    s_moge64  MoGe's solver on the 64x64 grid      min sum |s p - g| / g, as in the MoGe-2 protocol
    s_full    MoGe's solver on all valid pixels

    python compare_alignment.py

Prints:
  1. agreement with s_med (share of image x evaluation pairs within 1% / 5%), per benchmark (Table C.1);
  2. pooled scale-aligned delta1 / Rel, structure failures (scale-aligned delta1 < 0.5), metric passes that fail after
     alignment, s < 0.05, and the share of metric failures with intact structure / fixed by the scale;
  3. per model, equal weight over the 7 benchmarks: dispersion std(ln s), bias mean(ln 1/s), structure failure rate (Table C.2);
  4. the images that MoGe's 64x64 solution marks as structure failures in at least 9 of the 18 evaluations while s_med does so in fewer.
"""
import numpy as np

from load_results import DS7, FAIL, load, runs18

KEYS = ['s_med', 's_lmean', 's_rom', 's_moge64', 's_full']


def s_of(r, k):
    return r['scales'][k]


def sad1(r, k):
    return r['depth_scale_invariant']['delta1'] if k == 's_med' else r['sa'][k]['delta1']


def sarel(r, k):
    return r['depth_scale_invariant']['rel'] if k == 's_med' else r['sa'][k]['rel']


def main():
    runs = runs18()
    data = {ds: [x for _, _, r in runs for x in load(r, ds)] for ds in DS7}
    data['all'] = [x for ds in DS7 for x in data[ds]]

    print('1. Agreement with s_med: share of image x evaluation pairs within 1% / 5% (18 evaluations)')
    print('%-8s ' % '' + ' '.join('%15s' % k for k in KEYS[1:]))
    for ds in DS7 + ['all']:
        cells = []
        for k in KEYS[1:]:
            g = np.abs(np.log(np.array([s_of(x, k) / s_of(x, 's_med') for x in data[ds]])))
            cells.append('%6.1f%% %6.1f%%' % (100 * np.mean(g < np.log(1.01)), 100 * np.mean(g < np.log(1.05))))
        print('%-8s ' % ds + ' '.join('%15s' % c for c in cells))

    print('\n2. Pooled over the 18 evaluations')
    print('%-9s %7s %7s %8s %9s %7s %8s %7s' % ('', 'SA d1', 'SA Rel', 'SA fail', 'ok->fail', 's<0.05', 'intact', 'fixed'))
    rows = data['all']; m = np.array([x['depth_metric']['delta1'] for x in rows]); mf = m < FAIL
    for k in KEYS:
        a = np.array([sad1(x, k) for x in rows]); rl = np.array([sarel(x, k) for x in rows]); s = np.array([s_of(x, k) for x in rows])
        print('%-9s %7.4f %7.4f %8d %9d %7d %7.1f%% %6.1f%%' % (k, a.mean(), rl.mean(), (a < FAIL).sum(), ((m >= FAIL) & (a < FAIL)).sum(),
              (s < 0.05).sum(), 100 * (a[mf] >= 0.9).mean(), 100 * (a[mf] >= FAIL).mean()))

    print('\n3. Per model, equal weight over the 7 benchmarks: dispersion / bias / structure failure rate')
    print('%-14s %-3s | %s' % ('model', 'K', ' | '.join('%-23s' % k for k in KEYS)))
    for n, c, r in runs:
        cells = []
        for k in KEYS:
            st, bi, sf = [], [], []
            for ds in DS7:
                R = load(r, ds); ls = np.log(1 / np.array([s_of(x, k) for x in R])); a = np.array([sad1(x, k) for x in R])
                st.append(ls.std()); bi.append(ls.mean()); sf.append((a < FAIL).mean())
            cells.append('%.3f %+.3f %4.1f%%' % (np.mean(st), np.mean(bi), 100 * np.mean(sf)))
        print('%-14s %-3s | %s' % (n, c, ' | '.join('%-23s' % x for x in cells)))

    print("\n4. Images where MoGe's 64x64 solution gives SA delta1 < 0.5 in >= 9 of 18 evaluations, s_med in fewer")
    for ds in DS7:
        R = [load(r, ds) for _, _, r in runs]
        for i, f in enumerate(x['filename'] for x in R[0]):
            v = [Q[i] for Q in R]
            n64 = sum(sad1(x, 's_moge64') < FAIL for x in v); nmed = sum(sad1(x, 's_med') < FAIL for x in v)
            if n64 >= 9 and nmed < n64:
                print('  %-8s %-62s 64x64 %2d / median %2d of 18 | median s over the 18: ' % (ds, f, n64, nmed)
                      + ', '.join('%s %.3f' % (k[2:], np.median([s_of(x, k) for x in v])) for k in KEYS))


if __name__ == '__main__':
    main()
