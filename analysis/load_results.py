"""Loader for the released per-image results (the 7 MoGe-2 benchmarks and the two TUM-derived sets).

Every statistic in the analysis scripts is recomputed from these files; nothing is typed by hand.

Layout (relative to the release root):
    results/main/<run>/<benchmark>.jsonl     36 model evaluations on the 7 MoGe-2 benchmarks
    results/tum_family/<run>/TUM-Family.jsonl
    results/tum_ref/<run>/TUM-Ref.jsonl

Each line (results v6):
    {"filename", "metric_scale",
     "depth_metric":          {"rel", "delta1", "delta2", "delta3", "sq_rel", "rmse", "rmse_log", "log10"},
     "depth_scale_invariant": {the same eight metrics after multiplying the prediction by s*},
     "scales": {"s_med", "s_lmean", "s_rom", "s_moge64", "s_full"},
     "sa": {"s_lmean" | "s_rom" | "s_moge64" | "s_full": {"rel", "delta1"}},
     "log_stats": {"log_mse", "log_mean", "log_var", "n", "n_bad_pred"}, "far10", "n_pix"}
- metric_scale is s*, the global factor with s* * prediction ~ ground truth: the median of gt / pred over all valid pixels
  (= scales.s_med).  s* > 1: depth underestimated; s* < 1: overestimated.
- depth_scale_invariant holds the metrics after multiplying the prediction by s* ("structure").
- scales / sa: the five alignment scales compared in Appendix C of the paper and the scale-aligned rel / delta1 under each:
  s_med median(gt / pred) (= s*); s_lmean exp(-mean(ln pred - ln gt)), the log-MSE optimal scale; s_rom median(gt) / median(pred);
  s_moge64 / s_full MoGe's solver (min sum |s p - g| / g) on the 64x64 grid of the MoGe-2 protocol / on all valid pixels.
- log_stats: d = ln pred - ln gt over the valid pixels with pred > 0 (Appendix B).  far10: share of pixels with pred / gt > 10.
- HAMMER only: pair, s_bg / s_obj (the five scales on background / object pixels), n_bg_pixels, n_obj_pixels.
- A failure is metric delta1 < 0.5.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
DS7 = ['NYUv2', 'KITTI', 'ETH3D', 'iBims-1', 'DDAD', 'DIODE', 'HAMMER']
FAIL = 0.5

# (display name, run without K, run with K); None = the architecture needs K.  Ordered as in Table 1 of the paper.
ARCH = [
    ('Metric3D-v2', None, 'metric3d_v2'),
    ('DA3METRIC', None, 'da3_metric_withK'),
    ('Depth Pro', 'depth_pro', 'depth_pro_withK'),
    ('MA-DepthMap', None, 'metricanything_depthmap'),
    ('DepthVLM', None, 'depthvlm_withK'),
    ('MoGe-2', 'moge2', 'moge2_withK'),
    ('MoGe-3', 'moge3', 'moge3_withK'),
    ('MA-PointMap', 'metricanything_pointmap', 'metricanything_pointmap_withK'),
    ('FoundationGeo', 'foundationgeo', 'foundationgeo_withK'),
    ('UniDepthV2', 'unidepth_v2', 'unidepth_v2_withK'),
    ('UniK3D', 'unik3d', 'unik3d_withK'),
]
# families used for the family-merged counts (Section 5.1 / Appendix G)
FAMILIES = {
    'MoGe': ['MoGe-2', 'MoGe-3', 'MA-PointMap', 'FoundationGeo'],
    'Uni': ['UniDepthV2', 'UniK3D'],
    'DepthPro': ['Depth Pro', 'MA-DepthMap'],
    'Metric3D': ['Metric3D-v2'],
    'DA3': ['DA3METRIC'],
    'DepthVLM': ['DepthVLM'],
}
# the TUM directories name the two K-only runs of the first batch with a `_withK` suffix
_TUM_ALIAS = {}   # run names are the same in all three sets

_cache = {}


def runs18():
    """[(display name, 'noK' | 'K', run)] in the order of Table 3."""
    out = []
    for name, nok, k in ARCH:
        if nok:
            out.append((name, 'noK', nok))
        out.append((name, 'K', k))
    return out


def load(run, ds, setname='main'):
    key = (setname, run, ds)
    if key not in _cache:
        if setname != 'main':
            run = _TUM_ALIAS.get(run, run)
        path = ROOT / setname / run / f'{ds}.jsonl'
        _cache[key] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return _cache[key]


def _arr(rows, group, key):
    return np.array([r[group][key] for r in rows], dtype=np.float64)


def d1(rows): return _arr(rows, 'depth_metric', 'delta1')
def rel(rows): return _arr(rows, 'depth_metric', 'rel')
def sid1(rows): return _arr(rows, 'depth_scale_invariant', 'delta1')
def sirel(rows): return _arr(rows, 'depth_scale_invariant', 'rel')
def scale(rows): return np.array([r['metric_scale'] for r in rows], dtype=np.float64)
def names(rows): return [r['filename'] for r in rows]


def scene(ds, filename):
    """Recorded scene of an image (used for scene-level resampling); other datasets resample images."""
    parts = filename.split('/')
    if ds == 'DIODE':
        return next(p for p in parts if p.startswith('scene_'))
    if ds == 'HAMMER':
        return parts[0].split('_traj')[0]
    if ds == 'ETH3D':
        return parts[0]
    return filename
