"""Per-image evaluation of a single-image metric depth model on the MoGe-2 benchmarks (or the TUM sets).

Uses MoGe's unmodified data loader (EvalDataLoaderPipeline) and metric code (compute_metrics, delta_depth); the model is
loaded through an adapter in eval/baselines implementing MoGe's MGEBaselineInterface.

    python scripts/eval_depth.py --baseline baselines/moge.py --output_dir <out>/<run> [--with_intrinsics] [model args]

--with_intrinsics passes the ground-truth intrinsics to the model (models that need them must use it).

Alignment scale.  For every image we solve one global scale s* with s* * pred ~ gt as the MEDIAN of gt / pred over all valid
pixels (= exp(median(ln gt - ln pred)), the L1-optimal scale for the per-pixel log error).  We do not use MoGe's solver
(min sum |s pred - gt| / gt on a 64x64 nearest-neighbour grid): that objective is a weighted median of gt / pred with
weights pred / gt, so it leans towards small s (depth taken as too near) and a few pixels with very large pred / gt (sky
borders, sparse LiDAR noise, depth clipped at a maximum) can pull it close to 0.  Four other scales are stored for
comparison (Appendix C of the paper): the log mean exp(-mean(ln pred - ln gt)) (log-MSE optimal), the ratio of medians
median(gt) / median(pred), and MoGe's solver on the 64x64 grid and on all valid pixels.

Output: <output_dir>/samples/<benchmark>.jsonl, one line per image (the format of the released results v6):
    filename
    metric_scale            s* (median)
    depth_metric            {rel, delta1, delta2, delta3, sq_rel, rmse, rmse_log, log10}   prediction vs ground truth
    depth_scale_invariant   the same eight metrics for s* x prediction vs ground truth (structure)
    scales                  {s_med (= s*), s_lmean, s_rom, s_moge64, s_full}
    sa                      {s_lmean | s_rom | s_moge64 | s_full: {rel, delta1}}   after multiplying by each scale
    log_stats               {log_mse, log_mean, log_var, n, n_bad_pred}    d = ln pred - ln gt on pixels with pred > 0
    far10                   share of valid pixels with pred / gt > 10
    n_pix                   number of valid pixels
and <output_dir>/results.json (dataset means; scale bias / dispersion / abs_err of ln s*; equal-weight mean).
Benchmarks whose samples/<benchmark>.jsonl already exists are skipped (resume).
"""
import json
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent))
from eval_paths import add_moge_to_path, resolve_data_path  # noqa: E402
add_moge_to_path()  # unmodified upstream MoGe from <root>/third_party/MoGe

from typing import Sequence

import click
import torch
from tqdm import tqdm

from moge.test.dataloader import EvalDataLoaderPipeline
from moge.test.metrics import compute_metrics, delta_depth
from moge.utils.alignment import align_depth_scale
from moge.utils.geometry_torch import mask_aware_nearest_resize
from moge.utils.tools import import_file_as_module, write_jsonl


def _load_baseline(path: str, extra_args: Sequence[str]):
    return getattr(import_file_as_module(path, Path(path).stem), 'Baseline').load.main(list(extra_args), standalone_mode=False)


def median_scale(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """s* = median(gt / pred) over the given (valid) pixels."""
    return torch.median(gt / pred.clamp_min(1e-7)).item()


def all_scales(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor) -> dict:
    """The five alignment scales on the valid pixels: median (s*), log mean, ratio of medians, MoGe's solver (MoGe-2
    protocol: 64x64 mask-aware nearest resize, weight 1/gt) and the same objective on all pixels."""
    _, lr_mask, lr_index = mask_aware_nearest_resize(None, mask, (64, 64), return_index=True)
    p_lr, g_lr = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
    p, g = pred_depth[mask].float(), gt_depth[mask].float()
    ok = torch.isfinite(p) & (p > 0)
    return {'s_med': median_scale(p, g),
            's_lmean': torch.exp(-(torch.log(p[ok].double()) - torch.log(g[ok].double())).mean()).item() if ok.any() else float('nan'),
            's_rom': (torch.median(g) / torch.median(p).clamp_min(1e-7)).item(),
            's_moge64': align_depth_scale(p_lr, g_lr, 1 / g_lr).item(),
            's_full': align_depth_scale(p, g, 1 / g).item()}


def _depth_metrics(q, g, full=True):
    """Metrics of q against g.  full=False: rel, delta1.  full=True: also delta2/3, sq_rel, rmse, rmse_log, log10
    (the log metrics on the pixels with q > 0)."""
    out = {'rel': ((q - g).abs() / g).mean().item(), 'delta1': delta_depth(q, g, 1.25)}
    if full:
        out['delta2'] = delta_depth(q, g, 1.25 ** 2); out['delta3'] = delta_depth(q, g, 1.25 ** 3)
        qd, gd = q.double(), g.double()
        out['sq_rel'] = ((qd - gd) ** 2 / gd).mean().item()
        out['rmse'] = ((qd - gd) ** 2).mean().sqrt().item()
        ok = torch.isfinite(qd) & (qd > 0)
        lr = torch.log(qd[ok]) - torch.log(gd[ok])
        out['rmse_log'] = (lr ** 2).mean().sqrt().item() if lr.numel() else float('nan')
        out['log10'] = (lr.abs() / math.log(10)).mean().item() if lr.numel() else float('nan')
    return out


def evaluate_image(pred: dict, sample: dict) -> dict:
    gt, mask = sample['depth'], sample['depth_mask']
    metrics, _ = compute_metrics(pred, sample, mg='depth_metric')   # MoGe's metric-depth group (rel, delta1)
    p, g = pred['depth_metric'][mask].float(), gt[mask].float()
    sc = all_scales(pred['depth_metric'], gt, mask)
    s = sc['s_med']
    ok = torch.isfinite(p) & (p > 0)
    d = torch.log(p[ok].double()) - torch.log(g[ok].double())
    lm = d.mean() if d.numel() else torch.tensor(float('nan'))
    dm = _depth_metrics(p, g)
    return {'filename': sample['filename'], 'metric_scale': s,
            'depth_metric': {**metrics['depth_metric'], **{k: v for k, v in dm.items() if k not in ('rel', 'delta1')}},
            'depth_scale_invariant': _depth_metrics(p * s, g),
            'scales': sc, 'sa': {k: _depth_metrics(p * sc[k], g, False) for k in ('s_lmean', 's_rom', 's_moge64', 's_full')},
            'log_stats': {'log_mse': (d ** 2).mean().item(), 'log_mean': lm.item(), 'log_var': ((d - lm) ** 2).mean().item(),
                          'n': int(d.numel()), 'n_bad_pred': int((~ok).sum().item())},
            'far10': ((p / g) > 10).float().mean().item(), 'n_pix': int(p.numel())}


def _summary(rows):
    mean = lambda v: sum(v) / len(v)
    ls = [math.log(1 / r['metric_scale']) for r in rows]; mu = mean(ls)
    return {'num_samples': len(rows),
            'depth_metric': {k: mean([r['depth_metric'][k] for r in rows]) for k in ('rel', 'delta1')},
            'depth_scale_invariant': {k: mean([r['depth_scale_invariant'][k] for r in rows]) for k in ('rel', 'delta1')},
            'fail_rate': mean([r['depth_metric']['delta1'] < 0.5 for r in rows]),
            'scale': {'bias': mu, 'dispersion': math.sqrt(mean([(v - mu) ** 2 for v in ls])), 'abs_err': mean([abs(v) for v in ls])}}


def evaluate(baseline, benchmarks, output_dir: Path, with_intrinsics: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in benchmarks:
        samples_path = output_dir / 'samples' / f'{name}.jsonl'
        if samples_path.exists():
            print(f'skip {name}: {samples_path} exists', flush=True)
            continue
        records = []
        pipe = EvalDataLoaderPipeline(**{**cfg, 'path': resolve_data_path(cfg['path'])})
        with pipe, tqdm(total=len(pipe), desc=name, mininterval=30) as pbar:
            for _ in range(len(pipe)):
                sample = pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                with torch.inference_mode():
                    pred = baseline.infer_for_evaluation(sample['image'], sample['intrinsics']) if with_intrinsics \
                        else baseline.infer_for_evaluation(sample['image'])
                    records.append(evaluate_image(pred, sample))
                pbar.update(1)
        tmp = samples_path.with_suffix('.jsonl.tmp')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, tmp)
        os.replace(tmp, samples_path)
        print(f'wrote {samples_path} ({len(records)})', flush=True)
    res = {}
    for f in sorted((output_dir / 'samples').glob('*.jsonl')):
        res[f.stem] = _summary([json.loads(l) for l in open(f)])
    if res:
        ks = list(res); mean = lambda v: sum(v) / len(v)
        res['mean_equal_weight'] = {g: {k: mean([res[d][g][k] for d in ks]) for k in res[ks[0]][g]} for g in ('depth_metric', 'depth_scale_invariant', 'scale')}
        res['mean_equal_weight']['fail_rate'] = mean([res[d]['fail_rate'] for d in ks])
        (output_dir / 'results.json').write_text(json.dumps(res, indent=1))


@click.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Adapter in eval/baselines.')
@click.option('--config', 'config_path', type=click.Path(), default=str(Path(__file__).absolute().parents[1] / 'configs' / 'metric_depth_only.json'))
@click.option('--output_dir', type=click.Path(), required=True)
@click.option('--with_intrinsics', is_flag=True, help='Pass the ground-truth intrinsics to the model.')
@click.option('--only', type=str, default=None, help='Comma-separated subset of the benchmarks in the config.')
@click.pass_context
def main(ctx, baseline_code_path, config_path, output_dir, with_intrinsics, only):
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    benchmarks = list(json.load(open(config_path)).items())
    if only:
        benchmarks = [b for b in benchmarks if b[0] in only.split(',')]
    evaluate(_load_baseline(baseline_code_path, ctx.args), benchmarks, Path(output_dir), with_intrinsics)


if __name__ == '__main__':
    main()
