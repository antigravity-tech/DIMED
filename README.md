# DIMED: DIagnosing MEtric Depth

**DIMED** evaluates single-image metric depth models image by image instead of only by dataset averages. For every image it keeps the metric error, the error after a single global scale alignment (structure), and the alignment scale `s`* itself. It then reports:

- **failure rate**: the share of images whose metric δ1 < 0.5;
- **structure vs. scale**: scale-aligned δ1 / Rel for structure, and three statistics of `ln s`* for scale: **bias** (signed: is depth over- or underestimated?), **dispersion** (does the error change from image to image?) and **absolute error**;
- **shared vs. model-specific failures**: for each image, how many models fail on it.

This is the code and data of the paper

> **What the Averages Hide: Monocular Metric Depth Models Still Fail in Everyday Scenes, Mostly on Scale**  
>
> [Hualie Jiang](https://hualie.github.io/)1, Yujie Luo1,2, Ziyang Song1, Ximeng Li1, Zhiqiang Lou1, Rui Xu1  
>
> 1Antigravity Tech, Insta360   2Nanyang Technological University  
>
> [arXiv](https://arxiv.org/abs/2611.20518) · [Project page](https://antigravity-tech.github.io/DIMED/) · [Data on Hugging Face](https://huggingface.co/datasets/hljiang/DIMED)

The evaluation imports MoGe's data loader and metrics from an unmodified upstream [MoGe](https://github.com/microsoft/MoGe) checkout and adds adapters for 11 models.

## What we found

On the 7 everyday-scene benchmarks used by MoGe-2 (4,406 images), with 11 recent architectures evaluated with and without ground-truth intrinsics:

- The best of the 18 evaluations still has a failure rate of **7.7%** (datasets weighted equally); on DIODE and HAMMER even the lowest failure rates are 15.2% and 23.9%.
- **78.8%** of the failures have intact structure and are wrong only in scale, and after an ideal scale correction from the ground truth **98.3%** no longer fail: the remaining difficulty is mainly scale.
- Images failed by at least 6 of the 11 architectures are 7.3% of the data, concentrated on HAMMER (4 of its 7 scenes) and DIODE; even when selected only by architectures of other families, they hold 34% to 40% of the top-6 models' 1 − δ1 error, against 14% to 18% for as many random images from the same datasets.

## Contents

| Path | What |
|---|---|
| `eval/` | Model adapters, evaluation scripts (without / with ground-truth intrinsics), configs, launcher `run.sh` |
| `analysis/` | Recompute the paper's main numbers from per-image results; compare the median alignment with four other alignment scales (Appendix C); export shared-failure labels; stratify a new model by them |
| `results/shared_failure_labels.csv` | For each of the 4,406 benchmark images: `k` (number of the 11 architectures that fail it, with ground-truth intrinsics), tier, and the 11 failure flags |

The per-image results of all 36 model evaluations (137 MB, including TUM) and the two TUM-derived diagnostic sets (112 MB) are on Hugging Face (see [Data](#data)).

## Quick start: use the shared-failure labels for your model

```bash
# 1. evaluate your model with eval/run.sh (Section "Run"), which writes results/<set>/<run>/samples/<benchmark>.jsonl
# 2. stratify its errors by the labels
python analysis/stratify_new_model.py results/shared_failure_labels.csv <results_root>/main/<run>/samples
```

It prints, per tier (none / specific / intermediate / shared), the failure rate and the share of your model's mean 1 − δ1 error, as in Table 7 of the paper.

## Reproduce the paper's numbers from the released results

```bash
hf download hljiang/DIMED --repo-type dataset --include "results/*" --local-dir .   # -> results/main, results/tum_family, results/tum_ref
cd analysis
python reproduce_main_numbers.py      # Table 3, failures fixed by scale, shared failures, best-of-18 reference
python export_shared_labels.py ../results/shared_failure_labels.csv
```

Only `numpy` is needed. `RESULTS_DIR=/path/to/results` overrides the location.

## Data

| Item | Where | Size |
|---|---|---|
| MoGe-2 benchmarks (NYUv2, KITTI, ETH3D, iBims-1, DDAD, DIODE, HAMMER) | [`Ruicheng/monocular-geometry-evaluation`](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) | — |
| Per-image results, 36 evaluations + TUM | [`hljiang/DIMED`](https://huggingface.co/datasets/hljiang/DIMED) | 78 MB |
| TUM-Family (514 images), TUM-Ref (215 images) | [`hljiang/DIMED`](https://huggingface.co/datasets/hljiang/DIMED) | 112 MB |

```bash
hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval
cd data/eval && unzip '*.zip' && cd ../..
hf download hljiang/DIMED --repo-type dataset --include "TUM-Family.zip" "TUM-Ref.zip" --local-dir data/extra && cd data/extra
for n in TUM-Family TUM-Ref; do mkdir -p "$n" && unzip -q "$n.zip" -d "$n"; done   # each archive has no top-level folder
cd ../..
```

**TUM sets.** Derived from TUM RGB-D (CC BY 4.0): the Freiburg 3 structure/texture family (8 sequences) and `fr3_long_office_household`. A frame is kept when the camera moved ≥ 0.10 m or rotated ≥ 10° since the last kept frame, with ≥ 500 valid depth pixels. Each sequence is a separate handheld recording. They are diagnostics about which image cues are available, not benchmarks for ranking models.

## Setup

Layout (`<root>` = this repository):

```
<root>/
  eval/  analysis/  results/
  third_party/   upstream model repositories, unmodified
  weights/       checkpoints, one directory per model
  data/eval/     MoGe-2 benchmarks
  data/extra/    TUM-Family, TUM-Ref
```

### Model code (the commits used for the paper)

```bash
mkdir -p third_party weights data && cd third_party
git clone https://github.com/microsoft/MoGe.git                  && git -C MoGe checkout 74fbce0
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git && git -C Depth-Anything-3 checkout 4173623
git clone https://github.com/YvanYin/Metric3D.git                && git -C Metric3D checkout eb5b6fa
git clone https://github.com/metric-anything/metric-anything.git && git -C metric-anything checkout 616a5e6
git clone https://github.com/lpiccinelli-eth/UniDepth.git        && git -C UniDepth checkout 8d8cfe4
git clone https://github.com/lpiccinelli-eth/UniK3D.git          && git -C UniK3D checkout 29f7862
git clone https://github.com/apple/ml-depth-pro.git              && git -C ml-depth-pro checkout 9e65e4d
git clone https://github.com/mx-liu6/FoundationGeo.git           && git -C FoundationGeo checkout 66cf258
git clone https://github.com/hanxunyu/DepthVLM.git               && git -C DepthVLM checkout 5d1472d
cd ..
```

### Weights

```bash
pip install -U huggingface_hub
for r in moge-2-vitl moge-2-vitl-normal moge-2-vitb-normal moge-2-vits-normal moge-3-vitl moge-3-vitg; do
  hf download Ruicheng/$r model.pt --local-dir weights/$r; done
for s in l b s; do
  hf download lpiccinelli/unidepth-v2-vit${s}14 --local-dir weights/unidepth-v2-vit${s}14
  hf download lpiccinelli/unik3d-vit${s}        --local-dir weights/unik3d-vit${s}; done
hf download mxliu-hku/FoundationGeo-1.1 FoundationGeo.pt --local-dir weights/foundationgeo-1.1
hf download JUGGHM/Metric3D metric_depth_vit_small_800k.pth metric_depth_vit_large_800k.pth metric_depth_vit_giant2_800k.pth --local-dir weights/metric3d
hf download apple/DepthPro depth_pro.pt --local-dir weights/depthpro
hf download depth-anything/DA3METRIC-LARGE --local-dir weights/da3metric-large
hf download yjh001/metricanything_student_depthmap student_depthmap.pt --local-dir weights/metricanything-student-depthmap
hf download yjh001/metricanything_student_pointmap student_pointmap.pt --local-dir weights/metricanything-student-pointmap
hf download JonnyYu828/DepthVLM-4B --local-dir weights/depthvlm-4b
```

### Environments

| Env | Models | Python / torch |
|---|---|---|
| default | MoGe-2/3, MetricAnything (both), Metric3D-v2, UniDepthV2, UniK3D, Depth Pro, FoundationGeo | 3.11 / 2.9.1: `pip install -r third_party/MoGe/requirements.txt`, plus each model's requirements |
| `da3` | DA3METRIC | 3.10 / 2.2.2: environment of `third_party/Depth-Anything-3`, plus `utils3d`, `utils3d-moge` |
| `depthvlm` | DepthVLM | 3.12 / 2.5.0, transformers 5.2.0: see `eval/docs/env_depthvlm.md` |

Point `run.sh` at the interpreters with `PY_MOGE`, `PY_DA3`, `PY_DEPTHVLM`.

## Run

```bash
cd eval
bash run.sh MODEL GPU [SET] [both|noK|withK]      # SET = moge2 (default) | tum_family | tum_ref
bash run.sh moge2 0                               # without and with ground-truth intrinsics
bash run.sh metric3d_v2 0                         # needs intrinsics: only the with-K run
```

Models: `moge2 moge3 metricanything_pointmap foundationgeo unidepth_v2 unik3d depth_pro metric3d_v2 metricanything_depthmap da3_metric depthvlm`, and the size variants `moge2_vits moge2_vitb moge2_vitl_normal moge3_vitg unidepth_v2_vits unidepth_v2_vitb unik3d_vits unik3d_vitb metric3d_v2_vits metric3d_v2_vitg`.

Outputs: `results/<set>/<run>[_withK]/results.json` and `samples/<benchmark>.jsonl`. Each line has `filename`, `metric_scale` (`s*` = median of ground truth / prediction over valid pixels, so `s* · prediction ≈ ground truth`; `s* > 1` means depth is underestimated), `depth_metric` and `depth_scale_invariant` (`rel`, `delta1`–`delta3`, `sq_rel`, `rmse`, `rmse_log`, `log10`), `scales` and `sa` (five alignment scales and the scale-aligned `rel` / `delta1` under each, compared in Appendix C), `log_stats` (`log_mse`, `log_mean`, `log_var` of ln prediction − ln ground truth), `far10` and `n_pix`. The record format is described in the Hugging Face dataset card.

### Adding a model

Write an adapter in `eval/baselines/` implementing MoGe's `MGEBaselineInterface` (`infer_for_evaluation(image, intrinsics=None)` returning `depth_metric`), and add a line to `eval/run.sh`. The existing adapters show the four ways intrinsics enter a model (canonical camera space at the output or input, post-hoc focal/shift solve, rays as input).

## Citation

```bibtex
@article{jiang2026averages,
  title   = {What the Averages Hide: Monocular Metric Depth Models Still Fail in Everyday Scenes, Mostly on Scale},
  author  = {Jiang, Hualie and Luo, Yujie and Song, Ziyang and Li, Ximeng and Lou, Zhiqiang and Xu, Rui},
  journal = {arXiv preprint arXiv:2611.20518},
  year    = {2026}
}
```

## Licence

The code in this repository is released under the [MIT License](LICENSE). It covers our evaluation scripts, model adapters and analysis scripts only: each evaluated model's code and weights keep their own licence (several are non-commercial), and the benchmarks keep the licences of their original releases. The TUM-derived sets on Hugging Face follow the TUM RGB-D licence (CC BY 4.0).
