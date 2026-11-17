#!/usr/bin/env bash
# Run one evaluation (one model, on one benchmark set).  Always run from anywhere; paths are resolved from this file.
#
#   bash run.sh MODEL GPU [SET] [both|noK|withK]
#
#   MODEL  one of the names in the table below
#   GPU    CUDA device index
#   SET    moge2 (default; the 7 MoGe-2 benchmarks) | tum_family | tum_ref
#   mode   both (default): without and with ground-truth intrinsics; models that need intrinsics are always "withK"
#
# Python interpreters: set PY_MOGE (default `python`), PY_DA3, PY_DEPTHVLM if those models need their own environments.
# Results: $RESULTS_ROOT/<SET dir>/<run>[_withK]/{results.json,samples/*.jsonl} and <run>[_withK].log next to them
#   (scripts/eval_depth.py; the alignment scale s* is the median of gt / pred, see that file).
#   RESULTS_ROOT defaults to <root>/results (the directory above eval/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RESULTS_ROOT=${RESULTS_ROOT:-$(cd .. && pwd)/results}

MODEL=${1:?usage: run.sh MODEL GPU [SET] [both|noK|withK]}; GPU=${2:?}; SET=${3:-moge2}; MODE=${4:-both}
PY_MOGE=${PY_MOGE:-python}; PY_DA3=${PY_DA3:-$PY_MOGE}; PY_DEPTHVLM=${PY_DEPTHVLM:-$PY_MOGE}

case "$SET" in
  moge2)      CFG=configs/metric_depth_only.json; OUT=$RESULTS_ROOT/main ;;
  tum_family) CFG=configs/tum_family.json;        OUT=$RESULTS_ROOT/tum_family ;;
  tum_ref)    CFG=configs/tum_ref.json;           OUT=$RESULTS_ROOT/tum_ref ;;
  *) echo "unknown SET: $SET" >&2; exit 2 ;;
esac

# MODEL -> adapter, run name, interpreter, intrinsics (needs = model cannot run without K), extra args
NEEDS=0; ARGS=()
case "$MODEL" in
  moge2)                   B=moge;                    NAME=moge2;                   PY=$PY_MOGE; ARGS=(--version v2 --pretrained ../weights/moge-2-vitl/model.pt) ;;
  moge3)                   B=moge;                    NAME=moge3;                   PY=$PY_MOGE; ARGS=(--version v3 --pretrained ../weights/moge-3-vitl/model.pt) ;;
  moge2_vitl_normal)         B=moge;                    NAME=moge2_vitl_normal;         PY=$PY_MOGE; ARGS=(--version v2 --pretrained ../weights/moge-2-vitl-normal/model.pt) ;;
  moge2_vitb)              B=moge;                   NAME=moge2_vitb;              PY=$PY_MOGE; ARGS=(--version v2 --pretrained ../weights/moge-2-vitb-normal/model.pt) ;;
  moge2_vits)              B=moge;                    NAME=moge2_vits;              PY=$PY_MOGE; ARGS=(--version v2 --pretrained ../weights/moge-2-vits-normal/model.pt) ;;
  moge3_vitg)              B=moge;                    NAME=moge3_vitg;              PY=$PY_MOGE; ARGS=(--version v3 --pretrained ../weights/moge-3-vitg/model.pt) ;;
  unidepth_v2_vitb)        B=unidepth_v2;             NAME=unidepth_v2_vitb;        PY=$PY_MOGE; ARGS=(--pretrained ../weights/unidepth-v2-vitb14) ;;
  unidepth_v2_vits)        B=unidepth_v2;             NAME=unidepth_v2_vits;        PY=$PY_MOGE; ARGS=(--pretrained ../weights/unidepth-v2-vits14) ;;
  unik3d_vitb)             B=unik3d;                  NAME=unik3d_vitb;             PY=$PY_MOGE; ARGS=(--pretrained ../weights/unik3d-vitb) ;;
  unik3d_vits)             B=unik3d;                  NAME=unik3d_vits;             PY=$PY_MOGE; ARGS=(--pretrained ../weights/unik3d-vits) ;;
  metric3d_v2_vits)        B=metric3d_v2;             NAME=metric3d_v2_vits;        PY=$PY_MOGE; NEEDS=1; ARGS=(--backbone vits --pretrained ../weights/metric3d/metric_depth_vit_small_800k.pth) ;;
  metricanything_pointmap) B=metricanything_pointmap; NAME=metricanything_pointmap; PY=$PY_MOGE ;;
  foundationgeo)           B=foundationgeo;           NAME=foundationgeo;           PY=$PY_MOGE ;;
  unidepth_v2)             B=unidepth_v2;             NAME=unidepth_v2;             PY=$PY_MOGE ;;
  unik3d)                  B=unik3d;                  NAME=unik3d;                  PY=$PY_MOGE ;;
  depth_pro)               B=depth_pro;               NAME=depth_pro;               PY=$PY_MOGE ;;
  metric3d_v2)             B=metric3d_v2;             NAME=metric3d_v2;             PY=$PY_MOGE; NEEDS=1; ARGS=(--backbone vitl --pretrained ../weights/metric3d/metric_depth_vit_large_800k.pth) ;;
  metric3d_v2_vitg)        B=metric3d_v2;             NAME=metric3d_v2_vitg;        PY=$PY_MOGE; NEEDS=1; ARGS=(--backbone vitg --pretrained ../weights/metric3d/metric_depth_vit_giant2_800k.pth) ;;
  metricanything_depthmap) B=metricanything_depthmap; NAME=metricanything_depthmap; PY=$PY_MOGE; NEEDS=1 ;;
  da3_metric)              B=da3_metric;              NAME=da3_metric;              PY=$PY_DA3;  NEEDS=1 ;;
  depthvlm)                B=depthvlm;                NAME=depthvlm;                PY=$PY_DEPTHVLM; NEEDS=1 ;;
  *) echo "unknown MODEL: $MODEL" >&2; exit 2 ;;
esac
[ "$NEEDS" = 1 ] && MODE=withK

mkdir -p "$OUT"
if [ "$MODE" != withK ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_depth.py --baseline baselines/$B.py --config $CFG \
    --output_dir "$OUT/$NAME" "${ARGS[@]}" > "$OUT/$NAME.log" 2>&1
fi
if [ "$MODE" != noK ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_depth.py --with_intrinsics --baseline baselines/$B.py --config $CFG \
    --output_dir "$OUT/${NAME}_withK" "${ARGS[@]}" > "$OUT/${NAME}_withK.log" 2>&1
fi
echo "done: $MODEL on $SET -> $OUT"
