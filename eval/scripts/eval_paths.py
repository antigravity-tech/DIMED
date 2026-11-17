"""Where things live, for the evaluation code in this directory.

Reader layout (everything is a sibling of `eval/`):

    <root>/eval/            this directory (the only thing that is ours)
    <root>/third_party/     MoGe and every other model's official code (git clone them here)
    <root>/weights/         checkpoints we download
    <root>/data/            datasets (`data/eval/...` = MoGe benchmarks, `data/extra/...` = ours)
    <root>/results/         evaluation outputs (results.json, per-sample jsonl, logs)

`MOGE_ROOT`, `DATA_ROOT` and `RESULTS_ROOT` can be overridden with environment variables of the same name.
"""
import os
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
ROOT = EVAL_ROOT.parent
THIRD_PARTY = ROOT / 'third_party'
MOGE_ROOT = Path(os.environ.get('MOGE_ROOT', THIRD_PARTY / 'MoGe'))
DATA_ROOT = Path(os.environ.get('DATA_ROOT', ROOT / 'data'))
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', ROOT / 'results'))


def add_moge_to_path() -> None:
    """Make `import moge` resolve to the unmodified upstream checkout."""
    import sys
    if not (MOGE_ROOT / 'moge').is_dir():
        raise FileNotFoundError(f'Cannot find MoGe at {MOGE_ROOT}. Run: git clone https://github.com/microsoft/MoGe {THIRD_PARTY / "MoGe"}')
    if str(MOGE_ROOT) not in sys.path:
        sys.path.insert(0, str(MOGE_ROOT))


def resolve_data_path(path) -> str:
    """Benchmark `path` entries are relative to DATA_ROOT (e.g. `eval/NYUv2`); absolute paths are kept."""
    p = Path(path)
    return str(p if p.is_absolute() else (DATA_ROOT / p))
