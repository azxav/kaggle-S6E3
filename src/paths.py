"""Local path resolution for this repository.

Training scripts were written for a flat Kaggle working directory. When they
are launched from this checkout, these helpers point them at ``data/`` and
``outputs/`` without discarding an explicit path that already exists.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

_KAGGLE_TRAIN = Path("/kaggle/input/competitions/playground-series-s6e3/train.csv")
_KAGGLE_TEST = Path("/kaggle/input/competitions/playground-series-s6e3/test.csv")
_ORIGINAL_NAMES = (
    "orig-Telco-Customer-Churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "original.csv",
)
_KAGGLE_ORIGINALS = (
    Path(
        "/kaggle/input/datasets/azizbekxasanov/telco-customer-churn/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ),
    Path(
        "/kaggle/input/datasets/blastchar/telco-customer-churn/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ),
)


def resolve_data_file(configured: Path | str, filename: str) -> Path:
    """Return the first existing path among the configured path and local data."""
    configured_path = Path(configured)
    candidates = [
        configured_path,
        DATA_DIR / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]
    if filename == "train.csv":
        candidates.append(_KAGGLE_TRAIN)
    elif filename == "test.csv":
        candidates.append(_KAGGLE_TEST)
    for path in candidates:
        if path.exists():
            return path
    return configured_path


def resolve_original(configured: Path | str | None = None) -> Path | None:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(Path(configured))
    for name in _ORIGINAL_NAMES:
        candidates.append(DATA_DIR / name)
        candidates.append(Path.cwd() / name)
        candidates.append(Path.cwd() / "data" / name)
    candidates.extend(_KAGGLE_ORIGINALS)
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_output_dir(configured: Path | str) -> Path:
    """Map Kaggle ``/kaggle/working`` outputs onto ``outputs/`` in this repo."""
    path = Path(configured)
    if path.is_absolute():
        if path.exists() or not str(path).startswith("/kaggle/"):
            return path
        marker = "/kaggle/working/"
        text = str(path)
        if text.startswith(marker):
            tail = text[len(marker) :].strip("/")
            if not tail:
                return OUTPUTS_DIR
            if tail.startswith("outputs/"):
                return REPO_ROOT / tail
            return OUTPUTS_DIR / tail
        return OUTPUTS_DIR / path.name
    return (REPO_ROOT / path).resolve()


def prepare_local_paths(config):
    """Rewrite dataset and output attributes on a config object, in place."""
    if hasattr(config, "train_path"):
        config.train_path = resolve_data_file(config.train_path, "train.csv")
    if hasattr(config, "test_path"):
        config.test_path = resolve_data_file(config.test_path, "test.csv")
    if hasattr(config, "train_csv"):
        config.train_csv = resolve_data_file(config.train_csv, "train.csv")
    if hasattr(config, "test_csv"):
        config.test_csv = resolve_data_file(config.test_csv, "test.csv")
    if hasattr(config, "original_path"):
        resolved = resolve_original(getattr(config, "original_path"))
        if resolved is not None:
            config.original_path = resolved
    if hasattr(config, "orig_path"):
        resolved = resolve_original(config.orig_path)
        if resolved is not None:
            config.orig_path = resolved
    if hasattr(config, "output_dir"):
        config.output_dir = resolve_output_dir(config.output_dir)
    return config


def map_churn(series):
    """Map Yes/No or 0/1 churn labels to a float array."""
    import pandas as pd

    values = pd.Series(series)
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=float)
    mapped = values.astype(str).str.strip().str.lower().map(
        {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Unsupported Churn labels: {bad}")
    return mapped.to_numpy(dtype=float)
