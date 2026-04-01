from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TorchFrameRuntimeConfig:
    seed: int
    channels: int = 32
    num_layers: int = 3
    num_heads: int = 8
    batch_size: int = 1024
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    use_amp: bool = True
    num_workers: int = 0
    excel_in_channels: int = 32
    diam_dropout: float = 0.0
    aium_dropout: float = 0.0
    residual_dropout: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except Exception:
        pass


def import_torch_frame():
    try:
        import torch
        import torch_frame
        from torch_frame import TaskType, stype
        from torch_frame.data import DataLoader, Dataset
        from torch_frame.transforms import CatToNumTransform, MutualInformationSort
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "Torch Frame trainers require torch and pytorch-frame. "
            "Install them in the target environment, for example: pip install torch pytorch-frame"
        ) from exc
    try:
        from torch_frame.nn.models import ExcelFormer, FTTransformer
    except Exception:
        try:
            from torch_frame.nn.models.excelformer import ExcelFormer  # type: ignore
        except Exception:
            from torch_frame.nn.models.excel_former import ExcelFormer  # type: ignore
        from torch_frame.nn.models.ft_transformer import FTTransformer  # type: ignore
    return {
        "torch": torch,
        "torch_frame": torch_frame,
        "TaskType": TaskType,
        "stype": stype,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "ExcelFormer": ExcelFormer,
        "FTTransformer": FTTransformer,
        "CatToNumTransform": CatToNumTransform,
        "MutualInformationSort": MutualInformationSort,
    }


def resolve_device(device: str | None = None):
    torch = import_torch_frame()["torch"]
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_col_to_stype(df: pd.DataFrame):
    deps = import_torch_frame()
    stype = deps["stype"]
    col_to_stype: dict[str, Any] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_to_stype[col] = stype.numerical
        else:
            col_to_stype[col] = stype.categorical
    return col_to_stype


def materialize_split_datasets(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
):
    deps = import_torch_frame()
    Dataset = deps["Dataset"]
    stype = deps["stype"]

    target_col = "__target__"
    train_df = X_train.copy()
    train_df[target_col] = y_train.astype(np.float32)
    valid_df = X_valid.copy()
    valid_df[target_col] = y_valid.astype(np.float32)
    test_df = X_test.copy()

    feat_col_to_stype = infer_col_to_stype(X_train)
    # torch_frame's feat_cols property calls list.remove(target_col) on col_to_stype keys,
    # so target_col must be present in col_to_stype for train/valid datasets.
    train_col_to_stype = {**feat_col_to_stype, target_col: stype.numerical}

    train_dataset = Dataset(train_df, col_to_stype=train_col_to_stype, target_col=target_col).materialize()
    valid_dataset = Dataset(valid_df, col_to_stype=train_col_to_stype, target_col=target_col).materialize(
        col_stats=train_dataset.col_stats
    )
    test_dataset = Dataset(test_df, col_to_stype=feat_col_to_stype, target_col=None).materialize(
        col_stats=train_dataset.col_stats
    )
    return train_dataset, valid_dataset, test_dataset


def build_fttransformer_inputs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
):
    return materialize_split_datasets(X_train, y_train, X_valid, y_valid, X_test)


def build_excelformer_inputs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
):
    deps = import_torch_frame()
    TaskType = deps["TaskType"]
    stype = deps["stype"]
    CatToNumTransform = deps["CatToNumTransform"]
    MutualInformationSort = deps["MutualInformationSort"]

    train_dataset, valid_dataset, test_dataset = materialize_split_datasets(
        X_train, y_train, X_valid, y_valid, X_test
    )

    cat_to_num = CatToNumTransform()
    cat_to_num.fit(train_dataset.tensor_frame, train_dataset.col_stats)
    train_tf = cat_to_num(train_dataset.tensor_frame)
    valid_tf = cat_to_num(valid_dataset.tensor_frame)
    # CatToNumTransform._forward calls torch.is_floating_point(tf.y) without
    # guarding against tf.y being None (test has no target). Temporarily attach
    # a dummy y so the transform runs, then clear it from the result.
    torch = deps["torch"]
    _test_raw = test_dataset.tensor_frame
    _test_raw.y = torch.zeros(len(_test_raw), dtype=torch.float32)
    test_tf = cat_to_num(_test_raw)
    _test_raw.y = None
    test_tf.y = None

    transformed_stats = dict(cat_to_num.transformed_stats)
    if stype.numerical in train_dataset.tensor_frame.col_names_dict:
        for col in train_dataset.tensor_frame.col_names_dict[stype.numerical]:
            transformed_stats.setdefault(col, train_dataset.col_stats[col])

    mi_sort = MutualInformationSort(task_type=TaskType.BINARY_CLASSIFICATION)
    mi_sort.fit(train_tf, transformed_stats)
    train_tf = mi_sort(train_tf)
    valid_tf = mi_sort(valid_tf)
    test_tf = mi_sort(test_tf)
    return train_tf, valid_tf, test_tf, dict(mi_sort.transformed_stats)


def build_fttransformer_model(train_dataset, cfg: TorchFrameRuntimeConfig):
    deps = import_torch_frame()
    FTTransformer = deps["FTTransformer"]
    return FTTransformer(
        channels=cfg.channels,
        out_channels=1,
        num_layers=cfg.num_layers,
        col_stats=train_dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
    )


def build_excelformer_model(train_tf, transformed_stats: dict[str, dict[str, Any]], cfg: TorchFrameRuntimeConfig):
    deps = import_torch_frame()
    ExcelFormer = deps["ExcelFormer"]
    stype = deps["stype"]
    num_cols = len(train_tf.col_names_dict.get(stype.numerical, []))
    return ExcelFormer(
        in_channels=cfg.excel_in_channels,
        out_channels=1,
        num_cols=num_cols,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        col_stats=transformed_stats,
        col_names_dict=train_tf.col_names_dict,
        diam_dropout=cfg.diam_dropout,
        aium_dropout=cfg.aium_dropout,
        residual_dropout=cfg.residual_dropout,
        mixup=None,
    )


def _loader(dataset_or_tf, batch_size: int, shuffle: bool, num_workers: int):
    deps = import_torch_frame()
    DataLoader = deps["DataLoader"]
    return DataLoader(
        dataset_or_tf,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _predict_proba(model, dataset_or_tf, batch_size: int, device, num_workers: int) -> np.ndarray:
    deps = import_torch_frame()
    torch = deps["torch"]
    loader = _loader(dataset_or_tf, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def train_binary_model(
    model,
    train_dataset_or_tf,
    valid_dataset_or_tf,
    test_dataset_or_tf,
    cfg: TorchFrameRuntimeConfig,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    deps = import_torch_frame()
    torch = deps["torch"]
    model = model.to(device)

    train_loader = _loader(
        train_dataset_or_tf,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    amp_enabled = bool(cfg.use_amp and device.type == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience

    from sklearn.metrics import roc_auc_score

    for _epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch).view(-1)
                y_true = batch.y.view(-1).float()
                loss = criterion(logits, y_true)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        valid_pred = _predict_proba(
            model,
            valid_dataset_or_tf,
            batch_size=cfg.batch_size,
            device=device,
            num_workers=cfg.num_workers,
        )
        y_valid = (
            valid_dataset_or_tf.y.detach().cpu().numpy()
            if hasattr(valid_dataset_or_tf, "y") and valid_dataset_or_tf.y is not None
            else valid_dataset_or_tf.tensor_frame.y.detach().cpu().numpy()
        )
        val_auc = float(roc_auc_score(y_valid, valid_pred))
        if val_auc > best_score + 1e-6:
            best_score = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    valid_pred = _predict_proba(
        model,
        valid_dataset_or_tf,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
    )
    test_pred = _predict_proba(
        model,
        test_dataset_or_tf,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
    )
    return valid_pred, test_pred
