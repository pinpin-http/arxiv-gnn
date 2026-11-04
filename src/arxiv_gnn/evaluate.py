"""Évaluation : l'`Evaluator` officiel d'OGB et l'agrégation sur les graines."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from .config import DATASET


@lru_cache(maxsize=1)
def _evaluator():
    from ogb.nodeproppred import Evaluator

    return Evaluator(DATASET)


def accuracy(pred: Tensor, y: Tensor, idx: Tensor) -> float:
    """Accuracy calculée par l'Evaluator OGB sur les nœuds `idx`."""
    y_true = y[idx].view(-1, 1).cpu()
    y_pred = pred[idx].view(-1, 1).cpu()
    return float(_evaluator().eval({"y_true": y_true, "y_pred": y_pred})["acc"])


def split_accuracies(pred: Tensor, y: Tensor, split: dict[str, Tensor], names=("train", "valid", "test")) -> dict:
    return {name: accuracy(pred, y, split[name]) for name in names}


def mean_std(values) -> tuple[float, float]:
    """Moyenne et écart-type non biaisé (comme `torch.std`, exigé par OGB)."""
    t = torch.tensor(list(values), dtype=torch.float64)
    if t.numel() == 0:
        return math.nan, math.nan
    std = float(t.std()) if t.numel() > 1 else 0.0
    return float(t.mean()), std
