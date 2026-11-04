"""Chemins, graines et budget d'hyperparamètres communs à tous les modèles."""

from __future__ import annotations

import itertools
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ARXIV_GNN_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("ARXIV_GNN_RESULTS", ROOT / "results"))
CHECKPOINT_DIR = Path(os.environ.get("ARXIV_GNN_CHECKPOINTS", ROOT / "checkpoints"))
DOCS_DIR = ROOT / "docs"

DATASET = "ogbn-arxiv"

# OGB demande 10 graines (0 à 9) pour son classement. Sur 2 threads CPU, une
# époque plein graphe prend de 3 à 7 s : on s'en tient aux graines 0, 1 et 2
# (moyenne et écart-type non biaisé, comme OGB).
SEEDS = (0, 1, 2)
SEARCH_SEED = 0

# Validation (et test en campagne finale) évalués toutes les 5 époques.
EVAL_EVERY = 5

# La machine est partagée : deux threads CPU au plus.
NUM_THREADS = int(os.environ.get("ARXIV_GNN_THREADS", "2"))

NEURAL_MODELS = ("mlp", "gcn", "sage", "gat")
PROPAGATION_MODELS = ("lp", "cs")
ALL_MODELS = NEURAL_MODELS + PROPAGATION_MODELS
# Méthodes entraînées et rapportées. GAT est implémenté et testé, mais une
# époque plein graphe prend environ 30 s sur 2 threads : il n'est que chronométré.
RESULT_MODELS = ("mlp", "lp", "gcn", "sage", "cs")
BENCH_MODELS = ("mlp", "gcn", "sage", "gat", "lp", "cs")

MODEL_LABELS = {
    "mlp": "MLP (sans graphe)",
    "lp": "Propagation d'étiquettes",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "gat": "GAT",
    "cs": "MLP + C&S",
}


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparamètres d'un modèle entraîné par descente de gradient."""

    hidden: int = 256
    num_layers: int = 3
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 0.0
    epochs: int = 300
    heads: int = 4  # GAT uniquement : la largeur `hidden` est répartie entre les têtes
    checkpoint: bool = False  # recalcul des convolutions à la rétropropagation (mémoire)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class LPConfig:
    num_layers: int = 50
    alpha: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CSConfig:
    correct_alpha: float = 0.8
    smooth_alpha: float = 0.8
    num_layers: int = 50
    autoscale: bool = True
    scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# Même budget pour tous : 4 configurations par méthode, jugées sur la validation
# (graine 0). Les réseaux gardent l'architecture des exemples officiels OGB
# (3 couches, BatchNorm, Adam, lr 0,01), avec 300 époques au lieu de 500.
NEURAL_GRID = {
    "hidden": (128, 256),
    "dropout": (0.25, 0.5),
}
LP_GRID = {
    "alpha": (0.5, 0.75, 0.9, 0.99),
}
CS_GRID = {
    "correct_alpha": (0.5, 0.9),
    "smooth_alpha": (0.6, 0.9),
}


def _expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def neural_grid(base: TrainConfig | None = None) -> list[TrainConfig]:
    base = base or TrainConfig()
    return [replace(base, **p) for p in _expand(NEURAL_GRID)]


def lp_grid() -> list[LPConfig]:
    return [LPConfig(**p) for p in _expand(LP_GRID)]


def cs_grid() -> list[CSConfig]:
    return [CSConfig(**p) for p in _expand(CS_GRID)]
