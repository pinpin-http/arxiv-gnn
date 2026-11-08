"""Entraînement plein graphe, recherche d'hyperparamètres et campagnes multi-graines.

Règles suivies (celles d'OGB) :
- les étiquettes d'entraînement sont les seules à entrer dans la perte, dans la
  propagation d'étiquettes et dans C&S ;
- la validation sert à choisir l'époque et la configuration ;
- le test n'est jamais calculé pendant la recherche ; pendant les campagnes
  finales, on rapporte le test de l'époque qui maximise la validation.

Chaque essai et chaque graine sont écrits sur disque dès qu'ils finissent :
une campagne interrompue reprend là où elle s'est arrêtée.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import config as C
from .config import CSConfig, LPConfig, TrainConfig
from .data import NodeData
from .evaluate import mean_std, split_accuracies
from .models import build_model, correct_and_smooth, count_parameters, label_propagation
from .thermal import throttle_events

EMBED_SAMPLE = 6000  # nœuds de test projetés en 2D pour la figure t-SNE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_threads(n: int = C.NUM_THREADS) -> None:
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:  # déjà fixé dans ce processus
        pass


def get_device(name: str = "cpu") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare(data: NodeData, name: str, device: torch.device, amp: bool | None = None) -> NodeData:
    """Ne garde que la structure de graphe utile au modèle et la place sur `device`.
    Sur GPU, la précision mixte (float16) est activée par défaut."""
    amp = (device.type == "cuda") if amp is None else amp
    return data.for_model(name).to(device, half=amp and device.type == "cuda")


def train_step(model, optimizer, scaler, data: NodeData, amp: bool) -> Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    idx = data.split["train"]
    with torch.autocast(data.x.device.type, dtype=torch.float16, enabled=amp):
        out = model(data.x, data.graph)
        loss = F.cross_entropy(out[idx].float(), data.y[idx])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


@torch.no_grad()
def predict(model, data: NodeData, amp: bool, return_hidden: bool = False):
    model.eval()
    with torch.autocast(data.x.device.type, dtype=torch.float16, enabled=amp):
        res = model(data.x, data.graph, return_hidden=return_hidden)
    if return_hidden:
        return res[0].float(), res[1].float()
    return res.float()


@dataclass
class RunResult:
    model: str
    seed: int
    config: dict
    params: int
    best_epoch: int
    train_acc: float
    valid_acc: float
    test_acc: float | None
    seconds: float
    epoch_s: float  # médiane du temps d'une étape d'entraînement (une époque)
    max_memory_mb: float | None  # VRAM maximale si GPU
    throttled: int = 0  # épisodes de ralentissement thermique pendant le run
    device: str = "cpu"
    precision: str = "fp32"  # fp16 : précision mixte (torch.amp)
    history: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_training(
    name: str,
    cfg: TrainConfig,
    data: NodeData,
    seed: int,
    amp: bool | None = None,
    eval_test: bool = True,
    eval_every: int = C.EVAL_EVERY,
    log_every: int = 0,
):
    """Entraîne un modèle et garde l'état de la meilleure évaluation en validation.

    La validation (et le test si `eval_test`) est évaluée toutes les
    `eval_every` époques et à la dernière. Avec `eval_test=False` (recherche
    d'hyperparamètres), l'accuracy test n'est jamais calculée.
    Renvoie (RunResult, état du meilleur modèle).
    """
    device = data.x.device
    # Par défaut, la précision suit celle des données préparées par `prepare`.
    amp = (data.x.dtype == torch.float16) if amp is None else amp
    if not amp and data.x.dtype != torch.float32:
        raise ValueError("données en float16 alors que la précision mixte est désactivée")
    set_seed(seed)
    model = build_model(name, data.x.size(1), data.num_classes, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    splits = ("train", "valid", "test") if eval_test else ("train", "valid")
    history = {"loss": [], "epoch": [], **{s: [] for s in splits}}
    best = {"valid": -1.0}
    best_state = None
    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start, wall_start = time.perf_counter(), time.time()
    for epoch in range(1, cfg.epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        loss = train_step(model, optimizer, scaler, data, amp)
        sync(device)
        step_times.append(time.perf_counter() - t0)
        history["loss"].append(float(loss))
        if epoch % eval_every and epoch != cfg.epochs:
            continue
        pred = predict(model, data, amp).argmax(dim=-1)
        accs = split_accuracies(pred, data.y, data.split, splits)
        history["epoch"].append(epoch)
        for s in splits:
            history[s].append(accs[s])
        if accs["valid"] > best["valid"]:
            best = {"epoch": epoch, **accs}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log_every and epoch % log_every == 0:
            msg = " ".join(f"{s}={accs[s]:.4f}" for s in splits)
            print(f"  [{name} seed={seed}] époque {epoch:4d} perte={float(loss):.4f} {msg}", flush=True)
    seconds = time.perf_counter() - t_start
    max_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = RunResult(
        model=name,
        seed=seed,
        config=cfg.to_dict(),
        params=count_parameters(model),
        best_epoch=best["epoch"],
        train_acc=best["train"],
        valid_acc=best["valid"],
        test_acc=best.get("test"),
        seconds=seconds,
        epoch_s=float(np.median(step_times)),
        max_memory_mb=max_mem,
        throttled=len(throttle_events(wall_start, time.time())),
        device=device.type,
        precision="fp16" if amp else "fp32",
        history=history,
    )
    return result, best_state


def restore(name: str, cfg: TrainConfig, data: NodeData, state: dict):
    model = build_model(name, data.x.size(1), data.num_classes, cfg).to(data.x.device)
    model.load_state_dict(state)
    return model


# --------------------------------------------------------------------------
# Persistance
# --------------------------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)  # écriture atomique : un arrêt brutal ne laisse pas de JSON tronqué


def _read_json(path: Path):
    return json.loads(path.read_text())


def _save_tensor(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def search_path(name: str) -> Path:
    return C.RESULTS_DIR / "search" / f"{name}.json"


def trial_path(name: str, i: int) -> Path:
    return C.RESULTS_DIR / "search" / name / f"trial{i}.json"


def trial_state_path(name: str, i: int) -> Path:
    return C.CHECKPOINT_DIR / "search" / f"{name}_trial{i}.pt"


def final_path(name: str) -> Path:
    return C.RESULTS_DIR / "final" / f"{name}.json"


def seed_path(name: str, seed: int) -> Path:
    return C.RESULTS_DIR / "final" / name / f"seed{seed}.json"


def preds_path(name: str, seed: int) -> Path:
    return C.CHECKPOINT_DIR / "preds" / f"{name}_seed{seed}.pt"


def embed_path(name: str) -> Path:
    return C.CHECKPOINT_DIR / "embeddings" / f"{name}_seed{C.SEARCH_SEED}.pt"


def best_config(name: str):
    """Configuration retenue sur la validation, ou None si la recherche n'a pas tourné."""
    path = search_path(name)
    if not path.exists():
        return None
    best = _read_json(path)["best"]["config"]
    if name == "lp":
        return LPConfig(**best)
    if name == "cs":
        return CSConfig(**best)
    return TrainConfig.from_dict(best)


def embedding_sample(data: NodeData, size: int = EMBED_SAMPLE) -> Tensor:
    """Sous-échantillon fixe de nœuds de test, le même pour tous les modèles."""
    gen = torch.Generator().manual_seed(1234)
    test = data.split["test"].cpu()
    return test[torch.randperm(test.numel(), generator=gen)[:size]].sort().values


# --------------------------------------------------------------------------
# Réseaux : recherche puis campagne finale
# --------------------------------------------------------------------------

def search_neural(name: str, data: NodeData, configs=None, epochs: int | None = None, log=print) -> dict:
    configs = configs or C.neural_grid()
    if epochs is not None:
        configs = [TrainConfig.from_dict({**c.to_dict(), "epochs": epochs}) for c in configs]
    trials = []
    for i, cfg in enumerate(configs):
        path = trial_path(name, i)
        if path.exists() and _read_json(path)["config"] == cfg.to_dict():
            trials.append(_read_json(path))
            log(f"[search {name}] essai {i + 1}/{len(configs)} déjà fait : valid={trials[-1]['valid_acc']:.4f}")
            continue
        res, state = run_training(name, cfg, data, C.SEARCH_SEED, eval_test=False)
        trial = res.to_dict()
        trial["trial"] = i
        _save_tensor(state, trial_state_path(name, i))
        _write_json(path, trial)
        trials.append(trial)
        log(f"[search {name}] essai {i + 1}/{len(configs)} {cfg.to_dict()} -> valid={res.valid_acc:.4f} "
            f"(époque {res.best_epoch}, {res.seconds / 60:.1f} min)")
    best = max(trials, key=lambda t: t["valid_acc"])
    summary = [{k: t[k] for k in ("trial", "config", "valid_acc", "train_acc", "best_epoch", "seconds", "epoch_s")}
               for t in trials]
    out = {"model": name, "seed": C.SEARCH_SEED, "selection": "valid_acc", "trials": summary,
           "best": {k: best[k] for k in ("trial", "config", "valid_acc", "best_epoch")}}
    _write_json(search_path(name), out)
    return out


def _reuse_search_run(name: str, cfg: TrainConfig, data: NodeData):
    """L'essai retenu par la recherche a été entraîné avec la graine 0 et la même
    configuration : on évalue son meilleur état sur le test au lieu de le
    réentraîner (sur CPU, le calcul est déterministe : même résultat)."""
    search = _read_json(search_path(name)) if search_path(name).exists() else None
    if search is None or search["best"]["config"] != cfg.to_dict():
        return None
    i = search["best"]["trial"]
    trial, state_file = _read_json(trial_path(name, i)), trial_state_path(name, i)
    if not state_file.exists():
        return None
    model = restore(name, cfg, data, torch.load(state_file))
    pred = predict(model, data, amp=data.x.dtype != torch.float32).argmax(dim=-1)
    accs = split_accuracies(pred, data.y, data.split)
    if abs(accs["valid"] - trial["valid_acc"]) > 1e-9:  # calcul non déterministe (GPU) : on réentraîne
        return None
    trial = {**trial, "test_acc": accs["test"], "reused_from_search": True}
    trial.pop("trial", None)
    return trial, model


def final_neural(name: str, data: NodeData, cfg: TrainConfig | None = None, seeds=C.SEEDS, log=print) -> dict:
    cfg = cfg or best_config(name) or TrainConfig()
    amp = data.x.dtype != torch.float32
    sample = embedding_sample(data).to(data.x.device)
    runs = []
    for seed in seeds:
        path = seed_path(name, seed)
        if path.exists() and _read_json(path)["config"] == cfg.to_dict():
            runs.append(_read_json(path))
            log(f"[final {name}] graine {seed} déjà faite : test={runs[-1]['test_acc']:.4f}")
            continue
        reused = _reuse_search_run(name, cfg, data) if seed == C.SEARCH_SEED else None
        if reused is not None:
            run, model = reused
        else:
            res, state = run_training(name, cfg, data, seed)
            run, model = res.to_dict(), restore(name, cfg, data, state)
        logits, hidden = predict(model, data, amp, return_hidden=True)
        if name == "mlp":
            _save_tensor(logits.softmax(dim=-1).half().cpu(), preds_path(name, seed))
        if seed == C.SEARCH_SEED:
            _save_tensor({"nodes": sample.cpu(), "hidden": hidden[sample].half().cpu()}, embed_path(name))
        _write_json(path, run)
        runs.append(run)
        log(f"[final {name}] graine {seed}: valid={run['valid_acc']:.4f} test={run['test_acc']:.4f} "
            f"(époque {run['best_epoch']}{', run de la recherche' if reused else ''})")
    out = _summarize(name, cfg.to_dict(), runs)
    _write_json(final_path(name), out)
    return out


def _summarize(name: str, cfg: dict, runs: list[dict]) -> dict:
    v_mean, v_std = mean_std(r["valid_acc"] for r in runs)
    t_mean, t_std = mean_std(r["test_acc"] for r in runs)
    return {
        "model": name,
        "config": cfg,
        "seeds": [r["seed"] for r in runs],
        "valid_mean": v_mean,
        "valid_std": v_std,
        "test_mean": t_mean,
        "test_std": t_std,
        "params": runs[0].get("params", 0),
        "runs": runs,
    }


# --------------------------------------------------------------------------
# Propagation d'étiquettes et Correct & Smooth
# --------------------------------------------------------------------------

def _lp_accuracies(data: NodeData, cfg: LPConfig, splits) -> dict:
    out = label_propagation(data.graph.adj_lp, data.y, data.split["train"], data.num_classes, cfg)
    return split_accuracies(out.argmax(dim=-1), data.y, data.split, splits)


def search_lp(data: NodeData, log=print) -> dict:
    trials = []
    for i, cfg in enumerate(C.lp_grid()):
        t0 = time.perf_counter()
        accs = _lp_accuracies(data, cfg, ("train", "valid"))
        trials.append({"trial": i, "config": cfg.to_dict(), "valid_acc": accs["valid"],
                       "seconds": time.perf_counter() - t0})
        log(f"[search lp] {cfg.to_dict()} -> valid={accs['valid']:.4f}")
    best = max(trials, key=lambda t: t["valid_acc"])
    out = {"model": "lp", "selection": "valid_acc", "trials": trials, "best": best}
    _write_json(search_path("lp"), out)
    return out


def final_lp(data: NodeData, seeds=C.SEEDS, log=print) -> dict:
    """La propagation est déterministe : un seul calcul, écart-type nul entre
    graines (comme pour l'entrée « Label Propagation » du classement OGB)."""
    cfg = best_config("lp") or LPConfig()
    t0 = time.perf_counter()
    accs = _lp_accuracies(data, cfg, ("train", "valid", "test"))
    seconds = time.perf_counter() - t0
    runs = [{"seed": s, "train_acc": accs["train"], "valid_acc": accs["valid"], "test_acc": accs["test"],
             "seconds": seconds, "params": 0} for s in seeds]
    out = _summarize("lp", cfg.to_dict(), runs)
    log(f"[final lp] valid={out['valid_mean']:.4f} test={out['test_mean']:.4f} ({seconds:.1f} s)")
    _write_json(final_path("lp"), out)
    return out


def _load_mlp_probs(seed: int, device) -> Tensor:
    path = preds_path("mlp", seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} manquant : lancer d'abord `make train MODEL=mlp`")
    return torch.load(path).float().to(device)


def _cs_accuracies(data: NodeData, probs: Tensor, cfg: CSConfig, splits) -> dict:
    out = correct_and_smooth(probs, data.y, data.split["train"], data.graph, cfg)
    return split_accuracies(out.argmax(dim=-1), data.y, data.split, splits)


def search_cs(data: NodeData, log=print) -> dict:
    probs = _load_mlp_probs(C.SEARCH_SEED, data.x.device)
    trials = []
    for i, cfg in enumerate(C.cs_grid()):
        t0 = time.perf_counter()
        accs = _cs_accuracies(data, probs, cfg, ("train", "valid"))
        trials.append({"trial": i, "config": cfg.to_dict(), "valid_acc": accs["valid"],
                       "seconds": time.perf_counter() - t0})
        log(f"[search cs] {cfg.to_dict()} -> valid={accs['valid']:.4f}")
    best = max(trials, key=lambda t: t["valid_acc"])
    out = {"model": "cs", "base": "mlp", "seed": C.SEARCH_SEED, "selection": "valid_acc",
           "trials": trials, "best": best}
    _write_json(search_path("cs"), out)
    return out


def final_cs(data: NodeData, seeds=C.SEEDS, log=print) -> dict:
    cfg = best_config("cs") or CSConfig()
    runs = []
    for seed in seeds:
        probs = _load_mlp_probs(seed, data.x.device)
        t0 = time.perf_counter()
        accs = _cs_accuracies(data, probs, cfg, ("train", "valid", "test"))
        base = split_accuracies(probs.argmax(dim=-1), data.y, data.split, ("valid", "test"))
        runs.append({"seed": seed, "train_acc": accs["train"], "valid_acc": accs["valid"], "test_acc": accs["test"],
                     "base_valid_acc": base["valid"], "base_test_acc": base["test"],
                     "seconds": time.perf_counter() - t0})
        log(f"[final cs] graine {seed}: valid={accs['valid']:.4f} test={accs['test']:.4f} "
            f"(MLP seul : test={base['test']:.4f})")
    out = _summarize("cs", cfg.to_dict(), runs)
    if final_path("mlp").exists():
        out["params"] = _read_json(final_path("mlp"))["params"]
    _write_json(final_path("cs"), out)
    return out


def load_final(name: str):
    path = final_path(name)
    return _read_json(path) if path.exists() else None


def load_search(name: str):
    path = search_path(name)
    return _read_json(path) if path.exists() else None
