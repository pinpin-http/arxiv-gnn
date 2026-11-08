"""Tableau récapitulatif des résultats mesurés (results/summary.md)."""

from __future__ import annotations

import json
import statistics

from . import config as C
from .bench import load_bench
from .reference import LEADERBOARD, LEADERBOARD_URL
from .train import load_final, load_search


def _pct(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}".replace(".", ",")


def _int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _cfg(model: str, cfg: dict) -> str:
    if model == "lp":
        text = f"alpha = {cfg['alpha']}, {cfg['num_layers']} itérations"
    elif model == "cs":
        text = f"alpha correction = {cfg['correct_alpha']}, alpha lissage = {cfg['smooth_alpha']}"
    else:
        text = f"{cfg['num_layers']} couches, {cfg['hidden']} neurones, dropout {cfg['dropout']}, {cfg['epochs']} époques"
    return text.replace(".", ",")


# Modèles entraînés une seconde fois sur GPU, avec le même protocole, pour contrôle.
GPU_CHECK_MODELS = ("gcn", "sage")


def gpu_check_lines() -> list[str]:
    """Section de contrôle CPU/GPU, si results/gpu_check/final/ existe."""
    root = C.RESULTS_DIR / "gpu_check" / "final"
    pairs = []
    for m in GPU_CHECK_MODELS:
        cpu, path = load_final(m), root / f"{m}.json"
        if cpu is not None and path.exists():
            pairs.append((m, cpu, json.loads(path.read_text())))
    if not pairs:
        return []
    lines = [
        "Contrôle CPU/GPU : mêmes modèles, même recherche (4 configurations, graine 0), mêmes graines, "
        "relancés sur GPU (results/gpu_check/).",
        "",
        "| Méthode | Configuration retenue, CPU | Configuration retenue, GPU | Validation CPU (%) "
        "| Validation GPU (%) | Test CPU (%) | Test GPU (%) | Écart de test, GPU moins CPU (points) "
        "| Précision GPU | s / époque sur GPU (médiane) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for m, cpu, gpu in pairs:
        cfg_cpu, cfg_gpu = _cfg(m, cpu["config"]), _cfg(m, gpu["config"])
        same = "identique" if cpu["config"] == gpu["config"] else cfg_gpu
        prec = sorted({r.get("precision", "fp32") for r in gpu["runs"]})
        times = [r["epoch_s"] for r in gpu["runs"] if "epoch_s" in r]
        t_txt = "" if not times else f"{statistics.median(times):.2f}".replace(".", ",")
        diff = f"{100 * (gpu['test_mean'] - cpu['test_mean']):+.2f}".replace(".", ",")
        lines.append(
            f"| {C.MODEL_LABELS[m]} | {cfg_cpu} | {same} | {_pct(cpu['valid_mean'], cpu['valid_std'])} "
            f"| {_pct(gpu['valid_mean'], gpu['valid_std'])} | {_pct(cpu['test_mean'], cpu['test_std'])} "
            f"| {_pct(gpu['test_mean'], gpu['test_std'])} | {diff} | {', '.join(prec)} | {t_txt} |"
        )
    for m in C.RESULT_MODELS:
        f = load_final(m)
        if m in GPU_CHECK_MODELS or f is None or not all(r.get("device") == "cuda" for r in f["runs"]):
            continue
        prec = ", ".join(sorted({r.get("precision", "fp32") for r in f["runs"]}))
        t = statistics.median(r["epoch_s"] for r in f["runs"])
        lines += ["", f"{C.MODEL_LABELS[m]}, entraîné sur GPU seulement ({prec}) : "
                      f"{t:.2f} s par époque (médiane des graines).".replace(".", ",", 1)]
    return lines + [""]


def write_report() -> str:
    bench = load_bench() or {"rows": []}
    by_model = {r["model"]: r for r in bench["rows"] if r.get("status") == "ok"}
    lines = [
        "| Méthode | Configuration retenue (validation) | Paramètres | Validation (%) | Test (%) "
        "| Classement OGB, test (%) | s / époque (CPU, 2 threads) | RAM max (Go) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in C.RESULT_MODELS:
        f = load_final(m)
        if f is None:
            continue
        ref = LEADERBOARD.get(m)
        b = by_model.get(m, {})
        t = b.get("epoch_s", b.get("run_s"))
        t_txt = "" if t is None else (f"{t:.2f}" if "epoch_s" in b else f"{t:.1f} (calcul complet)").replace(".", ",")
        rss = b.get("peak_rss_mb")
        gpu = any(r.get("device") == "cuda" for r in f["runs"])
        label = C.MODEL_LABELS[m] + (" (entraîné sur GPU)" if gpu else "")
        lines.append(
            f"| {label} | {_cfg(m, f['config'])} | {_int(f.get('params', 0))}"
            f" | {_pct(f['valid_mean'], f['valid_std'])} | {_pct(f['test_mean'], f['test_std'])} "
            f"| {'' if ref is None else _pct(*ref['test'])} | {t_txt} "
            f"| {'' if rss is None else f'{rss / 1024:.2f}'.replace('.', ',')} |"
        )
    if "gat" in by_model and load_final("gat") is None:
        b = by_model["gat"]
        lines.append(f"| GAT (chronométré, non entraîné) | {_cfg('gat', b['config'])} | {_int(b['params'])}"
                     f" | | | | {b['epoch_s']:.2f}".replace(".", ",")
                     + f" | {b['peak_rss_mb'] / 1024:.2f} |".replace(".", ","))
    seeds = next((load_final(m)["seeds"] for m in C.RESULT_MODELS if load_final(m)), [])
    lines += ["", f"Graines : {', '.join(map(str, seeds))}. Classement : {LEADERBOARD_URL}", ""]
    lines += gpu_check_lines()
    lines.append("Recherche d'hyperparamètres (graine 0, accuracy de validation) :")
    lines.append("")
    for m in C.RESULT_MODELS:
        s = load_search(m)
        if s is None:
            continue
        trials = " ; ".join(f"{_cfg(m, t['config'])} : {100 * t['valid_acc']:.2f}".replace(".", ",")
                           for t in s["trials"])
        lines.append(f"- {C.MODEL_LABELS[m]} : {trials}")
    text = "\n".join(lines) + "\n"
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "summary.md").write_text(text)
    return text
