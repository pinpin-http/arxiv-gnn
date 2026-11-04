"""Tableau récapitulatif des résultats mesurés (results/summary.md)."""

from __future__ import annotations

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
        lines.append(
            f"| {C.MODEL_LABELS[m]} | {_cfg(m, f['config'])} | {_int(f.get('params', 0))}"
            f" | {_pct(f['valid_mean'], f['valid_std'])} | {_pct(f['test_mean'], f['test_std'])} "
            f"| {'' if ref is None else _pct(*ref['test'])} | {t_txt} "
            f"| {'' if rss is None else f'{rss / 1024:.2f}'.replace('.', ',')} |"
        )
    if "gat" in by_model:
        b = by_model["gat"]
        lines.append(f"| GAT (chronométré, non entraîné) | {_cfg('gat', b['config'])} | {_int(b['params'])}"
                     f" | | | | {b['epoch_s']:.2f}".replace(".", ",")
                     + f" | {b['peak_rss_mb'] / 1024:.2f} |".replace(".", ","))
    seeds = next((load_final(m)["seeds"] for m in C.RESULT_MODELS if load_final(m)), [])
    lines += ["", f"Graines : {', '.join(map(str, seeds))}. Classement : {LEADERBOARD_URL}", ""]
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
