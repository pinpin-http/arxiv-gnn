"""Figures PNG (matplotlib, backend Agg) écrites dans docs/."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as C
from .bench import load_bench
from .reference import LEADERBOARD
from .train import embed_path, load_final

# Palette catégorielle de référence (ordre fixe, validée pour le daltonisme).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
CURVE_MODELS = ("mlp", "gcn", "sage")
MODEL_COLOR = {"mlp": SERIES[0], "gcn": SERIES[1], "sage": SERIES[2]}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    })


def fr(x: float, digits: int = 1) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{digits}f}".replace(".", ",")


def _save(fig, name: str):
    C.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.DOCS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_accuracy():
    rows = [(m, load_final(m)) for m in C.RESULT_MODELS]
    rows = [(m, r) for m, r in rows if r is not None]
    if not rows:
        return None
    rows.sort(key=lambda mr: mr[1]["test_mean"])
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(rows) + 1.5))
    y = np.arange(len(rows))
    for i, (m, r) in enumerate(rows):
        tm, ts = 100 * r["test_mean"], 100 * r["test_std"]
        vm, vs = 100 * r["valid_mean"], 100 * r["valid_std"]
        ax.errorbar(vm, i + 0.14, xerr=vs, fmt="o", ms=6, color=SERIES[1], ecolor=SERIES[1], capsize=3,
                    mec=SURFACE, mew=1.2, label="validation" if i == 0 else None, zorder=3)
        ax.errorbar(tm, i - 0.14, xerr=ts, fmt="o", ms=7, color=SERIES[0], ecolor=SERIES[0], capsize=3,
                    mec=SURFACE, mew=1.2, label="test" if i == 0 else None, zorder=4)
        ref = LEADERBOARD.get(m)
        if ref:
            ax.plot(100 * ref["test"][0], i - 0.14, marker="D", ms=7, mfc="none", mec=INK, mew=1.3, zorder=5,
                    label="classement OGB (test)" if m == "mlp" or i == 0 else None)
        ax.annotate(f"test {fr(tm, 2)} ± {fr(ts, 2)}", (max(tm, vm) + 1.2, i), va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [C.MODEL_LABELS[m] for m, _ in rows])
    ax.set_xlabel("accuracy (%)")
    lo = min(min(100 * r["test_mean"], 100 * r["valid_mean"]) for _, r in rows)
    ax.set_xlim(np.floor(lo / 5) * 5 - 1, 85)
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("test", "validation", "classement OGB (test)") if k in labels]
    ax.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower right", fontsize=9)
    n_seeds = len(rows[0][1]["seeds"])
    ax.set_title(f"ogbn-arxiv : accuracy moyenne ± écart-type sur {n_seeds} graines", loc="left")
    return _save(fig, "accuracy.png")


def fig_learning_curves():
    finals = {m: load_final(m) for m in CURVE_MODELS}
    finals = {m: f for m, f in finals.items() if f}
    if not finals:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, f in finals.items():
        color = MODEL_COLOR[m]
        loss = np.array([r["history"]["loss"] for r in f["runs"]])
        ep = np.arange(1, loss.shape[1] + 1)
        ax1.plot(ep, loss.mean(0), color=color, lw=2, label=C.MODEL_LABELS[m])
        ax1.fill_between(ep, loss.min(0), loss.max(0), color=color, alpha=0.12, lw=0)
        val = 100 * np.array([r["history"]["valid"] for r in f["runs"]])
        tr = 100 * np.array([r["history"]["train"] for r in f["runs"]])
        ev = np.array(f["runs"][0]["history"]["epoch"])
        ax2.plot(ev, val.mean(0), color=color, lw=2, label=f"{C.MODEL_LABELS[m]} (validation)")
        ax2.fill_between(ev, val.min(0), val.max(0), color=color, alpha=0.12, lw=0)
        ax2.plot(ev, tr.mean(0), color=color, lw=1.2, ls=(0, (1, 2)), label=f"{C.MODEL_LABELS[m]} (entraînement)")
        ax2.annotate(fr(val.mean(0)[-1]), (ev[-1], val.mean(0)[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK2)
    ax1.set_yscale("log")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("perte d'entraînement (échelle log)")
    ax1.set_title("Perte (moyenne, enveloppe min-max des graines)", loc="left")
    ax1.legend(fontsize=9)
    ax2.set_xlabel("époque")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(40, 85)
    ax2.set_title("Accuracy : validation (trait plein), entraînement (pointillé)", loc="left")
    ax2.legend(fontsize=7.5, ncol=1, loc="lower right")
    fig.tight_layout()
    return _save(fig, "learning_curves.png")


def fig_cpu_cost():
    bench = load_bench()
    if not bench:
        return None
    rows = [r for r in bench["rows"] if r.get("status") == "ok" and "epoch_s" in r]
    if not rows:
        return None
    order = [m for m in ("mlp", "gcn", "sage", "gat") if any(r["model"] == m for r in rows)]
    rows = [next(r for r in rows if r["model"] == m) for m in order]
    trained = set(C.RESULT_MODELS)
    labels = [C.MODEL_LABELS.get(r["model"], r["model"]) + ("" if r["model"] in trained else "\n(non entraîné)")
              for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.7 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))
    for ax, key, unit, fmt in ((ax1, "epoch_s", "s", lambda v: f"{fr(v)} s"),
                               (ax2, "peak_rss_mb", "Mo", lambda v: f"{fr(v / 1024, 2)} Go")):
        vals = [r[key] for r in rows]
        colors = [SERIES[0] if r["model"] in trained else MUTED for r in rows]
        ax.barh(y, vals, height=0.5, color=colors, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.annotate(fmt(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis="y", visible=False)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("secondes par époque (médiane)")
    ax2.set_xlabel("mémoire vive maximale du processus (Go)")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v / 1024, 1)))
    ax1.set_title(f"Une époque plein graphe sur CPU, {bench['threads']} threads", loc="left")
    ax2.set_title("RAM maximale (données + modèle + gradients)", loc="left")
    fig.tight_layout()
    return _save(fig, "cpu_cost.png")


def _tsne(hidden: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    z = PCA(n_components=50, random_state=seed).fit_transform(hidden)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, n_jobs=C.NUM_THREADS).fit_transform(z)


def fig_tsne():
    from sklearn.metrics import silhouette_score

    from .data import category_names, load_arxiv

    finals = {m: load_final(m) for m in ("gcn", "sage")}
    finals = {m: f for m, f in finals.items() if f}
    if not finals or not embed_path("mlp").exists():
        return None
    best_gnn = max(finals, key=lambda m: finals[m]["valid_mean"])  # choix sur la validation
    data = load_arxiv()
    names = category_names()
    panels = []
    for m in ("mlp", best_gnn):
        emb = torch.load(embed_path(m))
        h = emb["hidden"].float().numpy()
        y = data.y[emb["nodes"]].numpy()
        panels.append((m, _tsne(h), y, silhouette_score(h, y, metric="cosine", sample_size=3000, random_state=0)))
    y = panels[0][2]
    top = [int(c) for c in np.argsort(-np.bincount(y, minlength=data.num_classes))[:8]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (m, xy, yy, sil) in zip(axes, panels):
        other = ~np.isin(yy, top)
        ax.scatter(xy[other, 0], xy[other, 1], s=3, c=MUTED, alpha=0.5, lw=0, label="32 autres catégories")
        for k, c in enumerate(top):
            sel = yy == c
            ax.scatter(xy[sel, 0], xy[sel, 1], s=4, c=SERIES[k], alpha=0.8, lw=0, label=names[c])
            cx, cy = np.median(xy[sel, 0]), np.median(xy[sel, 1])
            ax.annotate(names[c], (cx, cy), fontsize=8, color=INK, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{C.MODEL_LABELS[m]} : silhouette {fr(sil, 3)}", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center", ncol=5, fontsize=8.5,
               markerscale=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"t-SNE de la dernière couche cachée, {len(y)} articles de test (graine 0) ; "
                 "couleur = 8 catégories les plus fréquentes", x=0.02, ha="left", fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return _save(fig, "tsne.png")


def make_all() -> list:
    _style()
    paths = [fig_accuracy(), fig_learning_curves(), fig_cpu_cost(), fig_tsne()]
    return [p for p in paths if p is not None]
