"""Temps par époque et mémoire vive maximale, sur CPU avec 2 threads.

L'entraînement est plein graphe : une époque est une seule étape (propagation
avant sur les 169 343 nœuds, perte sur les nœuds d'entraînement,
rétropropagation, pas d'Adam). Chaque mesure tourne dans un sous-processus
neuf, pour que la mémoire maximale (RSS) d'un modèle ne dépende pas des
mesures précédentes. Une mesure qui chevauche un ralentissement thermique
signalé par la sonde de la machine est refaite (voir thermal.py). La charge
moyenne du système est notée avant et après chaque mesure, parce que la
machine peut être partagée avec d'autres tâches.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from . import config as C
from .config import CSConfig, LPConfig, TrainConfig

MAX_ATTEMPTS = 4


def bench_path():
    return C.RESULTS_DIR / "bench.json"


def _cpu_name() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _loadavg() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_one(name: str, warmup: int, epochs: int) -> dict:
    """Mesure exécutée dans le sous-processus (`python -m arxiv_gnn _bench_one`)."""
    from .data import load_arxiv
    from .models import build_model, correct_and_smooth, count_parameters, label_propagation
    from .train import best_config, predict, prepare, set_seed, setup_threads, train_step

    setup_threads()
    dev = torch.device("cpu")
    data = prepare(load_arxiv(), name, dev, amp=False)
    res = {"model": name, "threads": torch.get_num_threads(), "rss_after_load_mb": _peak_rss_mb()}
    if name in C.NEURAL_MODELS:
        cfg = best_config(name) or TrainConfig()
        set_seed(0)
        model = build_model(name, data.x.size(1), data.num_classes, cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        times, infer = [], []
        for i in range(warmup + epochs):
            t0 = time.perf_counter()
            train_step(model, opt, scaler, data, amp=False)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
        for _ in range(2):
            t0 = time.perf_counter()
            predict(model, data, amp=False)
            infer.append(time.perf_counter() - t0)
        res.update(config=cfg.to_dict(), params=count_parameters(model), epoch_s=float(np.median(times)),
                   epoch_s_all=times, infer_s=float(np.median(infer)))
    else:
        times = []
        for _ in range(warmup + epochs):
            t0 = time.perf_counter()
            if name == "lp":
                cfg = best_config("lp") or LPConfig()
                label_propagation(data.graph.adj_lp, data.y, data.split["train"], data.num_classes, cfg)
            else:
                cfg = best_config("cs") or CSConfig()
                probs = torch.full((data.num_nodes, data.num_classes), 1.0 / data.num_classes)
                correct_and_smooth(probs, data.y, data.split["train"], data.graph, cfg)
            times.append(time.perf_counter() - t0)
        res.update(config=cfg.to_dict(), run_s=float(np.median(times[warmup:])))
    res["peak_rss_mb"] = _peak_rss_mb()
    return res


def _measure(cmd: list[str], env: dict, log) -> tuple[dict, list[str]]:
    from .thermal import throttle_events

    wall0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    events = throttle_events(wall0, time.time())
    if proc.returncode != 0:
        log(proc.stderr[-2000:])
        lines = proc.stderr.strip().splitlines()
        return {"model": cmd[cmd.index("--model") + 1], "status": "error", "error": lines[-1] if lines else "?"}, events
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["status"] = "ok"
    row["wall_s"] = time.time() - wall0
    return row, events


def run_bench(models=C.BENCH_MODELS, log=print) -> list[dict]:
    env = {**os.environ, "OMP_NUM_THREADS": str(C.NUM_THREADS), "MKL_NUM_THREADS": str(C.NUM_THREADS),
           "CUDA_VISIBLE_DEVICES": ""}
    rows = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for name in models:
        warmup, epochs = (1, 3) if name == "gat" else (1, 5)
        cmd = [sys.executable, "-m", "arxiv_gnn", "_bench_one", "--model", name,
               "--warmup", str(warmup), "--epochs", str(epochs)]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            load_before = _loadavg()
            row, events = _measure(cmd, env, log)
            row["attempt"] = attempt
            row["loadavg_before"], row["loadavg_after"] = load_before, _loadavg()
            if not events:
                break
            row["throttle_events"] = events
            log(f"[bench] {name} : ralentissement thermique pendant la mesure ({events[0]}), on recommence")
            time.sleep(60)
        rows.append(row)
        t = row.get("epoch_s", row.get("run_s"))
        log(f"[bench] {name:5s} {row['status']:5s} "
            f"{'' if t is None else f'{t:7.2f} s'} RSS max {row.get('peak_rss_mb', 0):.0f} Mo")
    out = {"device": "cpu", "threads": C.NUM_THREADS, "torch": torch.__version__,
           "cpu_quota": os.environ.get("ARXIV_GNN_CPU_QUOTA"), "cpu": _cpu_name(), "logical_cpus": os.cpu_count(),
           "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    bench_path().parent.mkdir(parents=True, exist_ok=True)
    bench_path().write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return rows


def load_bench():
    path = bench_path()
    return json.loads(path.read_text()) if path.exists() else None
