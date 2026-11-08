"""Interface en ligne de commande : python -m arxiv_gnn <commande>."""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OUTDATED_IGNORE", "1")  # ogb ne vérifie pas sa version en ligne


def _load(device: str, model: str | None = None, precision: str = "auto"):
    from .data import load_arxiv
    from .train import get_device, prepare, setup_threads

    setup_threads()
    dev = get_device(device)
    data = load_arxiv()
    amp = None if precision == "auto" else precision == "fp16"
    data = prepare(data, model, dev, amp=amp) if model else data.to(dev)
    print(f"ogbn-arxiv sur {dev} ({data.x.dtype}) : {data.num_nodes} nœuds, {data.graph.num_edges} arêtes orientées "
          f"après symétrisation, {data.num_classes} classes, "
          + ", ".join(f"{k}={v.numel()}" for k, v in data.split.items()), flush=True)
    return data


def cmd_data(args) -> None:
    data = _load("cpu")
    year = data.year
    for k, idx in data.split.items():
        y = year[idx]
        print(f"  {k:5s}: {idx.numel():6d} articles, années {int(y.min())}-{int(y.max())}")
    deg = data.graph.adj.crow_indices().diff().float()
    print(f"  degré moyen {deg.mean():.2f}, médian {deg.median():.0f}, max {deg.max():.0f}, "
          f"nœuds isolés {(deg == 0).sum().item()}")


def _search(model: str, data, epochs=None) -> None:
    from . import train as T

    if model == "lp":
        T.search_lp(data)
    elif model == "cs":
        T.search_cs(data)
    else:
        T.search_neural(model, data, epochs=epochs)


def cmd_search(args) -> None:
    _search(args.model, _load(args.device, args.model, args.precision), args.epochs)


def cmd_train(args) -> None:
    from . import train as T
    from .config import SEEDS

    data = _load(args.device, args.model, args.precision)
    seeds = tuple(args.seed) if args.seed else (tuple(range(args.seeds)) if args.seeds else SEEDS)
    if T.best_config(args.model) is None and not args.no_search:
        print(f"Pas de recherche enregistrée pour {args.model} : on la lance d'abord.", flush=True)
        _search(args.model, data, args.epochs)
    if args.model == "lp":
        T.final_lp(data, seeds)
    elif args.model == "cs":
        T.final_cs(data, seeds)
    else:
        cfg = T.best_config(args.model)
        if cfg is not None and args.epochs:
            cfg = type(cfg).from_dict({**cfg.to_dict(), "epochs": args.epochs})
        T.final_neural(args.model, data, cfg, seeds)


def cmd_bench(args) -> None:
    from .bench import run_bench
    from .config import BENCH_MODELS

    run_bench(args.models or BENCH_MODELS)


def cmd_bench_one(args) -> None:
    from .bench import bench_one

    print(json.dumps(bench_one(args.model, args.warmup, args.epochs)))


def cmd_figures(args) -> None:
    from .figures import make_all

    for path in make_all():
        print(f"écrit : {path}")


def cmd_report(args) -> None:
    from .report import write_report

    print(write_report())


def main(argv=None) -> None:
    from .config import ALL_MODELS

    p = argparse.ArgumentParser(prog="arxiv_gnn", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="télécharge ogbn-arxiv et affiche ses statistiques").set_defaults(fn=cmd_data)

    for name, fn, helptext in (("search", cmd_search, "recherche d'hyperparamètres sur la validation"),
                               ("train", cmd_train, "campagne finale (graines 0, 1, 2 par défaut)")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--model", required=True, choices=ALL_MODELS)
        sp.add_argument("--device", default="cpu", help="cpu (défaut) ou cuda (NVIDIA ou AMD via ROCm)")
        sp.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16"),
                        help="auto : fp16 (précision mixte) sur GPU, fp32 sur CPU")
        sp.add_argument("--epochs", type=int, default=None, help="surcharge du nombre d'époques (essais rapides)")
        if name == "train":
            sp.add_argument("--seeds", type=int, default=None, help="nombre de graines (défaut : 3)")
            sp.add_argument("--seed", type=int, action="append",
                            help="graine précise, répétable (pour répartir les graines entre deux processus)")
            sp.add_argument("--no-search", action="store_true")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("bench", help="temps par époque et mémoire vive maximale sur CPU")
    sp.add_argument("--models", nargs="*", choices=ALL_MODELS)
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("_bench_one")
    sp.add_argument("--model", required=True)
    sp.add_argument("--warmup", type=int, default=1)
    sp.add_argument("--epochs", type=int, default=10)
    sp.set_defaults(fn=cmd_bench_one)

    sub.add_parser("figures", help="figures PNG dans docs/").set_defaults(fn=cmd_figures)
    sub.add_parser("report", help="tableau récapitulatif (results/summary.md)").set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
