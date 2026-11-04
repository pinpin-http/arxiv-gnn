"""Chargement d'ogbn-arxiv (package `ogb`) et construction des structures creuses.

Le graphe officiel est orienté (A cite B). Comme dans les exemples OGB, on le
rend symétrique : une citation relie les deux articles dans les deux sens. Le
découpage train/val/test et les étiquettes restent ceux d'OGB, sans retouche.
"""

from __future__ import annotations

import gzip
import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import remove_self_loops, to_undirected

from .config import DATA_DIR, DATASET

SPLITS = ("train", "valid", "test")


# Structure de graphe utilisée par chaque méthode.
STRUCTURE = {"mlp": (), "gcn": ("adj_gcn",), "sage": ("adj",), "gat": ("gat_index",), "lp": ("adj_lp",), "cs": ("adj_lp",)}


@dataclass
class Graph:
    """Le même graphe sous les formes dont chaque modèle a besoin.

    Convention PyG : `edge_index[0]` est la source, `edge_index[1]` la cible.
    Les matrices creuses sont au format CSR (indices int32), lignes = cibles,
    colonnes = sources, si bien que `adj @ h` agrège sur chaque nœud les
    messages de ses voisins. Les champs inutiles à un modèle peuvent valoir
    None (voir `NodeData.for_model`) pour économiser la mémoire.
    """

    num_nodes: int
    num_edges: int  # arêtes orientées après symétrisation (2 par citation)
    edge_index: Tensor | None = None  # (2, E) symétrique, sans boucle, trié
    adj: Tensor | None = None  # CSR binaire, pour la moyenne de GraphSAGE
    adj_gcn: Tensor | None = None  # CSR D^-1/2 (A + I) D^-1/2, pour GCN
    adj_lp: Tensor | None = None  # CSR D^-1/2 A D^-1/2, pour la propagation d'étiquettes et C&S
    gat_index: Tensor | None = None  # (2, E + N) avec boucles, trié par cible, pour GAT

    def to(self, device: torch.device | str) -> "Graph":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device) if isinstance(v, Tensor) else v
        return Graph(**kw)

    def half(self) -> "Graph":
        """Poids des matrices CSR en float16 (les indices ne changent pas)."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        for k in ("adj", "adj_gcn", "adj_lp"):
            a = kw[k]
            if a is not None:
                kw[k] = torch.sparse_csr_tensor(a.crow_indices(), a.col_indices(), a.values().half(), a.shape)
        return Graph(**kw)

    def only(self, names) -> "Graph":
        """Copie qui ne garde que les structures `names` (les autres à None)."""
        structures = ("edge_index", "adj", "adj_gcn", "adj_lp", "gat_index")
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        kw.update({k: None for k in structures if k not in names})
        return Graph(**kw)


@dataclass
class NodeData:
    x: Tensor  # (N, F) float32 (float16 sur GPU en précision mixte)
    y: Tensor  # (N,) int64
    split: dict[str, Tensor]  # indices int64 des nœuds de chaque ensemble
    graph: Graph
    num_classes: int
    year: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def to(self, device: torch.device | str, half: bool = False) -> "NodeData":
        """Copie sur `device` ; avec `half`, caractéristiques et poids des
        matrices creuses en float16 (cuSPARSE refuse de mélanger une matrice
        float32 et un gradient float16 dans la rétropropagation)."""
        x = self.x.to(device)
        graph = self.graph.to(device)
        if half:
            graph = graph.half()
        return NodeData(
            x=x.half() if half else x,
            y=self.y.to(device),
            split={k: v.to(device) for k, v in self.split.items()},
            graph=graph,
            num_classes=self.num_classes,
            year=None if self.year is None else self.year.to(device),
        )

    def for_model(self, name: str) -> "NodeData":
        """Ne garde que la structure de graphe dont la méthode `name` a besoin."""
        return NodeData(self.x, self.y, self.split, self.graph.only(STRUCTURE[name]), self.num_classes, self.year)


def _csr(dst: Tensor, src: Tensor, values: Tensor, n: int) -> Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
        coo = torch.sparse_coo_tensor(torch.stack([dst, src]), values, (n, n), check_invariants=False)
        csr = coo.coalesce().to_sparse_csr()
        return torch.sparse_csr_tensor(csr.crow_indices().int(), csr.col_indices().int(), csr.values(), (n, n))


def _sym_norm_weights(dst: Tensor, src: Tensor, n: int) -> Tensor:
    deg = torch.zeros(n, dtype=torch.float32).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    inv_sqrt = deg.pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
    return inv_sqrt[dst] * inv_sqrt[src]


def build_graph(edge_index: Tensor, num_nodes: int) -> Graph:
    """Symétrise le graphe, retire boucles et doublons, prépare toutes les variantes."""
    ei = to_undirected(edge_index.long(), num_nodes=num_nodes)
    ei, _ = remove_self_loops(ei)
    src, dst = ei[0], ei[1]

    adj = _csr(dst, src, torch.ones(src.numel()), num_nodes)
    adj_lp = _csr(dst, src, _sym_norm_weights(dst, src, num_nodes), num_nodes)

    loops = torch.arange(num_nodes)
    src_sl, dst_sl = torch.cat([src, loops]), torch.cat([dst, loops])
    adj_gcn = _csr(dst_sl, src_sl, _sym_norm_weights(dst_sl, src_sl, num_nodes), num_nodes)

    order = torch.argsort(dst_sl * num_nodes + src_sl)
    gat_index = torch.stack([src_sl[order], dst_sl[order]])

    return Graph(num_nodes, int(ei.size(1)), ei, adj, adj_gcn, adj_lp, gat_index)


def load_arxiv(root: Path | str = DATA_DIR) -> NodeData:
    """Télécharge (première fois) puis charge ogbn-arxiv avec le découpage officiel."""
    from ogb.nodeproppred import NodePropPredDataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # ogb 1.3.6 relit son propre cache (tableaux numpy) avec torch.load ; depuis
    # torch 2.6 il faut autoriser le chargement complet de ce fichier local.
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = NodePropPredDataset(name=DATASET, root=str(root))
            g, labels = ds[0]
            split = ds.get_idx_split()
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous

    x = torch.from_numpy(np.asarray(g["node_feat"], dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels).reshape(-1).astype(np.int64))
    n = int(g["num_nodes"])
    graph = build_graph(torch.from_numpy(np.asarray(g["edge_index"])), n)
    year = torch.from_numpy(np.asarray(g["node_year"]).reshape(-1).astype(np.int64))
    return NodeData(
        x=x,
        y=y,
        split={k: torch.from_numpy(np.asarray(split[k]).astype(np.int64)) for k in SPLITS},
        graph=graph,
        num_classes=int(y.max()) + 1,
        year=year,
    )


def category_names(root: Path | str = DATA_DIR) -> list[str]:
    """Noms des 40 catégories arXiv (ex. « cs.LG »), dans l'ordre des étiquettes."""
    path = Path(root) / "ogbn_arxiv" / "mapping" / "labelidx2arxivcategeory.csv.gz"
    with gzip.open(path, "rt") as f:
        rows = [line.strip().split(",", 1) for line in f.readlines()[1:] if line.strip()]
    names = {}
    for idx, cat in rows:
        parts = cat.split()  # « arxiv cs lg »
        names[int(idx)] = f"{parts[-2]}.{parts[-1].upper()}" if len(parts) >= 2 else cat
    return [names[i] for i in sorted(names)]


def synthetic_graph(
    num_nodes: int = 90,
    num_classes: int = 3,
    feat_dim: int = 8,
    p_in: float = 0.15,
    p_out: float = 0.01,
    feature_noise: float = 2.0,
    seed: int = 0,
) -> NodeData:
    """Petit graphe à blocs (SBM) pour les tests : homophile, avec des
    caractéristiques bruitées corrélées à la classe et un découpage disjoint."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(num_nodes) % num_classes
    y = y[torch.randperm(num_nodes, generator=gen)]
    same = y[:, None] == y[None, :]
    prob = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
    upper = torch.triu(torch.rand(num_nodes, num_nodes, generator=gen) < prob, diagonal=1)
    edge_index = upper.nonzero().t()
    centers = torch.randn(num_classes, feat_dim, generator=gen)
    x = centers[y] + feature_noise * torch.randn(num_nodes, feat_dim, generator=gen)
    perm = torch.randperm(num_nodes, generator=gen)
    n_tr, n_va = int(0.5 * num_nodes), int(0.25 * num_nodes)
    split = {
        "train": perm[:n_tr].sort().values,
        "valid": perm[n_tr : n_tr + n_va].sort().values,
        "test": perm[n_tr + n_va :].sort().values,
    }
    return NodeData(x=x, y=y, split=split, graph=build_graph(edge_index, num_nodes), num_classes=num_classes)
