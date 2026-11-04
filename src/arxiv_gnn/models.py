"""Modèles : MLP sans graphe, GCN, GraphSAGE, GAT, propagation d'étiquettes, C&S.

GCN et GraphSAGE utilisent les couches de PyTorch Geometric sur une matrice
d'adjacence creuse CSR (une multiplication creuse par couche, rien n'est
matérialisé par arête). GAT est réécrit ici : la couche `GATConv` de PyG
garde pour la rétropropagation des tenseurs par arête, dont les messages
(arêtes × têtes × canaux, soit 2,5 M × 256 valeurs par couche sur ogbn-arxiv,
plus de 1 Go par couche en float32). `SparseGATConv` calcule la même chose
(test d'équivalence avec PyG dans tests/) avec une attention fusionnée, par
paquets d'arêtes, qui ne conserve aucun tenseur par arête.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.models import CorrectAndSmooth

from .config import CSConfig, LPConfig, TrainConfig
from .data import Graph

# Nombre maximal de valeurs (arêtes × têtes × canaux) traitées d'un coup par
# l'attention de GAT : 2^24 valeurs, soit 64 Mo en float32.
EDGE_BUDGET = 1 << 24


def _compute_dtype(t: Tensor) -> torch.dtype:
    return torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype


class _FusedGATAttention(torch.autograd.Function):
    """Attention de GAT et agrégation fusionnées, par paquets d'arêtes.

    Pour chaque arête j -> i et chaque tête : z = s_j + t_i, e = LeakyReLU(z),
    alpha = softmax de e sur les voisins entrants de i, out_i = somme alpha · h_j.

    Rien n'est gardé par arête : seuls s, t (N × H), h et out (N × H × C), le
    maximum et la somme du softmax par nœud (N × H) sont conservés ; les alpha
    sont recalculés pendant la rétropropagation, qui utilise
    d e_ij = alpha_ij · (G_i · h_j - G_i · out_i), avec G le gradient de out.
    """

    @staticmethod
    def forward(ctx, s, t, h, src, dst, slope: float, chunk: int):
        n, heads, ch = h.shape
        dt = _compute_dtype(h)
        m = torch.full((n, heads), float("-inf"), dtype=dt, device=h.device)
        for a in range(0, src.numel(), chunk):
            js, i = src[a : a + chunk], dst[a : a + chunk]
            e = F.leaky_relu(s[js].to(dt) + t[i].to(dt), slope)
            m.scatter_reduce_(0, i.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True)
        den = torch.zeros(n, heads, dtype=dt, device=h.device)
        acc = torch.zeros(n, heads, ch, dtype=dt, device=h.device)
        for a in range(0, src.numel(), chunk):
            js, i = src[a : a + chunk], dst[a : a + chunk]
            w = torch.exp(F.leaky_relu(s[js].to(dt) + t[i].to(dt), slope) - m[i])
            den.index_add_(0, i, w)
            acc.index_add_(0, i, w.unsqueeze(-1) * h[js].to(dt))
        den += 1e-16  # comme torch_geometric.utils.softmax
        out = acc.div_(den.unsqueeze(-1)).to(h.dtype)
        del acc
        ctx.save_for_backward(s, t, h, out, m, den, src, dst)
        ctx.slope, ctx.chunk = slope, chunk
        return out

    @staticmethod
    def backward(ctx, grad):
        s, t, h, out, m, den, src, dst = ctx.saved_tensors
        slope, chunk = ctx.slope, ctx.chunk
        dt = _compute_dtype(h)
        n = h.size(0)
        rows = max(1, chunk // 4)
        d_node = torch.empty(s.shape, dtype=dt, device=s.device)  # G_i · out_i, (N, H)
        for a in range(0, n, rows):
            d_node[a : a + rows] = (grad[a : a + rows].to(dt) * out[a : a + rows].to(dt)).sum(-1)
        ds = torch.zeros(s.shape, dtype=dt, device=s.device)
        dtt = torch.zeros(t.shape, dtype=dt, device=t.device)
        dh = torch.zeros(h.shape, dtype=dt, device=h.device)
        for a in range(0, src.numel(), chunk):
            js, i = src[a : a + chunk], dst[a : a + chunk]
            z = s[js].to(dt) + t[i].to(dt)
            alpha = torch.exp(F.leaky_relu(z, slope) - m[i]) / den[i]
            gi = grad[i].to(dt)  # (paquet, H, C)
            d_alpha = (gi * h[js].to(dt)).sum(-1)
            d_e = alpha * (d_alpha - d_node[i])
            d_z = torch.where(z > 0, d_e, d_e * slope)
            ds.index_add_(0, js, d_z)
            dtt.index_add_(0, i, d_z)
            dh.index_add_(0, js, alpha.unsqueeze(-1) * gi)
        return ds.to(s.dtype), dtt.to(t.dtype), dh.to(h.dtype), None, None, None, None


def gat_attention(s: Tensor, t: Tensor, h: Tensor, src: Tensor, dst: Tensor, slope: float = 0.2,
                  budget: int | None = None) -> Tensor:
    chunk = max(1, (budget or EDGE_BUDGET) // (h.size(1) * h.size(2)))
    return _FusedGATAttention.apply(s, t, h, src, dst, slope, chunk)


class _SymSpMM(torch.autograd.Function):
    """out = diag(s) · A · X pour une matrice creuse A symétrique (s optionnel).

    Le gradient par rapport à X vaut Aᵀ · diag(s) · G = A · (s ⊙ G) : comme A est
    symétrique, on réutilise la même matrice CSR au lieu de la transposer à
    chaque rétropropagation. Le membre dense est rendu contigu, car certains
    gradients arrivent rangés par colonnes et la multiplication creuse est
    alors beaucoup plus lente.
    """

    @staticmethod
    def forward(ctx, adj: Tensor, x: Tensor, scale: Tensor | None) -> Tensor:
        out = torch.sparse.mm(adj, x.contiguous())
        if scale is not None:
            out = out * scale.to(out.dtype).unsqueeze(-1)
        ctx.save_for_backward(scale)
        ctx.adj = adj  # la matrice ne reçoit pas de gradient
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        (scale,) = ctx.saved_tensors
        if scale is not None:
            grad = grad * scale.to(grad.dtype).unsqueeze(-1)
        return None, torch.sparse.mm(ctx.adj, grad.to(ctx.adj.dtype).contiguous()), None


def sym_spmm(adj: Tensor, x: Tensor, scale: Tensor | None = None) -> Tensor:
    if x.dtype != adj.dtype:  # sous autocast : la matrice est déjà en float16
        x = x.to(adj.dtype)
    return _SymSpMM.apply(adj, x, scale)


class SymGCNConv(GCNConv):
    """`GCNConv` de PyG dont l'agrégation passe par `sym_spmm` (Â symétrique)."""

    def message_and_aggregate(self, adj_t: Tensor, x: Tensor) -> Tensor:
        return sym_spmm(adj_t, x)


class SymSAGEConv(SAGEConv):
    """`SAGEConv` (moyenne des voisins) de PyG avec `sym_spmm` : D⁻¹ A X."""

    def message_and_aggregate(self, adj_t: Tensor, x) -> Tensor:
        x = x[0] if isinstance(x, tuple) else x
        deg = adj_t.crow_indices().diff().clamp_(min=1)
        return sym_spmm(adj_t, x, 1.0 / deg.float())


class SparseGATConv(nn.Module):
    """Couche GAT (Veličković et al., 2018), mêmes équations et mêmes paramètres
    que `torch_geometric.nn.GATConv` sans ajout de boucles (elles sont déjà
    dans `Graph.gat_index`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels, self.out_channels, self.heads = in_channels, out_channels, heads
        self.concat, self.negative_slope, self.attn_dropout = concat, negative_slope, attn_dropout
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.bias = nn.Parameter(torch.empty(heads * out_channels if concat else out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n, heads, ch = x.size(0), self.heads, self.out_channels
        h = self.lin(x).view(n, heads, ch)
        # einsum plutôt que (h * att).sum(-1) : sous autocast, le produit par un
        # paramètre float32 créerait un tenseur N × H × C en float32.
        a_src = torch.einsum("nhc,hc->nh", h, self.att_src[0].to(h.dtype))
        a_dst = torch.einsum("nhc,hc->nh", h, self.att_dst[0].to(h.dtype))
        if self.attn_dropout > 0 and self.training:
            raise NotImplementedError("dropout d'attention non pris en charge par l'attention fusionnée")
        out = gat_attention(a_src, a_dst, h, edge_index[0], edge_index[1], self.negative_slope)
        out = out.reshape(n, heads * ch) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class MLP(nn.Module):
    """Baseline sans graphe : chaque article est classé d'après son seul texte."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers - 1))
        self.dropout = dropout

    def forward(self, x: Tensor, graph: Graph | None = None, return_hidden: bool = False):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = F.dropout(F.relu(bn(lin(x)), inplace=True), p=self.dropout, training=self.training)
        out = self.lins[-1](x)
        return (out, x) if return_hidden else out


def _autocast_dtype(t: Tensor) -> torch.dtype:
    """Type de calcul courant : float16 sous autocast, sinon celui de `t`."""
    dev = t.device.type
    return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else t.dtype


class GNN(nn.Module):
    """Squelette commun à GCN, GraphSAGE et GAT, calqué sur les exemples OGB :
    [convolution, BatchNorm, ReLU, dropout] × (L - 1), puis une convolution.

    Sous autocast, la sortie de chaque convolution est ramenée en float16 :
    `GCNConv` ajoute son biais float32 après la multiplication creuse, ce qui
    repasserait toute la couche cachée (N × 256) en float32. Avec
    `checkpoint=True`, chaque convolution est recalculée pendant la
    rétropropagation au lieu de garder ses tenseurs par arête (utile pour GAT).
    """

    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        heads: int = 4,
        checkpoint: bool = False,
    ):
        super().__init__()
        if kind not in ("gcn", "sage", "gat"):
            raise ValueError(f"modèle inconnu : {kind}")
        if kind == "gat" and hidden % heads:
            raise ValueError("hidden doit être divisible par heads")
        self.kind, self.dropout, self.checkpoint = kind, dropout, checkpoint
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == num_layers - 1
            if kind == "gcn":
                conv = SymGCNConv(a, b, normalize=False, add_self_loops=False)
            elif kind == "sage":
                conv = SymSAGEConv(a, b, aggr="mean")
            else:
                conv = SparseGATConv(a, b if last else b // heads, heads=heads, concat=not last)
            self.convs.append(conv)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers - 1))

    def structure(self, graph: Graph) -> Tensor:
        return {"gcn": graph.adj_gcn, "sage": graph.adj, "gat": graph.gat_index}[self.kind]

    def _conv(self, conv: nn.Module, x: Tensor, a: Tensor) -> Tensor:
        if self.checkpoint and self.training and torch.is_grad_enabled():
            out = checkpoint(conv, x, a, use_reentrant=False)
        else:
            out = conv(x, a)
        return out.to(_autocast_dtype(out))

    def forward(self, x: Tensor, graph: Graph, return_hidden: bool = False):
        a = self.structure(graph)
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = F.relu(bn(self._conv(conv, x, a)), inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self._conv(self.convs[-1], x, a)
        return (out, x) if return_hidden else out


def build_model(name: str, in_dim: int, num_classes: int, cfg: TrainConfig) -> nn.Module:
    if name == "mlp":
        model = MLP(in_dim, cfg.hidden, num_classes, cfg.num_layers, cfg.dropout)
    else:
        model = GNN(name, in_dim, cfg.hidden, num_classes, cfg.num_layers, cfg.dropout, cfg.heads, cfg.checkpoint)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def label_propagation(adj: Tensor, y: Tensor, train_idx: Tensor, num_classes: int, cfg: LPConfig) -> Tensor:
    """Propagation d'étiquettes (Zhou et al., 2004), sans apprentissage :
    Y <- alpha * Â Y + (1 - alpha) * Y0, Y0 = one-hot des seuls nœuds d'entraînement,
    Â = D^-1/2 A D^-1/2. Les valeurs sont bornées à [0, 1] après chaque pas
    (même convention que `torch_geometric.nn.models.LabelPropagation`)."""
    y0 = torch.zeros(adj.size(0), num_classes, device=y.device)
    y0[train_idx] = F.one_hot(y[train_idx], num_classes).float()
    out = y0.clone()
    for _ in range(cfg.num_layers):
        out = cfg.alpha * (adj @ out) + (1.0 - cfg.alpha) * y0
        out.clamp_(0.0, 1.0)
    return out


@torch.no_grad()
def correct_and_smooth(y_soft: Tensor, y: Tensor, train_idx: Tensor, graph: Graph, cfg: CSConfig) -> Tensor:
    """Correct & Smooth (Huang et al., 2021) avec l'implémentation de PyG.
    Seules les étiquettes d'entraînement servent (règle OGB : la validation ne
    sert qu'au choix des hyperparamètres)."""
    cs = CorrectAndSmooth(
        num_correction_layers=cfg.num_layers,
        correction_alpha=cfg.correct_alpha,
        num_smoothing_layers=cfg.num_layers,
        smoothing_alpha=cfg.smooth_alpha,
        autoscale=cfg.autoscale,
        scale=cfg.scale,
    )
    adj = graph.adj_lp
    y_train = y[train_idx]
    out = cs.correct(y_soft, y_train, train_idx, adj)
    return cs.smooth(out, y_train, train_idx, adj)
