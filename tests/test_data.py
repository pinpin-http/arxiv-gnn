import pytest
import torch

from arxiv_gnn import config as C
from arxiv_gnn.data import build_graph, load_arxiv


def test_graph_is_symmetric_without_loops_or_duplicates():
    ei = torch.tensor([[0, 1, 1, 2, 3, 3], [1, 0, 2, 2, 0, 0]])  # doublon 3->0, boucle 2->2
    g = build_graph(ei, num_nodes=5)
    pairs = {tuple(e) for e in g.edge_index.t().tolist()}
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (3, 0)}
    assert g.num_edges == len(pairs)
    dense = g.adj.to_dense()
    assert torch.equal(dense, dense.t())
    assert dense.diagonal().sum() == 0
    assert dense[4].sum() == 0  # nœud isolé


def test_gat_index_has_all_self_loops_and_is_sorted_by_target():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    g = build_graph(ei, num_nodes=5)
    src, dst = g.gat_index
    assert src.numel() == g.num_edges + 5
    loops = src == dst
    assert sorted(src[loops].tolist()) == list(range(5))
    assert torch.all(dst[1:] >= dst[:-1])


def test_normalized_adjacencies():
    ei = torch.tensor([[0, 0, 1], [1, 2, 2]])
    g = build_graph(ei, num_nodes=4)
    a = torch.zeros(4, 4)
    a[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = 1
    deg = a.sum(1)
    inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    assert torch.allclose(g.adj_lp.to_dense(), inv[:, None] * a * inv[None, :])
    a_hat = a + torch.eye(4)
    inv = a_hat.sum(1).pow(-0.5)
    assert torch.allclose(g.adj_gcn.to_dense(), inv[:, None] * a_hat * inv[None, :])


def test_synthetic_split_is_disjoint_and_complete(toy):
    parts = [set(toy.split[k].tolist()) for k in ("train", "valid", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert set().union(*parts) == set(range(toy.num_nodes))


_has_arxiv = (C.DATA_DIR / "ogbn_arxiv").exists()


@pytest.mark.skipif(not _has_arxiv, reason="ogbn-arxiv non téléchargé (make data)")
def test_official_split_follows_publication_year():
    data = load_arxiv()
    assert data.num_nodes == 169_343
    assert data.num_classes == 40
    sizes = {k: v.numel() for k, v in data.split.items()}
    assert sizes == {"train": 90_941, "valid": 29_799, "test": 48_603}
    tr, va, te = (data.year[data.split[k]] for k in ("train", "valid", "test"))
    assert tr.max() <= 2017 and torch.all(va == 2018) and te.min() >= 2019
    all_idx = torch.cat(list(data.split.values()))
    assert all_idx.unique().numel() == data.num_nodes
