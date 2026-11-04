import torch
from torch_geometric.nn.models import LabelPropagation

from arxiv_gnn.config import CSConfig, LPConfig
from arxiv_gnn.data import build_graph
from arxiv_gnn.models import correct_and_smooth, label_propagation


def two_triangles():
    """Deux triangles {0,1,2} et {3,4,5} reliés par l'arête 2-3."""
    ei = torch.tensor([[0, 0, 1, 3, 3, 4, 2], [1, 2, 2, 4, 5, 5, 3]])
    g = build_graph(ei, num_nodes=6)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    return g, y


def test_label_propagation_on_toy_graph():
    g, y = two_triangles()
    train = torch.tensor([0, 5])  # une étiquette connue par triangle
    out = label_propagation(g.adj_lp, y, train, 2, LPConfig(num_layers=50, alpha=0.9))
    assert out.argmax(1).tolist() == [0, 0, 0, 1, 1, 1]
    assert out.min() >= 0 and out.max() <= 1


def test_label_propagation_matches_pyg():
    torch.manual_seed(0)
    n = 40
    ei = torch.randint(0, n, (2, 120))
    g = build_graph(ei, num_nodes=n)
    y = torch.randint(0, 4, (n,))
    train = torch.arange(0, n, 3)
    cfg = LPConfig(num_layers=10, alpha=0.8)
    ours = label_propagation(g.adj_lp, y, train, 4, cfg)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[train] = True
    # PyG calcule lui-même D^-1/2 A D^-1/2 à partir d'edge_index : on vérifie
    # au passage notre normalisation.
    ref = LabelPropagation(num_layers=10, alpha=0.8)(y, g.edge_index, mask=mask)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_label_propagation_ignores_non_train_labels():
    g, y = two_triangles()
    train = torch.tensor([0, 5])
    y2 = y.clone()
    y2[[1, 2, 3, 4]] = torch.tensor([1, 1, 0, 0])
    cfg = LPConfig(num_layers=20, alpha=0.9)
    assert torch.equal(label_propagation(g.adj_lp, y, train, 2, cfg), label_propagation(g.adj_lp, y2, train, 2, cfg))


def test_correct_and_smooth_fixes_a_wrong_base_prediction():
    g, y = two_triangles()
    train = torch.tensor([0, 1, 4, 5])
    base = torch.tensor([[0.9, 0.1]] * 3 + [[0.6, 0.4]] + [[0.1, 0.9]] * 2)  # nœud 3 mal classé
    out = correct_and_smooth(base, y, train, g, CSConfig(correct_alpha=0.9, smooth_alpha=0.9, num_layers=30))
    assert out.argmax(1).tolist() == [0, 0, 0, 1, 1, 1]


def test_correct_and_smooth_ignores_non_train_labels():
    g, y = two_triangles()
    train = torch.tensor([0, 5])
    base = torch.softmax(torch.randn(6, 2, generator=torch.Generator().manual_seed(0)), dim=1)
    y2 = y.clone()
    y2[[1, 2, 3, 4]] = 1 - y2[[1, 2, 3, 4]]
    cfg = CSConfig()
    assert torch.allclose(correct_and_smooth(base, y, train, g, cfg), correct_and_smooth(base, y2, train, g, cfg))
