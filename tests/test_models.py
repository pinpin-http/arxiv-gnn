import pytest
import torch
from torch_geometric.nn import GATConv, GCNConv

from arxiv_gnn import models
from arxiv_gnn.config import TrainConfig
from arxiv_gnn.models import SparseGATConv, build_model, gat_attention


@pytest.mark.parametrize("name", ["mlp", "gcn", "sage", "gat"])
def test_output_shapes(toy, name):
    cfg = TrainConfig(hidden=16, num_layers=3, heads=4)
    model = build_model(name, toy.x.size(1), toy.num_classes, cfg).eval()
    out, hidden = model(toy.x, toy.graph, return_hidden=True)
    assert out.shape == (toy.num_nodes, toy.num_classes)
    assert hidden.shape == (toy.num_nodes, cfg.hidden)
    assert torch.isfinite(out).all()


def test_reference_parameter_counts():
    """Mêmes tailles que les exemples OGB (110 120 et 218 664 paramètres)."""
    cfg = TrainConfig(hidden=256, num_layers=3)
    assert models.count_parameters(build_model("mlp", 128, 40, cfg)) == 110_120
    assert models.count_parameters(build_model("gcn", 128, 40, cfg)) == 110_120
    assert models.count_parameters(build_model("sage", 128, 40, cfg)) == 218_664


def _dense_adjacency(edge_index, n):
    a = torch.zeros(n, n)
    a[edge_index[0], edge_index[1]] = 1.0
    a = torch.maximum(a, a.t())
    a.fill_diagonal_(0.0)
    return a


def test_gcn_layer_matches_dense_reference(toy):
    torch.manual_seed(0)
    conv = GCNConv(toy.x.size(1), 5, normalize=False, add_self_loops=False)
    torch.nn.init.normal_(conv.bias)
    out = conv(toy.x, toy.graph.adj_gcn)

    a_hat = _dense_adjacency(toy.graph.edge_index, toy.num_nodes) + torch.eye(toy.num_nodes)
    d = a_hat.sum(1).pow(-0.5)
    ref = (d[:, None] * a_hat * d[None, :]) @ (toy.x @ conv.lin.weight.t()) + conv.bias
    assert torch.allclose(out, ref, atol=1e-5)

    # Et la normalisation intégrée de PyG, sur edge_index, donne la même chose.
    pyg = GCNConv(toy.x.size(1), 5)
    pyg.load_state_dict(conv.state_dict())
    assert torch.allclose(pyg(toy.x, toy.graph.edge_index), ref, atol=1e-5)


def test_sage_mean_aggregation_matches_dense(toy):
    model = build_model("sage", toy.x.size(1), 3, TrainConfig(hidden=8, num_layers=1))
    conv = model.convs[0]
    out = conv(toy.x, toy.graph.adj)
    a = _dense_adjacency(toy.graph.edge_index, toy.num_nodes)
    mean = (a @ toy.x) / a.sum(1, keepdim=True).clamp(min=1)
    ref = conv.lin_l(mean) + conv.lin_r(toy.x)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("concat", [True, False])
@pytest.mark.parametrize("budget", [None, 7])  # 7 : force de nombreux petits paquets
def test_sparse_gat_matches_pyg_gatconv(toy, monkeypatch, concat, budget):
    if budget is not None:
        monkeypatch.setattr(models, "EDGE_BUDGET", budget)
    torch.manual_seed(0)
    heads, ch = 3, 4
    ours = SparseGATConv(toy.x.size(1), ch, heads=heads, concat=concat)
    torch.nn.init.normal_(ours.bias)
    ref = GATConv(toy.x.size(1), ch, heads=heads, concat=concat, add_self_loops=False)
    ref.lin.weight.data.copy_(ours.lin.weight.data)
    ref.att_src.data.copy_(ours.att_src.data)
    ref.att_dst.data.copy_(ours.att_dst.data)
    ref.bias.data.copy_(ours.bias.data)

    x1 = toy.x.clone().requires_grad_()
    x2 = toy.x.clone().requires_grad_()
    out1 = ours(x1, toy.graph.gat_index)
    out2 = ref(x2, toy.graph.gat_index)
    assert torch.allclose(out1, out2, atol=1e-5)

    w = torch.randn_like(out1)
    (out1 * w).sum().backward()
    (out2 * w).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(ours.lin.weight.grad, ref.lin.weight.grad, atol=1e-5)
    assert torch.allclose(ours.att_src.grad, ref.att_src.grad, atol=1e-5)
    assert torch.allclose(ours.att_dst.grad, ref.att_dst.grad, atol=1e-5)


def test_fused_attention_gradcheck(toy):
    """Gradients analytiques de l'attention fusionnée contre différences finies."""
    torch.manual_seed(0)
    g = toy.graph
    n, heads, ch = toy.num_nodes, 2, 3
    s = torch.randn(n, heads, dtype=torch.float64, requires_grad=True)
    t = torch.randn(n, heads, dtype=torch.float64, requires_grad=True)
    h = torch.randn(n, heads, ch, dtype=torch.float64, requires_grad=True)
    src, dst = g.gat_index
    assert torch.autograd.gradcheck(lambda s, t, h: gat_attention(s, t, h, src, dst, 0.2, budget=50), (s, t, h))


def test_gat_with_zero_attention_is_neighbourhood_mean(toy):
    """Avec des poids d'attention nuls, GAT fait la moyenne sur les voisins et soi."""
    conv = SparseGATConv(toy.x.size(1), 4, heads=2, concat=True)
    torch.nn.init.zeros_(conv.att_src)
    torch.nn.init.zeros_(conv.att_dst)
    out = conv(toy.x, toy.graph.gat_index)
    a = _dense_adjacency(toy.graph.edge_index, toy.num_nodes) + torch.eye(toy.num_nodes)
    ref = (a @ conv.lin(toy.x)) / a.sum(1, keepdim=True) + conv.bias
    assert torch.allclose(out, ref, atol=1e-5)


def test_symmetric_spmm_gradient_matches_dense(toy):
    """sym_spmm réutilise A au lieu de Aᵀ : exact parce que A est symétrique."""
    adj = toy.graph.adj
    x = torch.randn(toy.num_nodes, 4, requires_grad=True)
    scale = torch.rand(toy.num_nodes)
    out = models.sym_spmm(adj, x, scale)
    dense = adj.to_dense()
    xr = x.detach().clone().requires_grad_()
    ref = scale[:, None] * (dense @ xr)
    assert torch.allclose(out, ref, atol=1e-5)
    g = torch.randn_like(out)
    (gx,) = torch.autograd.grad(out, x, g)
    (rx,) = torch.autograd.grad(ref, xr, g)
    assert torch.allclose(gx, rx, atol=1e-5)


def test_mlp_ignores_the_graph(toy):
    model = build_model("mlp", toy.x.size(1), toy.num_classes, TrainConfig(hidden=16)).eval()
    assert torch.equal(model(toy.x, toy.graph), model(toy.x, None))
