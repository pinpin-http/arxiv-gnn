import pytest
import torch

from arxiv_gnn import config as C
from arxiv_gnn import train as T
from arxiv_gnn.config import TrainConfig
from arxiv_gnn.data import NodeData

SMALL = TrainConfig(hidden=16, num_layers=2, dropout=0.5, lr=0.01, epochs=15, heads=2)


def _with_scrambled_test_labels(data: NodeData, seed: int = 99) -> NodeData:
    y = data.y.clone()
    test = data.split["test"]
    gen = torch.Generator().manual_seed(seed)
    y[test] = torch.randint(0, data.num_classes, (test.numel(),), generator=gen)
    assert not torch.equal(y, data.y)
    return NodeData(x=data.x, y=y, split=data.split, graph=data.graph, num_classes=data.num_classes)


@pytest.mark.parametrize("name", ["mlp", "gcn", "sage", "gat"])
def test_same_seed_same_run(toy, name):
    r1, _ = T.run_training(name, SMALL, toy, seed=3)
    r2, _ = T.run_training(name, SMALL, toy, seed=3)
    assert r1.history == r2.history
    assert (r1.best_epoch, r1.valid_acc, r1.test_acc) == (r2.best_epoch, r2.valid_acc, r2.test_acc)


def test_different_seeds_differ(toy):
    r1, _ = T.run_training("gcn", SMALL, toy, seed=0)
    r2, _ = T.run_training("gcn", SMALL, toy, seed=1)
    assert r1.history["loss"] != r2.history["loss"]


@pytest.mark.parametrize("name", ["mlp", "gcn", "sage", "gat"])
def test_test_labels_do_not_influence_training_or_epoch_choice(toy, name):
    """Si l'on remplace toutes les étiquettes de test par du bruit, la
    trajectoire d'entraînement, l'époque retenue et le modèle retenu sont identiques."""
    r1, s1 = T.run_training(name, SMALL, toy, seed=0)
    r2, s2 = T.run_training(name, SMALL, _with_scrambled_test_labels(toy), seed=0)
    assert r1.history["loss"] == r2.history["loss"]
    assert r1.history["valid"] == r2.history["valid"]
    assert r1.best_epoch == r2.best_epoch
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def _isolate(monkeypatch, path):
    monkeypatch.setattr(C, "RESULTS_DIR", path / "results")
    monkeypatch.setattr(C, "CHECKPOINT_DIR", path / "checkpoints")


def test_search_never_evaluates_test_and_ignores_test_labels(toy, tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path / "a")
    asked = []
    real = T.split_accuracies

    def spy(pred, y, split, names=("train", "valid", "test")):
        asked.extend(names)
        return real(pred, y, split, names)

    monkeypatch.setattr(T, "split_accuracies", spy)
    configs = [TrainConfig(**{**SMALL.to_dict(), "hidden": h, "lr": lr}) for h in (8, 16) for lr in (0.005, 0.02)]
    s1 = T.search_neural("gcn", toy, configs=configs, log=lambda *_: None)
    assert asked and "test" not in asked
    assert T.best_config("gcn").to_dict() == s1["best"]["config"]
    _isolate(monkeypatch, tmp_path / "b")
    s2 = T.search_neural("gcn", _with_scrambled_test_labels(toy), configs=configs, log=lambda *_: None)
    assert s1["best"]["config"] == s2["best"]["config"]
    assert [t["valid_acc"] for t in s1["trials"]] == [t["valid_acc"] for t in s2["trials"]]


def test_final_reuses_search_run_for_seed_zero(toy, tmp_path, monkeypatch):
    """La graine 0 de la campagne finale reprend l'essai retenu par la recherche :
    le résultat doit être celui d'un entraînement refait de zéro."""
    _isolate(monkeypatch, tmp_path)
    configs = [TrainConfig(**{**SMALL.to_dict(), "hidden": h}) for h in (8, 16)]
    search = T.search_neural("sage", toy, configs=configs, log=lambda *_: None)
    final = T.final_neural("sage", toy, seeds=(0, 1), log=lambda *_: None)
    run0 = final["runs"][0]
    assert run0["reused_from_search"] and run0["config"] == search["best"]["config"]
    fresh, _ = T.run_training("sage", TrainConfig.from_dict(run0["config"]), toy, seed=0)
    assert (fresh.best_epoch, fresh.valid_acc, fresh.test_acc) == (run0["best_epoch"], run0["valid_acc"], run0["test_acc"])
    # Une relance reprend les graines déjà écrites sur disque.
    again = T.final_neural("sage", toy, seeds=(0, 1), log=lambda *_: None)
    assert again["test_mean"] == final["test_mean"]


def test_training_learns_on_homophilous_graph(toy):
    cfg = TrainConfig(hidden=32, num_layers=2, dropout=0.2, lr=0.02, epochs=60)
    res, _ = T.run_training("gcn", cfg, toy, seed=0)
    assert res.history["loss"][-1] < res.history["loss"][0]
    assert res.valid_acc > 0.6  # 3 classes : le hasard ferait environ 0,33


def test_train_loss_only_uses_train_nodes(toy, monkeypatch):
    seen = {}
    real = torch.nn.functional.cross_entropy

    def spy(logits, target, *a, **kw):
        seen["n"] = logits.size(0)
        return real(logits, target, *a, **kw)

    monkeypatch.setattr(T.F, "cross_entropy", spy)
    T.run_training("mlp", TrainConfig(hidden=8, num_layers=2, epochs=1), toy, seed=0)
    assert seen["n"] == toy.split["train"].numel()
