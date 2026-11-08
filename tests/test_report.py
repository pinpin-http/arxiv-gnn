import json

from arxiv_gnn import config as C
from arxiv_gnn import report

CFG = {"hidden": 256, "num_layers": 3, "dropout": 0.5, "lr": 0.01, "weight_decay": 0.0, "epochs": 300}


def _final(valid, test, device=None):
    run = {"seed": 0, "valid_acc": valid, "test_acc": test, "epoch_s": 0.8}
    if device:
        run.update(device=device, precision="fp32")
    return {"model": "gcn", "config": CFG, "seeds": [0], "params": 10, "valid_mean": valid, "valid_std": 0.001,
            "test_mean": test, "test_std": 0.002, "runs": [run]}


def test_gpu_check_section(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "gcn.json").write_text(json.dumps(_final(0.7275, 0.7150)))
    assert report.gpu_check_lines() == []  # pas de contrôle GPU : pas de section

    (tmp_path / "gpu_check" / "final").mkdir(parents=True)
    (tmp_path / "gpu_check" / "final" / "gcn.json").write_text(json.dumps(_final(0.7293, 0.7192, "cuda")))
    text = report.write_report()
    row = next(line for line in text.splitlines() if line.startswith("| GCN |") and "identique" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "identique"
    assert cells[3:7] == ["72,75 ± 0,10", "72,93 ± 0,10", "71,50 ± 0,20", "71,92 ± 0,20"]
    assert cells[7] == "+0,42" and cells[8] == "fp32"
    assert "GraphSAGE" not in text  # pas de résultat GraphSAGE : pas de ligne
