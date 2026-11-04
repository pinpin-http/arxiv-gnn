import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch

from arxiv_gnn.data import synthetic_graph

torch.set_num_threads(2)


@pytest.fixture
def toy():
    """Graphe à blocs de 90 nœuds, 3 classes, sur CPU."""
    return synthetic_graph(num_nodes=90, num_classes=3, feat_dim=8, seed=0)
