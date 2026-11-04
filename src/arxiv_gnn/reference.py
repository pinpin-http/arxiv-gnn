"""Chiffres publiés sur le classement OGB d'ogbn-arxiv, relevés le 4 novembre 2026.

Source : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
Accuracy moyenne ± écart-type sur 10 graines, telles qu'affichées sur la page.
"""

LEADERBOARD_URL = "https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv"

# Entrées qui correspondent à nos modèles (même architecture, même entrée).
LEADERBOARD = {
    "mlp": {"entry": "MLP (équipe OGB, 2020)", "test": (0.5550, 0.0023), "valid": (0.5765, 0.0012), "params": 110_120},
    "lp": {"entry": "Label Propagation (Horace He, 2020)", "test": (0.6832, 0.0000), "valid": (0.7014, 0.0000), "params": 0},
    "gcn": {"entry": "GCN (équipe OGB, 2020)", "test": (0.7174, 0.0029), "valid": (0.7300, 0.0017), "params": 110_120},
    "sage": {"entry": "GraphSAGE (équipe OGB, 2020)", "test": (0.7149, 0.0027), "valid": (0.7277, 0.0016), "params": 218_664},
    "cs": {"entry": "MLP + C&S (Huang et al., 2020)", "test": (0.7312, 0.0012), "valid": (0.7391, 0.0015), "params": 175_656},
}

# Entrées voisines, citées dans le README mais pas directement comparables.
RELATED = {
    "gcn_tuned": {"entry": "GCN (Luo et al., 2024, 1 463 336 paramètres)", "test": (0.7360, 0.0018)},
    "sage_tuned": {"entry": "GraphSAGE (Luo et al., 2024, 1 727 272 paramètres)", "test": (0.7295, 0.0031)},
    "gat_flag": {"entry": "GAT+FLAG (2020, 1 628 440 paramètres)", "test": (0.7371, 0.0013)},
    "gat_cs": {"entry": "GAT + C&S (2020)", "test": (0.7386, 0.0014)},
}
