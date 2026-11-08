"""Chiffres publiés sur le classement OGB d'ogbn-arxiv, relevés le 7 novembre 2026.

Source : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv
Accuracy moyenne ± écart-type sur 10 graines, telles qu'affichées sur la page.
Le classement n'a pas de ligne « GAT » simple : toutes ses entrées GAT sont des
variantes plus grosses (1,4 à 1,6 million de paramètres pour les plus simples)
avec des astuces en plus, d'où l'absence de GAT dans LEADERBOARD.
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
    "gat_cs": {"entry": "GAT + C&S (2020, 1 567 000 paramètres)", "test": (0.7386, 0.0014)},
    # Bases linéaires de C&S : 5 160 = 128 × 40 + 40 (caractéristiques seules),
    # 15 400 = 384 × 40 + 40 (caractéristiques et 256 dimensions de plongements en plus).
    "plain_linear_cs": {"entry": "Plain Linear + C&S (2020, 5 160 paramètres)", "test": (0.7126, 0.0001)},
    "linear_cs": {"entry": "Linear + C&S (2020, 15 400 paramètres)", "test": (0.7222, 0.0002)},
    # Représentations GIANT-XRT, apprises sur le texte brut en prédisant le voisinage
    # de chaque article dans le graphe (citées dans les limites du README).
    "giant_mlp": {"entry": "GIANT-XRT+MLP (use raw text) (2021, 273 960 paramètres)", "test": (0.7306, 0.0011)},
    "giant_sage": {"entry": "GIANT-XRT+GraphSAGE (use raw text) (2021, 546 344 paramètres)", "test": (0.7435, 0.0014)},
}
