# Reprendre l'expérience

L'expérience a été arrêtée le 4 novembre 2026 vers 16 h, en pleine campagne GraphSAGE, faute de temps. Ce fichier dit où elle en est et comment la finir.

## Où elle en est

| Méthode | Recherche (4 configurations, graine 0) | Graines finales (0, 1, 2) |
|---|---|---|
| Propagation d'étiquettes | faite | faite |
| MLP | faite | faites (0, 1, 2) |
| MLP + C&S | faite | faite |
| GCN | faite | 0 et 1 faites, **2 à lancer** |
| GraphSAGE | **essai 0 fait, essais 1 à 3 à lancer** | **à lancer** |
| GAT | implémenté et testé, jamais entraîné (trop lent sur CPU) | |

Les résultats déjà calculés sont dans `results/search/` et `results/final/`, un fichier JSON par essai et par graine. Ils ne sont **pas dans git** (comme les figures provisoires de `docs/`), pour ne rien publier avant la fin. Si tu changes de machine, copie le dossier `results/` avec le reste.

## Finir

Sur la même machine, depuis `~/Projects/arxiv-gnn` :

```bash
make install                 # si le venv a disparu
make data                    # seulement si data/ a disparu
CUDA_VISIBLE_DEVICES="" make experiments   # reprend où ça s'est arrêté : GCN graine 2, puis GraphSAGE
make bench figures report
```

`make experiments` saute tout essai et toute graine déjà écrits avec la même configuration. Pour aller deux fois plus vite, on peut couper en deux processus, comme le jour de l'arrêt :

```bash
# processus A : la dernière graine du GCN, puis la graine 2 de GraphSAGE une fois sa recherche finie
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m arxiv_gnn train --model gcn
until [ -f results/search/sage.json ]; do sleep 30; done
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m arxiv_gnn train --model sage --seed 2 --no-search
# processus B : la recherche GraphSAGE, puis ses graines 0 et 1
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m arxiv_gnn train --model sage --seed 0 --seed 1
```

Sur ce PC, chaque processus tournait avec 2 threads dans une portée plafonnée :
`systemd-run --user --scope -q --unit=heavy-gnn-$(date +%s) -p CPUQuota=200% -p MemoryMax=6G <commande>`.

Durées observées avec 2 threads : 11 à 22 min par essai de recherche, 17 à 22 min par graine finale. Il restait environ 1 h 30 de calcul au moment de l'arrêt, puis le benchmark et les figures.

## Ensuite

1. Remplir les sections Résultats, Coût du calcul sur CPU, Représentations apprises et Limites du README à partir de `results/summary.md` (`make report`) et des figures de `docs/` ; enlever la section « État ».
2. Comparer honnêtement au classement OGB avec les chiffres de `src/arxiv_gnn/reference.py`, et ne garder que ceux dont la source est vérifiée.
3. Relire les textes une fois, sans changer les chiffres.
4. Commiter `results/` (petits JSON) et `docs/`, puis pousser.

## À savoir

- **Pas de GPU sur le PC de développement.** Deux coupures brutales du PC ont suivi des charges CUDA sur la MX450 : garder `CUDA_VISIBLE_DEVICES=""`. Le code accepte `--device cuda` pour une autre machine (`make install TORCH=cu128`).
- **Temps par époque.** Les premières campagnes tournaient à 1,25 cœur (`CPUQuota=125%`), les dernières à 2 cœurs, et une sonde thermique abaissait parfois le plafond à 25 % au-dessus de 93 °C (le champ `throttled` de chaque JSON compte ces moments). Les temps `epoch_s` des JSON ne sont donc pas comparables entre eux : pour le README, prendre ceux de `make bench`, mesurés dans les mêmes conditions pour tous les modèles.
- Le README annonce encore « plafonnée à 1,25 cœur » : à corriger selon les conditions finales.
