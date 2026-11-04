# arxiv-gnn

Classer 169 343 articles d'informatique d'arXiv dans leurs 40 catégories, avec le texte seul puis avec le graphe de leurs citations, et mesurer ce que le graphe apporte sur le protocole officiel de l'Open Graph Benchmark.

Code : https://github.com/pinpin-http/arxiv-gnn

<!-- RESULTATS -->

## La question

Un article cite des articles proches de son sujet. Si l'on connaît déjà le texte d'un article (titre et résumé), que gagne-t-on à regarder aussi qui il cite et qui le cite ? Pour répondre, on compare sur les mêmes données :

- un **MLP**, qui ne voit que le texte de chaque article ;
- la **propagation d'étiquettes**, qui ne voit que le graphe et les étiquettes connues, sans apprentissage ;
- deux réseaux de neurones sur graphe, **GCN** et **GraphSAGE**, qui combinent le texte d'un article et celui de ses voisins ;
- **Correct & Smooth** (C&S), qui part des prédictions du MLP et les corrige en les diffusant le long des citations.

## Données

[ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) (Open Graph Benchmark), téléchargé par `make data` avec le package `ogb` dans `data/` (ignoré par git, environ 81 Mo compressés).

- **Graphe** : les citations entre les articles d'informatique d'arXiv indexés par Microsoft Academic Graph (MAG), 169 343 nœuds et 1 166 243 arêtes orientées.
- **Caractéristiques** : un vecteur de 128 dimensions par article, moyenne des plongements word2vec des mots du titre et du résumé.
- **Étiquettes** : 40 catégories (cs.AI, cs.LG, cs.CV…).
- **Découpage officiel par année** : entraînement sur les articles publiés jusqu'en 2017 (90 941), validation sur 2018 (29 799), test à partir de 2019 (48 603). Le test porte donc sur des articles plus récents que tout ce qui a servi à l'entraînement.
- **Licence** : ODC-BY (Open Data Commons Attribution), indiquée par OGB pour ogbn-arxiv sur la page du jeu de données. La source est MAG (Wang et al., « Microsoft Academic Graph: When experts are not enough », *Quantitative Science Studies*, 2020) ; la page OGB ne mentionne pas de licence distincte pour MAG.

Comme dans les exemples officiels d'OGB, le graphe est rendu symétrique (une citation relie les deux articles dans les deux sens) : 2 315 598 arêtes orientées après suppression des doublons, degré moyen 13,7, médian 6, maximum 13 161, aucun nœud isolé.

## Méthode

**Protocole.** Découpage et `Evaluator` officiels d'OGB (accuracy). Les étiquettes d'entraînement sont les seules à entrer dans la perte, dans la propagation d'étiquettes et dans C&S. La validation sert à choisir l'époque et la configuration. Le test n'est calculé qu'en campagne finale, et le chiffre rapporté est celui de l'évaluation qui maximise la validation (même règle que les exemples OGB). Des tests vérifient que remplacer toutes les étiquettes de test par du bruit ne change ni l'entraînement, ni l'époque retenue, ni la configuration retenue.

**Modèles.** Les trois réseaux reprennent l'architecture des exemples officiels d'OGB : 3 couches, BatchNorm, ReLU et dropout entre les couches, Adam avec un taux d'apprentissage de 0,01, entraînement plein graphe (tout le graphe à chaque époque).

| Méthode | Ce qu'elle voit | Implémentation |
|---|---|---|
| MLP | le texte de l'article | `torch.nn` |
| Propagation d'étiquettes | le graphe et les étiquettes d'entraînement | Y ← α·ÂY + (1 − α)·Y₀, Â = D^-1/2 A D^-1/2, 50 itérations ; vérifiée contre `LabelPropagation` de PyG |
| GCN | texte de l'article et de ses voisins | `GCNConv` de PyG sur Â = D^-1/2 (A + I) D^-1/2 en CSR |
| GraphSAGE | idem, agrégation par moyenne | `SAGEConv` de PyG (moyenne des voisins) |
| MLP + C&S | prédictions du MLP, graphe, étiquettes d'entraînement | `CorrectAndSmooth` de PyG, 50 itérations de correction et 50 de lissage |

**Même budget d'hyperparamètres pour tous** : 4 configurations par méthode, jugées sur la validation avec la graine 0.

- réseaux : largeur {128, 256} × dropout {0,25 ; 0,5}, 300 époques ;
- propagation d'étiquettes : α ∈ {0,5 ; 0,75 ; 0,9 ; 0,99} ;
- C&S : α de correction {0,5 ; 0,9} × α de lissage {0,6 ; 0,9}, sur les prédictions du MLP retenu.

**Graines et calcul.** La configuration retenue est relancée avec les graines 0, 1 et 2 ; on rapporte la moyenne et l'écart-type non biaisé, comme OGB. La validation est évaluée toutes les 5 époques. Tous les résultats publiés ici sont calculés sur CPU, avec 2 threads (`torch.set_num_threads(2)`) dans une portée systemd plafonnée à 1,25 cœur (`CPUQuota=125%`). Le code accepte `--device cuda` (précision mixte float16 avec `torch.amp` et `GradScaler`), mais les résultats publiés sont calculés sur CPU, parce que le GPU de la machine de développement n'était pas stable sous charge CUDA.

**Détails d'implémentation.**

- Matrices d'adjacence en CSR avec indices int32 (`torch.sparse_csr_tensor`), une multiplication creuse par couche.
- Comme le graphe est symétrique, la rétropropagation de Â·X réutilise Â au lieu de calculer Âᵀ à chaque pas (`sym_spmm`, testé contre un calcul dense).
- GAT est implémenté (`SparseGATConv`) avec une attention fusionnée qui ne garde aucun tenseur par arête, testé contre `GATConv` de PyG (sorties et gradients) et par différences finies. Il n'a pas été entraîné : voir plus bas.

## État

Expérience en cours. Les campagnes MLP, propagation d'étiquettes, C&S et GCN sont presque finies, celle de GraphSAGE vient de commencer, et les résultats seront publiés ici quand tout aura tourné. [`REPRISE.md`](REPRISE.md) dit où elle s'est arrêtée et comment la finir.

## Relancer

```bash
make install          # uv sync --frozen --extra cpu (torch CPU, dépendances épinglées)
make data             # téléchargement d'ogbn-arxiv dans data/
make train MODEL=sage # recherche d'hyperparamètres puis graines 0, 1, 2 (MODEL : mlp, lp, gcn, sage, cs)
make experiments      # toutes les méthodes, dans l'ordre lp, mlp, cs, gcn, sage
make bench            # temps par époque et RAM maximale sur CPU
make figures          # PNG dans docs/
make report           # tableau results/summary.md
make test             # tests pytest sur petits graphes synthétiques
```

`make train MODEL=cs` suppose que le MLP a déjà tourné. Les campagnes reprennent là où elles s'étaient arrêtées : chaque essai et chaque graine sont écrits dans `results/` dès qu'ils finissent. Pour le GPU : `make install TORCH=cu128` puis `make train MODEL=sage DEVICE=cuda`.

## Organisation du code

```
src/arxiv_gnn/
  data.py       chargement OGB, symétrisation, matrices CSR, graphe synthétique des tests
  models.py     MLP, GCN, GraphSAGE, GAT (attention fusionnée), propagation d'étiquettes, C&S
  train.py      boucle d'entraînement, recherche sur la validation, campagnes multi-graines
  evaluate.py   Evaluator OGB, moyenne et écart-type
  bench.py      temps par époque et RAM maximale
  figures.py    figures de docs/
  report.py     tableau récapitulatif
  reference.py  chiffres publiés du classement OGB
tests/          formes de sortie, GCN contre calcul dense, GAT contre PyG, graines, fuite de test, propagation
```

## Licence

Code sous licence MIT (voir `LICENSE`). Les données ogbn-arxiv restent sous licence ODC-BY.
