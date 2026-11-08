| Méthode | Configuration retenue (validation) | Paramètres | Validation (%) | Test (%) | Classement OGB, test (%) | s / époque (CPU, 2 threads) | RAM max (Go) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP (sans graphe) | 3 couches, 256 neurones, dropout 0,5, 300 époques | 110 120 | 57,81 ± 0,19 | 55,66 ± 0,16 | 55,50 ± 0,23 | 2,02 | 2,06 |
| Propagation d'étiquettes | alpha = 0,75, 50 itérations | 0 | 69,86 ± 0,00 | 67,60 ± 0,00 | 68,32 ± 0,00 | 2,1 (calcul complet) | 0,91 |
| GCN | 3 couches, 256 neurones, dropout 0,5, 300 époques | 110 120 | 72,75 ± 0,10 | 71,50 ± 0,27 | 71,74 ± 0,29 | 3,10 | 2,14 |
| GraphSAGE | 3 couches, 256 neurones, dropout 0,5, 300 époques | 218 664 | 72,99 ± 0,24 | 71,90 ± 0,27 | 71,49 ± 0,27 | 4,02 | 2,83 |
| GAT (entraîné sur GPU) | 3 couches, 256 neurones, dropout 0,5, 300 époques | 142 184 | 72,99 ± 0,05 | 71,74 ± 0,12 |  | 16,51 | 3,19 |
| MLP + C&S | alpha correction = 0,9, alpha lissage = 0,6 | 110 120 | 73,01 ± 0,02 | 71,16 ± 0,01 | 73,12 ± 0,12 | 4,0 (calcul complet) | 0,91 |

Graines : 0, 1, 2. Classement : https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv

Contrôle CPU/GPU : mêmes modèles, même recherche (4 configurations, graine 0), mêmes graines, relancés sur GPU (results/gpu_check/).

| Méthode | Configuration retenue, CPU | Configuration retenue, GPU | Validation CPU (%) | Validation GPU (%) | Test CPU (%) | Test GPU (%) | Écart de test, GPU moins CPU (points) | Précision GPU | s / époque sur GPU (médiane) |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| GCN | 3 couches, 256 neurones, dropout 0,5, 300 époques | identique | 72,75 ± 0,10 | 72,93 ± 0,21 | 71,50 ± 0,27 | 71,92 ± 0,18 | +0,42 | fp32 | 0,84 |
| GraphSAGE | 3 couches, 256 neurones, dropout 0,5, 300 époques | identique | 72,99 ± 0,24 | 72,90 ± 0,20 | 71,90 ± 0,27 | 71,53 ± 0,39 | -0,37 | fp32 | 1,00 |

GAT, entraîné sur GPU seulement (fp32) : 1,60 s par époque (médiane des graines).

Recherche d'hyperparamètres (graine 0, accuracy de validation) :

- MLP (sans graphe) : 3 couches, 128 neurones, dropout 0,25, 300 époques : 57,14 ; 3 couches, 128 neurones, dropout 0,5, 300 époques : 56,01 ; 3 couches, 256 neurones, dropout 0,25, 300 époques : 57,57 ; 3 couches, 256 neurones, dropout 0,5, 300 époques : 57,93
- Propagation d'étiquettes : alpha = 0,5, 50 itérations : 69,77 ; alpha = 0,75, 50 itérations : 69,86 ; alpha = 0,9, 50 itérations : 68,23 ; alpha = 0,99, 50 itérations : 51,52
- GCN : 3 couches, 128 neurones, dropout 0,25, 300 époques : 72,65 ; 3 couches, 128 neurones, dropout 0,5, 300 époques : 72,20 ; 3 couches, 256 neurones, dropout 0,25, 300 époques : 72,55 ; 3 couches, 256 neurones, dropout 0,5, 300 époques : 72,85
- GraphSAGE : 3 couches, 128 neurones, dropout 0,25, 300 époques : 72,11 ; 3 couches, 128 neurones, dropout 0,5, 300 époques : 72,23 ; 3 couches, 256 neurones, dropout 0,25, 300 époques : 71,75 ; 3 couches, 256 neurones, dropout 0,5, 300 époques : 72,87
- GAT : 3 couches, 128 neurones, dropout 0,25, 300 époques : 72,59 ; 3 couches, 128 neurones, dropout 0,5, 300 époques : 72,51 ; 3 couches, 256 neurones, dropout 0,25, 300 époques : 72,76 ; 3 couches, 256 neurones, dropout 0,5, 300 époques : 72,94
- MLP + C&S : alpha correction = 0,5, alpha lissage = 0,6 : 71,87 ; alpha correction = 0,5, alpha lissage = 0,9 : 72,06 ; alpha correction = 0,9, alpha lissage = 0,6 : 73,00 ; alpha correction = 0,9, alpha lissage = 0,9 : 72,17
