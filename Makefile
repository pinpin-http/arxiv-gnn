# Calcul sur CPU, limité à 2 threads pour cohabiter avec d'autres tâches.
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export ARXIV_GNN_THREADS ?= 2
export MPLBACKEND = Agg
export OUTDATED_IGNORE = 1

# Extra uv : cpu (défaut, celui des résultats publiés et de la CI) ou cu128.
TORCH ?= cpu
RUN = uv run --extra $(TORCH) python -m arxiv_gnn
MODEL ?= sage
DEVICE ?= cpu
# Ordre imposé : C&S part des prédictions du MLP.
MODELS = lp mlp cs gcn sage

.PHONY: install data search train experiments bench figures report test all clean

install:
	uv sync --frozen --extra $(TORCH)

data:
	$(RUN) data

search:
	$(RUN) search --model $(MODEL) --device $(DEVICE)

train:
	$(RUN) train --model $(MODEL) --device $(DEVICE)

experiments:
	for m in $(MODELS); do $(RUN) train --model $$m --device $(DEVICE) || exit 1; done

bench:
	$(RUN) bench

figures:
	$(RUN) figures

report:
	$(RUN) report

test:
	uv run --extra $(TORCH) pytest

all: data experiments bench figures report

clean:
	rm -rf results/search results/final checkpoints
