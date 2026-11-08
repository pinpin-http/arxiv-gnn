# Calcul sur CPU, limité à 2 threads pour cohabiter avec d'autres tâches.
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export ARXIV_GNN_THREADS ?= 2
export MPLBACKEND = Agg
export OUTDATED_IGNORE = 1

# Extra uv : cpu (défaut, celui de la CI), cu128 (NVIDIA) ou rocm (AMD, ROCm 7.2).
TORCH ?= cpu
RUN = uv run --extra $(TORCH) python -m arxiv_gnn
MODEL ?= sage
DEVICE ?= cpu
# auto : précision mixte float16 sur GPU, float32 sur CPU ; fp32 ou fp16 pour forcer.
PRECISION ?= auto
# Ordre imposé : C&S part des prédictions du MLP.
MODELS = lp mlp cs gcn sage

.PHONY: install data search train experiments bench figures report test all clean

install:
	uv sync --frozen --extra $(TORCH)

data:
	$(RUN) data

search:
	$(RUN) search --model $(MODEL) --device $(DEVICE) --precision $(PRECISION)

train:
	$(RUN) train --model $(MODEL) --device $(DEVICE) --precision $(PRECISION)

experiments:
	for m in $(MODELS); do $(RUN) train --model $$m --device $(DEVICE) --precision $(PRECISION) || exit 1; done

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
