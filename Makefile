# FinNet-GNN — developer entry points
#
# By default this uses the Python on PATH. To pin a specific interpreter:
#   make install PYTHON=/path/to/venv/bin/python
#
# All targets run from this directory and write to ./artifacts and ./data.

PYTHON ?= python3
FINNET ?= $(PYTHON) -m finnet
CONFIG ?= configs/default.yaml
CHECKPOINT ?= artifacts/risk_gnn.pt
DATA_DIR ?= data

# Gateway ports, overridable: make serve PORT=9000 METRICS_PORT=9102
PORT ?= 8583
METRICS_PORT ?= 9101

# Training knobs
EVENTS ?= 12000
EPOCHS ?= 12
BENCH_ITERATIONS ?= 2000
BENCH_EVENTS ?= 8000

.DEFAULT_GOAL := help
.PHONY: help install install-dev test lint train train-upgraded train-ieee serve serve-prod \
        simulate demo demo-retrain baseline-run calibrate benchmark download-data gen-data info report clean distclean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "FinNet-GNN targets:\n\n"} \
	      /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package (runtime dependencies only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install the package with dev/test extras
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the full test suite
	$(PYTHON) -m pytest -q

lint: ## Static checks (ruff must be installed)
	$(PYTHON) -m ruff check src tests

train: ## Train the default synthetic checkpoint
	$(FINNET) train --config $(CONFIG) --source synthetic \
		--events $(EVENTS) --epochs $(EPOCHS) --output $(CHECKPOINT)

train-ieee: ## Train on IEEE-CIS (requires data/ and Kaggle credentials)
	$(FINNET) train --config $(CONFIG) --source ieee \
		--transactions $(DATA_DIR)/train_transaction.csv \
		--identity $(DATA_DIR)/train_identity.csv \
		--epochs $(EPOCHS) --output artifacts/risk_gnn_ieee.pt

serve: ## Run the gateway in simulator/test mode (trusts ISO field 61 subnet)
	$(FINNET) serve --config $(CONFIG) --port $(PORT) --metrics-port $(METRICS_PORT) \
		--model $(CHECKPOINT) --trust-client-subnet

serve-prod: ## Run the gateway using the real TCP peer address (no field-61 trust)
	$(FINNET) serve --config $(CONFIG) --port $(PORT) --metrics-port $(METRICS_PORT) \
		--model $(CHECKPOINT)

simulate: ## Drive legitimate + carding + botnet traffic against a running gateway
	$(FINNET) simulate --port $(PORT) --events 900 --fraud-share 0.6 \
		--json-out artifacts/sim_report.json

demo: ## Train if needed, start an ephemeral gateway, and simulate an attack
	$(FINNET) demo --config $(CONFIG) --events 450

demo-retrain: ## Force a fresh demo model before simulating
	$(FINNET) demo --config $(CONFIG) --events 450 --retrain

train-upgraded: ## Train the risk-gated Time2Vec GNN
	$(FINNET) train --config $(CONFIG) --source synthetic \
		--events $(EVENTS) --epochs $(EPOCHS) --risk-gate --temporal-encoding --time-dim 16 \
		--output artifacts/risk_gnn_v2.pt

baseline-run: ## Compare classical baselines and the GNN
	$(FINNET) baseline-run --config $(CONFIG) --source synthetic \
		--events $(EVENTS) --model artifacts/risk_gnn_v2.pt \
		--json-out artifacts/baseline_report.json

calibrate: ## Fit a JSON calibrator for a checkpoint
	$(FINNET) calibrate --config $(CONFIG) --source synthetic \
		--events $(EVENTS) --model $(CHECKPOINT) \
		--method isotonic --output artifacts/calibrator.json

benchmark: ## Measure graph export + GNN scoring latency
	$(FINNET) benchmark --config $(CONFIG) --model $(CHECKPOINT) \
		--iterations $(BENCH_ITERATIONS) --events $(BENCH_EVENTS) \
		--json-out artifacts/benchmark.json

download-data: ## Download and extract IEEE-CIS via the Kaggle CLI
	$(FINNET) download-data --dest $(DATA_DIR)

gen-data: ## Write a reproducible synthetic transaction CSV
	$(FINNET) gen-data --events $(EVENTS) --out $(DATA_DIR)/synthetic_transactions.csv

info: ## Print effective configuration and checkpoint status
	$(FINNET) info --config $(CONFIG) --model $(CHECKPOINT)

report: ## Print the latest checkpoint training report
	@$(PYTHON) -c "import json,pathlib; p=pathlib.Path('$(CHECKPOINT)').with_suffix('.report.json'); \
		print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else 'no report found')"

clean: ## Remove caches and generated reports/logs (keeps checkpoints and data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f artifacts/decisions.jsonl artifacts/firewall.jsonl \
	      artifacts/sim_report.json artifacts/benchmark.json

distclean: clean ## Also remove trained checkpoints, scores and generated data
	rm -f artifacts/*.pt artifacts/*.npz artifacts/*.report.json
	rm -f $(DATA_DIR)/synthetic_transactions.csv
