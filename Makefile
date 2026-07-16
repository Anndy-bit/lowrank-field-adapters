# S³ — Spectral-Spatial-Smooth Fine-Tuning
# Complete workflow from SVD to benchmarks

.PHONY: help venv svd train evaluate all clean test

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

MODEL ?= Qwen/Qwen2.5-7B-Instruct
K ?= 128
CONFIG ?= experiments/configs/s3_standard.yaml
DEVICE ?= cuda:0

help:
	@echo "S³ Fine-Tuning Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make venv          Create virtual environment + install dependencies"
	@echo "  make svd           Pre-compute randomized SVD (~2 min CPU)"
	@echo "  make train         Train S³ on Alpaca (GTX 1050)"
	@echo "  make evaluate      Run benchmarks on trained model"
	@echo "  make all           Full pipeline: venv → svd → train → evaluate"
	@echo "  make clean         Remove checkpoints and results"
	@echo "  make test          Run unit tests"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL              HuggingFace model (default: Qwen/Qwen2.5-7B-Instruct)"
	@echo "  K                  SVD rank (default: 128)"
	@echo "  CONFIG             YAML config (default: experiments/configs/s3_standard.yaml)"
	@echo "  DEVICE             GPU device (default: cuda:0)"
	@echo ""
	@echo "Ablation examples:"
	@echo "  make train-ablation ABLATION=s3_no_stb"
	@echo "  make train-ablation ABLATION=svmo_only"

venv:
	@echo "[Venv] Creating virtual environment..."
	@if [ ! -d $(VENV) ]; then \
		python3 -m venv $(VENV); \
		echo "[Venv] Created $(VENV)"; \
	else \
		echo "[Venv] Already exists: $(VENV)"; \
	fi
	@echo "[Venv] Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "[Venv] Ready."

_ensure_venv:
	@if [ ! -d $(VENV) ]; then \
		echo "[Error] Virtual environment not found. Run: make venv"; \
		exit 1; \
	fi

svd: _ensure_venv
	@echo "[SVD] Computing randomized SVD for $(MODEL) (k=$(K))..."
	$(PYTHON) src/utils/randomized_svd.py \
		--model $(MODEL) \
		--k $(K) \
		--output ./svd_factors/ \
		--device cpu

EPOCHS ?= 3
MAXSEQ ?= 256
LIMIT ?= 50
BATCH ?= 1

smoke: _ensure_venv
	@echo "[Smoke] Self-testing the S³ pipeline on a tiny model (no download)..."
	$(PYTHON) run_s3_train.py --smoke

# Short real-data run on the GPU to confirm the loss actually drops (do this
# BEFORE committing hours to the full run). LIMIT=50 examples by default.
train-test: _ensure_venv
	@echo "[Train-Test] Short S³ streaming run on $(LIMIT) real examples ($(DEVICE))..."
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) run_s3_train.py \
		--model $(MODEL) --svd_dir ./svd_factors --device $(DEVICE) \
		--mode stream --epochs 1 --limit $(LIMIT) --max_seq $(MAXSEQ) --batch $(BATCH)

train: _ensure_venv
	@echo "[Train] Starting S³ streaming training (7B on <2GB VRAM)..."
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) run_s3_train.py \
		--model $(MODEL) --svd_dir ./svd_factors --device $(DEVICE) \
		--mode stream --epochs $(EPOCHS) --limit $(LIMIT) --max_seq $(MAXSEQ) --batch $(BATCH)

# RQ4 ablations: full / -STB / -NMF / SVMO-only / NMF-only, one comparison table.
# ABLIMIT examples per variant (default 150), ABATCH=4. ~5.6h for 5 variants @ 150.
ABLIMIT ?= 150
ABATCH ?= 4
ablations: _ensure_venv
	@echo "[Ablations] Running operator ablations on $(ABLIMIT) examples each (batch=$(ABATCH))..."
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) run_ablations.py \
		--model $(MODEL) --svd_dir ./svd_factors --device $(DEVICE) \
		--limit $(ABLIMIT) --max_seq $(MAXSEQ) --batch $(ABATCH)

# Deprecated path kept for reference only (known-broken forward/backward).
train-legacy: _ensure_venv
	@echo "[Train] LEGACY frugal_trainer path (deprecated)..."
	$(PYTHON) train_s3.py \
		--config $(CONFIG) \
		--device $(DEVICE)

ABLATION ?= s3_full
train-ablation: _ensure_venv
	@echo "[Train] Ablation: $(ABLATION)..."
	$(PYTHON) train_s3.py \
		--config experiments/configs/s3_standard.yaml \
		--device $(DEVICE) \
		--ablation $(ABLATION)

CKPT ?= ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt
evaluate: _ensure_venv
	@echo "[Eval] Running benchmarks..."
	$(PYTHON) train_s3.py \
		--config $(CONFIG) \
		--device $(DEVICE) \
		--eval_only \
		--checkpoint $(CKPT)

all: venv svd train evaluate
	@echo "[S³] Full pipeline complete."

test: _ensure_venv
	@echo "[Test] Running unit tests..."
	$(VENV)/bin/pytest tests/ -v --tb=short

clean:
	@echo "[Clean] Removing checkpoints and results..."
	rm -rf ./checkpoints/ ./results/

setup: venv svd
	@echo "[Setup] Ready for training."