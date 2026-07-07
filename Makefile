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

train: _ensure_venv
	@echo "[Train] Starting frugal training..."
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