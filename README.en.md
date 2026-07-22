# SemiTNet Paper Reproduction

**Language:** [فارسی](README.md) | **English**

This repository prepares a reproducible workflow for:

> **A Semi-Supervised Transformer-Based Deep Learning Framework for Automated Tooth Segmentation and Identification on Panoramic Radiographs**  
> *Diagnostics*, 2024, 14(17), 1948 — DOI `10.3390/diagnostics14171948`

The public author link now redirects to the later SemiT-SAM repository. That repository still contains the SemiTNet teacher/student training path, but its current branch includes private server paths, forced debug overrides, a 52-class challenge configuration, and launch scripts tied to eight GPUs. This wrapper pins the upstream commit and applies reproducibility patches automatically.

## Smoke run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
make smoke
```

The smoke run is a CPU-safe, synthetic teacher/student Transformer segmentation test. Its metrics are software-validation results, not paper results.

## Official checkpoint inference

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

## Full teacher/student training

```bash
make full
```

The paper reports eight V100 32GB GPUs, a total batch size of 16, 26,250 iterations, AdamW at `1e-4`, and learning-rate drops at iterations 24,000 and 25,000. Full retraining is a GPU workload; released-checkpoint inference is the preferred first reproduction target.

See [docs/RUNBOOK.en.md](docs/RUNBOOK.en.md) for the detailed procedure.

## Initialization note

The released SemiTNet checkpoint is intended for evaluation and is the closest route to the published final outputs. Clean full retraining defaults to no model weights unless a compatible TinyViT/Detectron2 initialization is supplied with `--init-weights`; using the released final checkpoint as training initialization would leak the target solution.
