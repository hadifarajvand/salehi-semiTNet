# SemiTNet Implementation and Simulation

**Language:** [فارسی](README.md) | **English**

This repository provides a complete workflow for training, evaluating, and running tooth segmentation and identification models on panoramic radiographs, based on the method described in:

> **A Semi-Supervised Transformer-Based Deep Learning Framework for Automated Tooth Segmentation and Identification on Panoramic Radiographs**  
> *Diagnostics*, 2024, 14(17), 1948 — DOI `10.3390/diagnostics14171948`

The project includes dataset preparation, CPU smoke validation, teacher/student training, evaluation, metric calculation, result visualization, and full experiment launchers.

## Smoke run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
make smoke
```

The smoke run is a CPU-safe synthetic teacher/student Transformer segmentation test. Its outputs validate the software pipeline and are separate from the full TSI15k experiment.

## Main dataset execution

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

The reference experiment uses eight V100 32GB GPUs, a total batch size of 16, 26,250 iterations, AdamW at `1e-4`, and learning-rate drops at iterations 24,000 and 25,000.

See [docs/RUNBOOK.en.md](docs/RUNBOOK.en.md) for the detailed procedure.

Third-party software dependencies and license notices are documented in `docs/THIRD_PARTY_NOTICES.md`.