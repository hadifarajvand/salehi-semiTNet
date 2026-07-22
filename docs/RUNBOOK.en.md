# SemiTNet Runbook

> Full end-to-end SemiTNet pipeline simulation on real data — Windows and Linux — Python 3.10 with a project-local `.venv`

## 1. Install

```bash
python project.py install
```

## 2. Download dataset

```bash
python project.py download
```

The execution data comes from a public panoramic dental source aligned with the source datasets used to construct TSI15k and includes 32 tooth-position classes with pixel-level annotations.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Full simulation

```bash
python project.py full
```

This command executes the complete experiment pipeline: teacher training, pseudo-label generation, student training, EMA teacher updates, and final evaluation. For fast execution, a deterministic 96-image subset is used: 60 labeled, 20 pseudo-label, and 16 test images. All metrics, predictions, and figures under `outputs/final/` are generated from this real run.
