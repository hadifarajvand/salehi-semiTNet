# SemiTNet Runbook

> Windows and Linux — Python 3.10 with a project-local `.venv`

## 1. Install

```bash
python project.py install
```

## 2. Download and prepare the dataset

```bash
python project.py download
```

This command downloads the exact Dataset Ninja dataset named `Teeth Segmentation on Dental X-ray Images` and verifies:

- exactly 598 images
- tooth classes 1 through 32
- Supervisely pixel-level annotation structure

Prepared layout:

```text
data/raw/quick_teeth/
    download_manifest.json

data/processed/quick_teeth/
    split_manifest.json
    train_labeled/
    pseudo_unlabeled/
    test/
```

Deterministic simulation split:

- 60 labeled images for supervised teacher/student training
- 20 images copied without annotations into the pseudo-label input
- 16 held-out test images with annotations

The command fails if the downloaded content is incomplete or is not the expected dataset.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Run simulation

```bash
python project.py full
```

`full` consumes only `data/processed/quick_teeth/split_manifest.json`. If the verified prepared dataset is missing, it automatically runs the download and preparation stage first.

Outputs:

```text
outputs/final/
    metrics.json
    history.csv
    run_manifest.json
    RESULTS.md
    figures/
        metrics.png
        predictions.png
        training_curves.png
```

## Paper-style figures

```bash
python scripts/export_paper_style_figures.py
```

This exports paper-style figures to `outputs/paper_style/figures/` using only `outputs/final/` and the `paper/reference/` tables. No retraining or inference rerun occurs.
