# SemiTNet

> Full end-to-end SemiTNet pipeline simulation on real panoramic dental data. Runs on Windows and Linux with Python 3.10 and a project-local `.venv`.

> Clone-safe default notebook path ships with bounded dataset already committed at `data/processed/quick_teeth/`.
> Default notebook run does **not** need the 598-image source dataset.

## 1. Install

```bash
python project.py install
```

## 1.1 Open notebook

```bash
jupyter notebook notebooks/SemiTNet_Reproduction_and_Evaluation.ipynb
```

Then run `Run All`.

The default notebook execution uses the committed bounded subset:

- 60 labeled training images
- 20 pseudo/unlabeled images
- 16 held-out test images

Measured outputs go to `outputs/final/` and paper-style figures/tables go to `outputs/paper_style/`.

## 2. Download and prepare dataset

```bash
python project.py download
```

This command downloads the Dataset Ninja dataset named `Teeth Segmentation on Dental X-ray Images` from the Humans in the Loop source and refuses to continue unless it verifies exactly 598 images and tooth classes 1 through 32.

For the default notebook demonstration, you do **not** need this download step because the bounded subset is already included in `data/processed/quick_teeth/`.

After download:
- Raw data: `data/raw/quick_teeth/`
- Download manifest: `data/raw/quick_teeth/download_manifest.json`
- Prepared data: `data/processed/quick_teeth/`
- Split manifest: `data/processed/quick_teeth/split_manifest.json`
- Deterministic split: 60 labeled + 20 label-hidden pseudo-label + 16 held-out test images
- Pseudo-label ground-truth annotations are not copied into the training input.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Full simulation

```bash
python project.py full
```

`full` executes teacher training, pseudo-label generation, student training, EMA teacher updates, collapse protection, held-out evaluation, strict output validation, and client packaging.

Low-confidence pseudo-label pixels are ignored rather than converted to background. A health check prevents a collapsed student from replacing a healthier teacher checkpoint. An all-zero final metric set is rejected and cannot be packaged as PASS.

Validated measured outputs are written under `outputs/final/`. After validation, the client package is created as:

```text
SemiTNet-client-deliverable.zip
```

## Paper-style figures

```bash
python scripts/export_paper_style_figures.py
```

This exports publication-style figures to `outputs/paper_style/figures/` using only the measured simulation outputs and paper reference tables. It does not retrain or rerun inference.
