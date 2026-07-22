# SemiTNet

> Full end-to-end SemiTNet pipeline simulation on real panoramic dental data. Runs on Windows and Linux with Python 3.10 and a project-local `.venv`.

## 1. Install

```bash
python project.py install
```

## 2. Download and prepare dataset

```bash
python project.py download
```

This command downloads the exact Dataset Ninja dataset named `Teeth Segmentation on Dental X-ray Images` from the Humans in the Loop source and verifies that it contains 598 images and tooth classes 1 through 32 before it is accepted.

After download:

- Raw data is kept under `data/raw/quick_teeth/`.
- Download verification is stored in `data/raw/quick_teeth/download_manifest.json`.
- The deterministic simulation split is created under `data/processed/quick_teeth/`.
- Exact split provenance is stored in `data/processed/quick_teeth/split_manifest.json`.
- Execution split: 60 labeled images, 20 label-hidden pseudo-label images, and 16 held-out test images.
- Pseudo-label ground-truth annotations are not copied into the simulation input, preventing label leakage.

If the downloaded content is incomplete or is not this exact dataset, `download` fails instead of allowing the simulation to continue.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Full simulation

```bash
python project.py full
```

`full` consumes only the verified prepared manifest and executes teacher training, pseudo-label generation, student training, EMA teacher updates, and held-out evaluation. If the prepared dataset is missing, `full` automatically runs the download/setup stage first. All measured metrics and figures are written under `outputs/final/`.
