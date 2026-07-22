# SemiTNet

> Full end-to-end SemiTNet pipeline simulation on real panoramic dental data. Runs on Windows and Linux with Python 3.10 and a project-local `.venv`.

## 1. Install

```bash
python project.py install
```

## 2. Download dataset

```bash
python project.py download
```

The execution dataset is a public panoramic dental segmentation source aligned with the data sources used to construct TSI15k and contains 32 tooth-position classes with pixel-level annotations.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Full simulation

```bash
python project.py full
```

`full` executes the complete experimental workflow end to end: teacher training, pseudo-label generation, student training, EMA teacher updates, and held-out evaluation. To keep execution time bounded, the simulation runs on a deterministic 96-image subset: 60 labeled images, 20 label-hidden pseudo-label images, and 16 held-out test images. All metrics and figures under `outputs/final/` are measured from this real execution.
