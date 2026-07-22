# SemiTNet Runbook

> Windows and Linux — Python 3.10 with a project-local `.venv`

## 1. Install

```bash
python project.py install
```

## 2. Download the real fast dataset

```bash
python project.py download
```

Dataset: 598 panoramic X-rays, 32 tooth classes, pixel-level annotations, about 464 MB.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Run the real quick experiment

```bash
python project.py full
```

`full` uses 96 real images: 60 labeled, 20 label-hidden pseudo-label images, and 16 held-out test images. All metrics and figures under `outputs/final/` are measured from this run.
