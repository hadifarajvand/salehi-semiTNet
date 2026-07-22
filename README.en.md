# SemiTNet

> Runs on Windows and Linux with Python 3.10 and a project-local `.venv`. Conda/Anaconda and Make are not required.

## 1. Install

```bash
python project.py install
```

## 2. Download dataset

```bash
python project.py download
```

The fast replacement dataset contains 598 panoramic X-rays, 32 tooth classes, and pixel-level annotations. The download is about 464 MB.

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Run the real quick experiment

```bash
python project.py full
```

`full` uses a deterministic 96-image real-data subset: 60 labeled images, 20 label-hidden pseudo-label images, and 16 held-out test images. Measured outputs are written to `outputs/final/`.
