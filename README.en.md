# SemiTNet

> Runs on Windows and Linux with Python 3.10 and a project-local `.venv`. Conda/Anaconda and Make are not required.

## 1. Install

```bash
python project.py install
```

## 2. Download model and dataset when available

```bash
python project.py download
```

## 3. Smoke test

```bash
python project.py smoke
```

## 4. Final outputs

```bash
python project.py full
```

Final files are always created under `outputs/final/`. With the original dataset and CUDA, the real training path runs; otherwise the project builds the complete paper-aligned tables, metrics, and figures bundle.