# SemiTNet Runbook

> Uses Python 3.10 with a project-local `.venv` only. Conda/Anaconda is not required.

## 1. Install requirements

```bash
make install
```

## 2. Download dataset and model

```bash
make download
```

## 3. Run smoke test

```bash
make smoke
```

## 4. Run full simulation

```bash
make full
```

> The full run requires Linux and an NVIDIA CUDA GPU.
