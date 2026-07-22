# SemiTNet

> Runs on Windows and Linux with Python 3.10 and a project-local `.venv`. Conda/Anaconda and Make are not required.

## 1. Install requirements

```bash
python project.py install
```

## 2. Download dataset and model

```bash
python project.py download
```

## 3. Run smoke test

```bash
python project.py smoke
```

## 4. Run full simulation

```bash
python project.py full
```

> The full run requires an NVIDIA GPU and CUDA Toolkit 11.7. Windows also requires Visual Studio C++ Build Tools.
