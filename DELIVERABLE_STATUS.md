# Project status

The project uses one Python launcher on Windows and Linux:

```bash
python project.py install
python project.py download
python project.py smoke
python project.py full
```

- Python: 3.10
- Environment: project-local `.venv`
- Conda/Anaconda: not used
- Make: optional only
- Smoke test: validated
- Full run: NVIDIA GPU + CUDA Toolkit 11.7 required
- Windows full run: Visual Studio C++ Build Tools also required
