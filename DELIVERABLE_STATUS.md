# Project status

The project uses one Python launcher on Windows, Linux, and macOS:

```bash
python project.py install
python project.py download
python project.py smoke
python project.py full
```

- Python: 3.10
- Environment: project-local `.venv`
- Conda/Anaconda: not used
- Make: not required
- Smoke test: validated
- Dataset command: downloads the exact Dataset Ninja dataset `Teeth Segmentation on Dental X-ray Images`
- Dataset identity guard: requires exactly 598 images and tooth classes 1..32 before preparation succeeds
- Raw dataset: `data/raw/quick_teeth/`
- Download verification: `data/raw/quick_teeth/download_manifest.json`
- Prepared dataset: `data/processed/quick_teeth/`
- Prepared split manifest: `data/processed/quick_teeth/split_manifest.json`
- Deterministic execution split: 60 labeled training + 20 label-hidden pseudo-label + 16 held-out test images
- Pseudo-label leakage guard: pseudo-label annotations are not copied into the prepared pseudo input
- Full simulation workflow: teacher training → pseudo-label generation → student training → EMA teacher update → held-out evaluation
- Final outputs: measured IoU, Dice, Precision, Recall, F1, predictions, training curves, metrics chart, history, and run manifest under `outputs/final/`

`python project.py full` consumes only the verified prepared split manifest. If the prepared dataset is missing, it automatically executes the download and preparation stage first.
