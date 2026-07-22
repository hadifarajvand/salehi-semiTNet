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
- Full simulation: end-to-end teacher/student pipeline executed on a deterministic reduced real-data subset
- Workflow: teacher training → pseudo-label generation → student training → EMA teacher update → held-out evaluation
- Data setup: 32 tooth-position classes with pixel-level panoramic dental annotations, aligned with the source domain used to construct TSI15k
- Reduced execution split: 60 labeled training + 20 label-hidden pseudo-label + 16 held-out test images
- Final outputs: measured IoU, Dice, Precision, Recall, F1, predictions, training curves, metrics chart, history, and run manifest under `outputs/final/`

The reduced subset is used to keep execution time practical while preserving the complete experimental workflow and output-generation process.
