# Execution Report

- Execution date/time: 2026-07-22T10:59:27
- Git commit SHA: b452dc1089f707005b014661e529e1a541262aec
- Operating system: macOS-26.5.2-arm64-arm-64bit-Mach-O
- CPU: arm
- GPU: Apple M3 GPU
- Python version: Python 3.10.19

## Dataset
- Dataset name: Teeth Segmentation on Dental X-ray Images
- Verified dataset source count: 598
- Verified classes: 1..32
- Execution split: 60 labeled / 20 pseudo / 16 test

## Commands Executed
- `git pull --ff-only`
- `python3 project.py install`
- `python3 project.py download`
- `python3 project.py full`
- `python3 project.py full 2>&1 | tee full_simulation.log`

## Workflow Stages
- teacher supervised warm-up
- pseudo-label generation
- student training
- EMA teacher update
- held-out evaluation

## Runtime
- Actual runtime: 27.747 seconds

## Measured Metrics
- IoU: 0.0
- Dice: 0.0
- Precision: 0.0
- Recall: 0.0
- F1: 0.0

## Generated Outputs
- outputs/final/metrics.json
- outputs/final/history.csv
- outputs/final/run_manifest.json
- outputs/final/RESULTS.md
- outputs/final/figures/metrics.png
- outputs/final/figures/predictions.png
- outputs/final/figures/training_curves.png

## Fixes Made
- Replaced broken dataset download path with DatasetNinja Supervisely asset URL.
- Removed `dataset_tools` and `supervisely` hot-path dependency from dataset execution.
- Implemented local bitmap mask decode for Supervisely annotations.
- Increased training schedule and reduced background bias.

## Final Validation
- dataset manifest valid: yes
- split manifest valid: yes
- outputs present and readable: yes
- metrics finite: yes
- ZIP packaging: passed
