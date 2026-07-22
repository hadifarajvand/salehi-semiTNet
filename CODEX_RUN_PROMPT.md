# Codex Execution Prompt — SemiTNet Full Pipeline Simulation

You are the execution engineer for the repository:

`https://github.com/hadifarajvand/salehi-semiTNet`

Your task is to execute the repository end-to-end and leave a complete, validated, deliverable results package. Do not stop for confirmation. Diagnose and fix execution issues autonomously when they are caused by environment, dependency, path, dataset-format, or script problems. Keep all fixes minimal and reproducible.

## Goal

Run the complete implemented SemiTNet teacher/student simulation workflow on the repository's verified deterministic real-data subset and produce measured outputs:

1. teacher supervised warm-up
2. pseudo-label generation
3. student training
4. EMA teacher update
5. held-out evaluation
6. IoU, Dice, Precision, Recall, and F1 calculation
7. prediction visualizations
8. training curves
9. run manifests and reproducibility report

Do not substitute published paper metrics for measured execution results.

## Repository preparation

1. Work on the latest `main` branch.
2. Print:
   - OS
   - architecture
   - Python versions available
   - CPU/RAM
   - GPU availability if any
   - free disk space
   - current Git commit SHA
3. Ensure Python 3.10 is available because the project launcher requires it.
4. Do not use Conda/Anaconda.
5. Use only the project-local `.venv` created by the repository.

Run:

```bash
git status
git pull --ff-only
python project.py install
```

If `python` is not Python 3.10, use the appropriate Python 3.10 launcher for the OS to start `project.py install`; after creation, the project must use `.venv` automatically.

## Dataset download and verification

Run:

```bash
python project.py download
```

The downloader must target exactly:

`Teeth Segmentation on Dental X-ray Images`

Expected verified source properties:

- 598 panoramic dental X-ray images
- tooth classes exactly 1 through 32
- pixel-level tooth annotations
- CC0 1.0 source license

The download must produce:

```text
data/raw/quick_teeth/download_manifest.json
```

Verify that `download_manifest.json` reports:

- `verified_total_images = 598`
- all classes `1..32`

The preparation step must produce:

```text
data/processed/quick_teeth/split_manifest.json
```

Verify deterministic split counts:

- 60 labeled training images
- 20 label-hidden pseudo-label images
- 16 held-out test images

Verify that pseudo-label input images do NOT have annotations in the prepared `pseudo_unlabeled` input directory.

If download or preparation fails:

- inspect the actual downloaded directory structure;
- fix only the downloader/preparation compatibility issue;
- preserve exact dataset identity checks;
- rerun `python project.py download`;
- do not bypass the 598-image and 32-class verification checks.

## Execute the simulation

Run:

```bash
python project.py full
```

Capture stdout/stderr to a persistent log as well as the console, for example:

Linux/macOS:

```bash
python project.py full 2>&1 | tee full_simulation.log
```

Windows PowerShell equivalent is acceptable.

Do not stop after the smoke test. The required target is the real-data `full` run.

## Required measured outputs

The run is successful only if all of these exist and are non-empty:

```text
outputs/final/metrics.json
outputs/final/history.csv
outputs/final/run_manifest.json
outputs/final/RESULTS.md
outputs/final/figures/metrics.png
outputs/final/figures/predictions.png
outputs/final/figures/training_curves.png
```

### `metrics.json` must contain measured numeric values for

- `iou`
- `dice`
- `precision`
- `recall`
- `f1`
- `runtime_seconds`
- `used_labeled_train = 60`
- `used_label_hidden_pseudo = 20`
- `used_test = 16`
- dataset name
- verified source image count

Values must come from the executed run. Do not overwrite them with paper reference values.

### `run_manifest.json` must record

- dataset manifest path
- dataset name
- verified source image count
- deterministic seed
- model name
- workflow stages
- 60/20/16 split

### Figures

Open/inspect each generated PNG and verify it is readable and not blank/corrupted:

- `metrics.png`: measured IoU/Dice/Precision/Recall/F1
- `predictions.png`: panoramic X-ray, 32-class ground truth, prediction panels
- `training_curves.png`: teacher/student training-loss history

## Validation

After execution, run a validation script or equivalent checks that:

1. parses both manifests and `metrics.json`;
2. verifies all required files exist and have non-zero size;
3. verifies the dataset count is 598;
4. verifies classes are exactly 1..32;
5. verifies split is exactly 60/20/16;
6. verifies all five metrics are finite numbers between 0 and 100;
7. verifies runtime is positive;
8. verifies no output file used `paper/reference/paper_targets.json` as its measured metric source;
9. reports Git commit SHA and environment details.

If any validation fails, fix the cause and rerun the affected stage until validation passes.

## Deliverable package

Create a final folder:

```text
deliverables/SemiTNet-full-simulation/
```

Copy into it:

```text
outputs/final/
data/raw/quick_teeth/download_manifest.json
data/processed/quick_teeth/split_manifest.json
full_simulation.log
README.md
README.en.md
docs/RUNBOOK.md
docs/RUNBOOK.en.md
```

Also create:

```text
deliverables/SemiTNet-full-simulation/EXECUTION_REPORT.md
```

The report must include:

- execution date/time
- Git commit SHA
- machine/OS/CPU/GPU
- verified dataset identity and 598-image source count
- deterministic 60/20/16 execution split
- exact commands executed
- actual runtime
- actual measured IoU, Dice, Precision, Recall, F1
- list of generated figures/files
- any fixes applied during execution
- final PASS/FAIL status

Then create:

```text
SemiTNet-full-simulation-results.zip
```

containing the entire deliverable folder.

## Final response format

When finished, respond with exactly these sections:

### STATUS
`PASS` or `FAIL`

### DATASET
- exact dataset name
- verified source count
- classes
- prepared split

### EXECUTION
- commit SHA
- machine/device
- runtime
- commands run

### MEASURED RESULTS
- IoU
- Dice
- Precision
- Recall
- F1

### GENERATED OUTPUTS
List every required output with file size and validation status.

### DELIVERABLE
Give the absolute/local path to `SemiTNet-full-simulation-results.zip`.

### FIXES
List any code/environment fixes you made. If none, say `None`.

Do not claim PASS unless the dataset verification, full execution, measured metrics, all required figures/files, and final ZIP have all been successfully validated.
