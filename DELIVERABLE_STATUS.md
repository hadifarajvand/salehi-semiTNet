# Deliverable status

## Completed in the GitHub repository

- Paper method and published target tables audited.
- Current public upstream implementation audited and pinned to commit `bf66074bee9da9b37fd68454bcbac9140c4f59e2`.
- Deterministic patcher included for private paths, debug overrides, dataset registration and 32-class configuration.
- TSI15k dataset and released-checkpoint downloader included.
- Official checkpoint SHA256 verification included.
- Dataset inspection, split preservation/fallback handling and COCO normalization included.
- Author-style metric computation included for overall, fully dentate and partially edentulous groups.
- Teacher/student full-training launchers included.
- Persian primary `README.md` and Persian `docs/RUNBOOK.md` included.
- English secondary documentation included.
- CPU smoke run executed and validated.
- Smoke metrics/history/manifests are committed.
- Published paper tables are stored locally as CSV/JSON.
- `make smoke` and `make reference` regenerate all derived figures.
- GitHub Actions workflow `smoke-and-reference` regenerates and archives the figures as artifact `semitnet-generated-outputs`.
- Repository manifest included as `FILE_MANIFEST.json`.

## Smoke validation result

The executed CPU smoke test is a software-validation experiment only, not a paper-result claim. It produced approximately:

- Dice: 82.13%
- IoU: 69.71%
- Precision: 70.34%
- Recall: 98.72%
- F1: 82.13%

## Not executed in this environment

The real TSI15k released-checkpoint inference and full teacher/student retraining were not executed here because the active execution environment is CPU-only and cannot run the required CUDA Detectron2/Mask2Former stack at practical speed.

The recommended next step on a Linux NVIDIA GPU system is:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

For full retraining:

```bash
make full
```

The repository deliberately does not commit the large raw TSI15k dataset or large released model checkpoint; both are downloaded by the provided scripts.
