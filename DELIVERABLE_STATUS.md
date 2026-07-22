# Project delivery status

## Included in the GitHub repository

- SemiTNet experiment workflow and published target tables reviewed.
- Project-specific configuration and launch scripts included.
- TSI15k dataset and model-checkpoint downloader included.
- Checkpoint SHA256 verification included.
- Dataset inspection, split handling and COCO normalization included.
- Metric calculation for overall, fully dentate and partially edentulous groups included.
- Teacher/student full-training launchers included.
- Persian primary `README.md` and Persian `docs/RUNBOOK.md` included.
- English secondary documentation included.
- CPU smoke run executed and validated.
- Smoke metrics, history and run manifest are committed.
- Published paper target tables are stored locally as CSV/JSON.
- `make smoke` and `make reference` generate the project figures and tables.
- GitHub Actions runs the smoke/reference workflow and archives outputs as `semitnet-generated-outputs`.
- Repository manifest included as `FILE_MANIFEST.json`.

## Smoke validation result

The executed CPU smoke test is a software-validation experiment and is separate from the full TSI15k experiment. It produced approximately:

- Dice: 82.13%
- IoU: 69.71%
- Precision: 70.34%
- Recall: 98.72%
- F1: 82.13%

## Full GPU execution

The full TSI15k inference and teacher/student training require the CUDA Detectron2/Mask2Former environment and an NVIDIA GPU system.

Recommended execution sequence:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

For full training:

```bash
make full
```

The large raw TSI15k dataset and model checkpoint are downloaded by the included scripts and are not committed directly to Git.