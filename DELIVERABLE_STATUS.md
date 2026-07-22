# Deliverable status

## Completed in this package

- Paper method and target tables audited.
- Current public upstream repository audited and pinned.
- Deterministic patcher created for private paths/debug overrides.
- TSI15k and released-checkpoint downloader created.
- Official checkpoint SHA256 verification included.
- Dataset inspection and COCO normalization included.
- Author-style metric computation included for overall, fully dentate, and partially edentulous groups.
- Teacher/student full-training launchers included.
- Persian primary README and runbook included.
- English secondary documentation included.
- CPU smoke run executed and validated.
- Published paper tables converted to local CSV/JSON and generated comparison charts.

## Not executed in this environment

The real TSI15k checkpoint inference and full training were not executed here because the active environment is CPU-only and cannot download/build the CUDA Detectron2/Mask2Former stack. The released checkpoint inference is the recommended next run on a Linux NVIDIA system. Full retraining is substantially heavier.
