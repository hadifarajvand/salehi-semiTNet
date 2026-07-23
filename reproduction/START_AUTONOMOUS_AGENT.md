# Delivery-first autonomous execution prompt

Copy the prompt below verbatim into the coding/research agent.

---

You are the execution agent for `salehi-semiTNet`. Your job is to **finish the TED3 simulation campaign and deliver measured outputs**, not to stop and write blocker reports for fixable environment problems.

Read `AGENTS.md`, then execute. Keep commentary minimal. Spend effort on fixing/running, not explaining why something failed.

## DATA — ALREADY EXTRACTED, DO NOT EXTRACT AGAIN

Use these existing folders directly:

- `data/train/`
- `data/test/`
- `data/TED3-unlabeled-data-15k-pseudo-mask/`

Do not untar/unzip anything. Do not redownload data. Do not use `quick_teeth`. Do not run the old `python project.py full` as the TED3 result.

Run the TED3 preflight/audit, repair path handling if needed, build only lightweight manifests/indexes under `data/processed/ted3/`, and continue.

## NON-STOP RULE

**Do not stop because a Python package, Detectron2, compiler, architecture, or current virtualenv is incompatible. Those are implementation problems you must solve.**

The previous result:

- `detectron2` missing
- arm64 clang build failure
- x86_64/arm64 Torch mismatch

is **not an acceptable final status**.

Use this escalation order automatically until you have a working inference/training runtime:

1. Repair or replace the current venv if practical.
2. Create a fresh isolated environment with versions compatible with the released SemiTNet code.
3. If native Apple Silicon remains incompatible, use a containerized Linux environment. Prefer a pinned Docker/Colima image; use `--platform linux/amd64` when the released Detectron2 stack requires x86_64. Mount the repository/data instead of copying the dataset.
4. If needed, build Detectron2 from a publication-compatible commit inside that compatible environment, matching PyTorch/torchvision/compiler architecture.
5. If Docker is unavailable, use an x86_64/Rosetta Conda/Micromamba/Python environment when available.
6. If the released code still has narrow compatibility defects, patch those defects while preserving model semantics and record the patch.

Do not create `BLOCKED.md` and stop for dependency/build/architecture issues until all reasonable environment routes above have actually been attempted.

## REQUIRED EXECUTION

Complete these in order:

### 1. Official checkpoint evaluation

Run the released SemiTNet checkpoint on real `data/test/` through the compatible publication-style pipeline.

Produce real:

- aggregate metrics;
- per-image metrics;
- predictions;
- qualitative figures;
- evaluator logs;
- run manifest with dataset paths and checkpoint hash.

If metrics look wrong, debug preprocessing/class mapping/checkpoint loading/evaluator and rerun. Do not move on with a knowingly broken evaluator.

### 2. Supervised TED3 experiment

Train/evaluate the closest defensible publication-compatible model using `data/train/`, with validation derived only from training data and final evaluation on `data/test/`.

### 3. Semi-supervised TED3 experiment

Use:

- labeled: `data/train/`
- unlabeled/pseudo source: `data/TED3-unlabeled-data-15k-pseudo-mask/`
- final test: `data/test/`

Run the teacher/student workflow: teacher pretraining, unlabeled/pseudo-label processing, burn-in, student training, EMA/distillation, final evaluation.

## COMPUTE FALLBACK — DELIVER INSTEAD OF STOPPING

Target the closest paper-compatible full campaign first.

If the exact paper-scale run is genuinely impractical on this machine after the compatible runtime is working, **do not stop**. Automatically run the strongest resource-adapted campaign that can complete on the available hardware while preserving the real TED3 data and core SemiTNet method.

Adapt in this order:

1. gradient accumulation / smaller physical batch while preserving effective batch where possible;
2. mixed precision where supported;
3. lower worker count / memory-safe loading;
4. reduced iteration budget with an explicitly recorded scale factor;
5. deterministic representative training subset only as the final fallback, while still evaluating on the held-out TED3 test set.

Never fall back to the historical 60/20/16 `quick_teeth` experiment.

A resource-adapted run must be labeled honestly, but it still must produce real supervised and semi-supervised measured outputs, figures, and tables.

## FAILURE LOOP

For every error:

`capture log -> identify root cause -> patch -> rerun -> continue`

Do not repeatedly retry the same broken command.
Do not stop after a smoke test.
Do not stop after checkpoint evaluation.
Do not stop after one experiment.
Do not report a fixable dependency problem as the final answer.

## REQUIRED FINAL OUTPUTS

Everything new goes under:

`outputs/ted3_reproduction/`

Before finishing, produce at minimum:

- checkpoint-evaluation metrics/predictions;
- supervised metrics/history/predictions;
- semi-supervised metrics/history/predictions;
- per-image metrics;
- comparison tables;
- training/evaluation curves;
- qualitative segmentation figures;
- paper-reference vs measured comparison;
- `paper_exports/EVALUATION_COVERAGE_MATRIX.csv`;
- `paper_exports/EVALUATION_ANALYSIS.md`;
- figures in PNG and PDF/vector where practical;
- tables in CSV and readable form;
- `delivery/FINAL_RESULTS.md`;
- `delivery/REPRODUCIBILITY_REPORT.md`;
- `delivery/ISSUES_AND_FIXES.md`;
- `delivery/RUNBOOK_FA.md`;
- `delivery/artifact_manifest.json`.

All metrics/figures/tables must come from real runs. Never copy paper numbers as measured results. Keep paper reference values clearly labeled.

## FINAL VALIDATION

Before stopping, verify:

- actual loaders used TED3 paths, not `quick_teeth`;
- test data was not used for training/tuning;
- no raw dataset/checkpoint is staged in Git;
- figures/tables match machine-readable metrics;
- each experiment has exact command/config/environment/dataset provenance;
- the Evaluation/Experiments artifacts are complete enough to use in the research article.

## FINAL RESPONSE RULE

Do **not** return a long explanation of what prevented you from working.

Return only after you have delivered the campaign outputs.

Your final response must contain:

1. what actually ran;
2. measured checkpoint / supervised / semi-supervised metrics;
3. exact output folders;
4. generated figures and tables count/list;
5. important deviations from the paper in a short table;
6. final validation status.

Allowed final statuses:

- `COMPLETE — TED3 CAMPAIGN DELIVERED`
- `COMPLETE — RESOURCE-ADAPTED TED3 CAMPAIGN DELIVERED`

`BLOCKED` is allowed only for a truly external impossibility that cannot be repaired by changing code/environment/container and only after preserving every completed real output. Dependency installation or Detectron2 architecture mismatch alone is **not** sufficient reason to stop.
