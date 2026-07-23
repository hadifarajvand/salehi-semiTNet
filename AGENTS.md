# Autonomous research agent contract — SemiTNet / TED3

This file is authoritative for any coding/research agent operating in this repository.

## Mission

Once the TED3 archives are present locally, execute an end-to-end, defensible SemiTNet reproduction/reimplementation campaign. Do not stop after setup, a smoke test, a partial experiment, or the first error. Diagnose recoverable failures, patch the repository, rerun the affected stage, and continue until the final evidence package passes the completion checklist below.

## Authoritative dataset inputs

The three required TED3 archives are expected under the repository `data/` directory or one of its subdirectories. The agent must prioritize `data/` and auto-discover these exact files:

- `TED3-train.tar`
- `TED3-test.tar`
- `TED3-unlabeled-data-15k-pseudo-mask.tar`

Preferred discovery order:

1. `data/`
2. `data/TED3/`
3. `data/ted3/`
4. `data/incoming/`
5. recursive search under `data/`
6. whole-repository fallback only if needed

These archives are the authoritative inputs for the TED3 campaign. Do not redownload or silently substitute the historical 598-image `quick_teeth` dataset.

The processed dataset root for the campaign is:

```text
data/processed/ted3/
```

The final supervised and semi-supervised dataloaders must consume processed TED3 data from this root. The historical `python project.py full` command belongs to the reduced `quick_teeth` experiment and must not be used as the final TED3 simulation.

The archives and all extracted datasets are local research inputs. They must never be committed, added to Git LFS, copied into a client ZIP, or pushed to GitHub.

Before extraction:

1. Run `python project.py ted3-preflight`.
2. Verify `.gitignore` protects the archive paths and processed data directories.
3. Confirm all three archives are ignored and not tracked/staged.
4. Compute/record archive SHA256 and byte sizes.
5. Inspect tar members for unsafe paths/links before extraction.
6. Preserve the original archives unchanged.

## Non-negotiable scientific rules

1. **Measured outputs only.** Never copy paper metrics into measured-output files and never fabricate predictions, curves, tables, baselines, confidence intervals, or missing experiment results.
2. **Separate published reference from measured evidence.** Any paper number used for comparison must be labeled `published_reference` and carry provenance.
3. Preserve the historical reduced baseline under `outputs/final/`; never overwrite it.
4. TED3 is a defensible public-author dataset for the new campaign, but it must not be claimed to be byte-identical to unavailable TSI15k unless identity-equivalence is proven.
5. Do not tune on the final test set. Use train-derived validation for model selection and reserve test for final evaluation.
6. Every experiment must have a machine-readable manifest: git SHA, dataset archive hashes, processed-root identities, config, seed, environment, device, checkpoint hash, timestamps and exact command.
7. Every locally recoverable failure must be repaired and rerun. Do not merely record a stack trace and stop.
8. Smoke runs are debugging only. They must not be presented as the final simulation.
9. The final execution manifest must prove that the actual training/evaluation dataloaders used `data/processed/ted3/` and not `data/processed/quick_teeth/`.

## Phase 0 — repository and TED3 input preflight

Start with:

```bash
python project.py ted3-preflight
```

Then:

- snapshot Git HEAD/status;
- locate all three archives under `data/`;
- hash and inspect them;
- safely extract/process them into separate roots under `data/processed/ted3/`;
- build source-member → processed-file manifests;
- record disk usage/environment/preflight state under `outputs/ted3_reproduction/preflight/`.

Do not use `scripts/download_quick_dataset.py`, `scripts/prepare_quick_dataset.py`, or the `quick_teeth` split as TED3 input.

## Phase 1 — dataset forensic audit

Inspect the real extracted archive structure before assuming loader formats. Implement/repair TED3-specific loaders/converters as needed.

Audit:

- exact image/annotation counts;
- category/tooth-ID mapping;
- dimensions and image formats;
- corrupt/missing files;
- exact duplicate leakage across train/test/unlabeled with cryptographic hashes;
- near-duplicate checks where practical;
- identity lists/hashes;
- human-label vs pseudo-label vs unlabeled provenance;
- whether test composition can be reconciled with the paper-reported 191-case protocol;
- fully dentate/partially edentulous grouping only if recoverable without guessing.

Output under `outputs/ted3_reproduction/dataset_audit/`, including `DATASET_AUDIT.md`, JSON/CSV manifests and leakage reports.

Do not start expensive training until real-data loading/inference smoke checks pass and train/test leakage/class mapping issues are resolved or explicitly documented.

## Phase 2 — publication-compatible implementation freeze

Use the authors' released SemiTNet/SemiT-SAM implementation and released checkpoint as primary executable evidence. Resolve paper-vs-current-code discrepancies explicitly using `reproduction/PAPER_CODE_DISCREPANCIES.md`.

Freeze and record:

- architecture/backbone;
- RGB/input-size preprocessing policy;
- class mapping;
- 100-query decoder behavior;
- losses and weights;
- teacher/student initialization;
- burn-in/pseudo-label filtering/EMA/distillation;
- optimizer/LR/iteration budget;
- evaluator/post-processing;
- dependency/runtime versions.

Create a completed publication-compatible method manifest. Do not silently use a newer upstream default that contradicts the paper.

## Phase 3 — official checkpoint evaluation first

Before retraining:

- acquire/verify the pinned author checkpoint;
- evaluate it on processed TED3 test data using the publication-compatible loader/evaluator;
- generate real per-image predictions/metrics and aggregate metrics;
- compare with published reference values while clearly labeling the dataset/protocol relationship.

Required output root:

```text
outputs/ted3_reproduction/checkpoint_eval/
```

Required artifacts include `metrics.json`, `per_image_metrics.csv`, `run_manifest.json`, checkpoint hash, predictions, evaluator logs and `CHECKPOINT_EVALUATION.md`.

If performance is unexpectedly poor, diagnose dataset/class/preprocessing/evaluator/checkpoint mismatch before changing training code.

## Phase 4 — supervised controlled experiment

Train the closest defensible publication-compatible architecture using processed labeled TED3 training data. Use a fixed train-derived validation protocol; evaluate the test set only for final assessment.

Output:

```text
outputs/ted3_reproduction/supervised/
```

Save complete configs, histories, validation-selection evidence, checkpoint hashes, final metrics/per-image metrics, qualitative predictions and runtime/resource statistics.

## Phase 5 — full TED3 semi-supervised experiment

Use the processed large TED3 unlabeled/pseudo-mask archive in its correct non-ground-truth role.

Run the closest defensible full workflow supported by publication/released code:

- teacher labeled-data pretraining;
- unlabeled/pseudo-label processing;
- burn-in;
- student training;
- teacher EMA/distillation updates;
- final evaluation with the same validated evaluator.

Do not silently reduce this to `QuickSemiTransformer`, the 60/20/16 quick split, 128×256 inputs, or a toy smoke run.

Output:

```text
outputs/ted3_reproduction/semi_supervised/
```

## Phase 6 — comparison and statistical analysis

Compare on a clearly documented evaluation protocol:

1. historical reduced run — context only;
2. released author checkpoint on TED3;
3. supervised publication-compatible TED3 run;
4. semi-supervised TED3 run.

Where repeated seeds are feasible, report mean/std/confidence intervals. If only one authoritative full run is feasible, state that limitation and do not invent uncertainty.

Create:

- `outputs/ted3_reproduction/comparison/metrics_comparison.csv`
- `outputs/ted3_reproduction/comparison/per_experiment_summary.json`
- `outputs/ted3_reproduction/comparison/STATISTICAL_ANALYSIS.md`
- any ablation/diagnostic tables supported by actual runs.

## Phase 7 — Evaluation/Experiments artifact coverage

Inspect the research article/reference material already present in the repository and enumerate every experiment-derived figure, table, metric and quantitative claim in the Evaluation/Experiments section.

Build:

```text
outputs/ted3_reproduction/paper_exports/EVALUATION_COVERAGE_MATRIX.csv
```

For each artifact record:

- paper identifier;
- purpose/metric;
- source experiment;
- generated artifact path;
- measured vs published-reference provenance;
- fidelity/status (`generated`, `not applicable`, `blocked by unavailable source output`).

Generate all defensible real-output figures/tables: PNG plus vector/PDF where practical, CSV plus readable rendered forms, training curves, metric comparisons, qualitative prediction panels, supervised vs semi-supervised comparison, checkpoint comparison and paper-reference comparison with dataset differences labeled.

Write manuscript-ready analysis to:

```text
outputs/ted3_reproduction/paper_exports/EVALUATION_ANALYSIS.md
```

## Phase 8 — final validation and delivery

Validate:

- final runs consumed `data/processed/ted3/`;
- no final measured result came from `quick_teeth`;
- no unresolved leakage is hidden;
- metrics are finite and traceable;
- figures/tables agree with machine-readable source data;
- published values are never mislabeled as measured;
- datasets/archives/checkpoints are not tracked/staged;
- rerun instructions are complete.

Final delivery root:

```text
outputs/ted3_reproduction/delivery/
```

Required reports:

- `FINAL_RESULTS.md`
- `REPRODUCIBILITY_REPORT.md`
- `ISSUES_AND_FIXES.md`
- `RUNBOOK_FA.md`
- `artifact_manifest.json`

The final execution manifest must explicitly record:

- discovered source archive paths under `data/`;
- archive SHA256 values;
- processed dataset paths under `data/processed/ted3/`;
- actual dataset roots used by each dataloader/experiment;
- confirmation that final experiments did not use `quick_teeth`.

## Failure-handling policy

Operate as a repair loop:

1. capture failing command/logs;
2. identify root cause;
3. patch the smallest defensible fix;
4. add/update a smoke/regression check where useful;
5. rerun the failed stage;
6. continue downstream after it passes.

Do not repeatedly retry unchanged commands. Do not weaken validation, fabricate output, train on test, substitute legacy data, or silently change the scientific question.

Only a genuinely external/non-recoverable blocker—such as corrupt/missing archives with no valid copy, unavailable required external credentials/assets, or objectively impossible storage/compute—may stop the campaign early. Preserve all completed artifacts and an exact resumable command/state.

## Completion checklist

The agent may stop only when all applicable items are satisfied:

- [ ] All three TED3 archives discovered under `data/`, Git-ignored, untracked, hashed and safely processed.
- [ ] `data/processed/ted3/` created with train/test/unlabeled provenance manifests.
- [ ] Dataset corruption/leakage/class mapping audit completed.
- [ ] Real TED3 loader/inference smoke checks pass.
- [ ] Publication-compatible method manifest frozen.
- [ ] Official checkpoint evaluation completed with measured outputs.
- [ ] Supervised TED3 experiment completed and validated.
- [ ] Semi-supervised TED3 experiment completed and validated.
- [ ] Final manifests prove the simulations consumed TED3, not `quick_teeth`.
- [ ] Per-image and aggregate metrics saved.
- [ ] Required Evaluation/Experiments figures generated from real outputs.
- [ ] Required Evaluation/Experiments tables generated from real outputs.
- [ ] Evaluation coverage matrix complete.
- [ ] Paper-vs-measured comparison and limitations documented.
- [ ] Persian runbook complete.
- [ ] Final delivery package integrity-checked.
- [ ] Git status confirms no raw data/archive/checkpoint is staged or tracked.

Final status must be one of:

- `COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION`
- `BLOCKED — EXTERNAL NON-RECOVERABLE` only with exact evidence and resumable state.
