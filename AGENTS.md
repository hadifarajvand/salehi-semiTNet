# Autonomous research agent contract — SemiTNet / TED3

This file is authoritative for any coding/research agent operating in this repository.

## Mission

Execute an end-to-end, defensible SemiTNet reproduction/reimplementation campaign using the TED3 datasets that are **already extracted under `data/`**. Do not stop after setup, a smoke test, a partial experiment, or the first error. Diagnose recoverable failures, patch the repository, rerun the affected stage, and continue until the final evidence package passes the completion checklist.

## Authoritative dataset inputs — already extracted

The authoritative local inputs are these directories:

- `data/train/`
- `data/test/`
- `data/TED3-unlabeled-data-15k-pseudo-mask/`

These folders have already been extracted by the user.

### Absolute rule: do not extract the archives again

Do **not** unzip, untar, re-extract, duplicate, or overwrite these datasets from any `.tar`, `.tar.gz`, or `.zip` file.

If the original archives are still present under `data/`, retain them only for provenance/hash verification. They are not the runtime inputs and must not be extracted again.

The final TED3 loaders must consume either the already-extracted directories directly or deterministic indexes/manifests created from them under:

```text
data/processed/ted3/
```

Do not copy the full image corpus into `data/processed/ted3/` unless conversion is technically required. Prefer manifests, indexes, metadata, converted annotations, symlinks/hardlinks where safe, or derived files so unnecessary multi-GB duplication is avoided.

The historical 598-image `quick_teeth` dataset is not an input to this campaign.

## Git/data safety

Before processing:

1. Run `python project.py ted3-preflight`.
2. Verify `.gitignore` protects:
   - `data/train/`
   - `data/test/`
   - `data/TED3-unlabeled-data-15k-pseudo-mask/`
   - `data/processed/`
   - any local archives/checkpoints.
3. Confirm no raw dataset file is tracked or staged.
4. Preserve source directories unchanged as much as possible.
5. Record source-directory inventories and, where practical, cryptographic identities/manifests.

## Non-negotiable scientific rules

1. **Measured outputs only.** Never copy paper metrics into measured-output files and never fabricate predictions, curves, tables, baselines, confidence intervals, or missing experiment results.
2. **Separate published reference from measured evidence.** Any paper number used for comparison must be labeled `published_reference` and carry provenance.
3. Preserve the historical reduced baseline under `outputs/final/`; never overwrite it.
4. TED3 is a defensible public-author dataset for this campaign, but it must not be claimed to be byte-identical to unavailable TSI15k unless identity-equivalence is actually proven.
5. Do not tune on the final `data/test/` set. Derive validation only from training data.
6. Every experiment must have a machine-readable manifest: git SHA, source-directory identities/manifests, processed-root identities, config, seed, environment, device, checkpoint hash, timestamps, and exact command.
7. Every locally recoverable failure must be repaired and rerun. Do not merely record a stack trace and stop.
8. Smoke runs are debugging only. They must not be presented as the final simulation.
9. The final execution manifest must prove that actual TED3 training/evaluation dataloaders resolved to the already-extracted TED3 sources / `data/processed/ted3/`, never `data/processed/quick_teeth/`.
10. Never re-extract the TED3 archives as part of this campaign.

## Phase 0 — extracted-dataset preflight

Start with:

```bash
python project.py ted3-preflight
```

The preflight must validate the three already-extracted directories, inventory their files/extensions/sizes, verify Git safety, and record optional archive hashes only if archives happen to still exist.

Then:

- snapshot Git HEAD/status;
- inspect the real directory structures under `data/train/`, `data/test/`, and `data/TED3-unlabeled-data-15k-pseudo-mask/`;
- do not move/delete/overwrite source datasets;
- build deterministic indexes/manifests/conversions under `data/processed/ted3/` only as needed;
- record environment/disk usage/preflight state under `outputs/ted3_reproduction/preflight/`.

Do not invoke archive extraction commands (`tar -x`, `unzip`, Python tar extraction, etc.) for TED3.
Do not use `scripts/download_quick_dataset.py`, `scripts/prepare_quick_dataset.py`, or the `quick_teeth` split.

## Phase 1 — dataset forensic audit and preparation

Inspect the real extracted structures before assuming formats. Implement/repair TED3-specific parsers/loaders/converters based on what actually exists.

Audit:

- exact image counts and annotation counts;
- class/category mapping and tooth-ID semantics;
- image dimensions and formats;
- missing/corrupt files;
- exact duplicate leakage across train/test/unlabeled using cryptographic hashes;
- near-duplicate checks where practical;
- identity lists/hashes;
- provenance of human labels, pseudo labels, and unlabeled images;
- test-set composition versus the paper-reported protocol without claiming exact TSI15k equivalence unless proven.

Write:

```text
outputs/ted3_reproduction/dataset_audit/
  DATASET_AUDIT.md
  dataset_manifest.json
  train_manifest.*
  test_manifest.*
  unlabeled_manifest.*
  leakage_report.*
```

Build the runtime dataset representation under `data/processed/ted3/`. This may contain indexes, converted annotations, split manifests, metadata, and derived files. The final loaders must reference the actual already-extracted TED3 data or deterministic derived representations of it.

Do not continue to expensive training until:

- class mappings are coherent;
- data are loadable;
- leakage/corruption is resolved or explicitly justified;
- a real-data loader/inference smoke test passes.

## Phase 2 — publication-compatible implementation freeze

Use the authors' released SemiTNet/SemiT-SAM implementation and checkpoint as primary executable evidence while resolving paper-vs-code discrepancies explicitly.

Freeze and record:

- backbone/model architecture;
- RGB/1024 preprocessing policy;
- class mapping;
- 100-query decoder behavior;
- losses and weights;
- teacher/student initialization;
- burn-in and pseudo-label filtering;
- EMA/distillation behavior;
- optimizer/LR schedule/iteration budget;
- evaluator/post-processing thresholds;
- dependency versions.

Complete a method manifest derived from `reproduction/paper_method_manifest.template.json`.

## Phase 3 — official checkpoint evaluation first

Before expensive retraining:

- acquire/verify the pinned official checkpoint if needed;
- run it against the processed/indexed TED3 **test source `data/test/`** using the publication-compatible pipeline;
- generate aggregate/per-image metrics, predictions, logs, and provenance;
- compare against paper reference values while clearly labeling TED3 as a distinct dataset unless equivalence is proven.

Required output:

```text
outputs/ted3_reproduction/checkpoint_eval/
```

If performance is poor, diagnose loader/class mapping/preprocessing/evaluator/checkpoint compatibility before changing training logic.

## Phase 4 — supervised controlled experiment

Train the closest defensible publication-compatible architecture using TED3 labeled data originating from `data/train/`.

- derive validation only from training data;
- never tune on `data/test/`;
- evaluate final selected checkpoint on `data/test/`;
- save configs, histories, checkpoints/hashes, validation evidence, metrics, per-image metrics, predictions, runtime/resource statistics, and exact commands.

Output:

```text
outputs/ted3_reproduction/supervised/
```

## Phase 5 — semi-supervised TED3 experiment

Use the already-extracted:

```text
data/TED3-unlabeled-data-15k-pseudo-mask/
```

as the unlabeled/pseudo-label source.

Identify its actual internal structure and distinguish raw unlabeled images from pseudo masks. Pseudo masks must never be mislabeled as human ground truth.

Run the closest defensible SemiTNet workflow supported by publication + released code:

- teacher/labeled pretraining;
- unlabeled/pseudo-label processing;
- burn-in;
- student training;
- teacher EMA/distillation;
- final evaluation with the same test/evaluator protocol as the supervised run.

Output:

```text
outputs/ted3_reproduction/semi_supervised/
```

## Phase 6 — experiment comparison and statistical analysis

Compare consistently:

1. historical reduced run (context only);
2. released author checkpoint on TED3;
3. supervised publication-compatible run;
4. semi-supervised TED3 run.

Where repeated seeds are computationally feasible, report mean/std/confidence intervals. Otherwise state single-run limitations explicitly.

Create:

```text
outputs/ted3_reproduction/comparison/
  metrics_comparison.csv
  per_experiment_summary.json
  STATISTICAL_ANALYSIS.md
```

## Phase 7 — research article Evaluation/Experiments coverage

Inspect the research article/reference material in the repository and build an artifact coverage matrix for every experiment-derived figure, table, metric, and quantitative statement relevant to the Evaluation/Experiments section.

Required:

```text
outputs/ted3_reproduction/paper_exports/
  EVALUATION_COVERAGE_MATRIX.csv
  EVALUATION_ANALYSIS.md
  figures/
  tables/
```

For each artifact record:

- paper figure/table identifier;
- purpose/metric;
- source experiment;
- generated path;
- measured vs published-reference provenance;
- status (`generated`, `not applicable`, `blocked by unavailable source output`).

Generate all defensible figures/tables from real measured outputs only. Prefer PNG + vector/PDF figures where practical and CSV + readable table forms.

## Phase 8 — final validation and delivery

Validate:

- all metrics finite and sourced from real outputs;
- figures traceable to source data;
- table values match machine-readable metrics;
- paper values are never mislabeled as measured;
- train/test leakage is absent or explicitly explained;
- final manifests prove TED3 paths were used;
- no `quick_teeth` dataset was used for TED3 final experiments;
- no dataset/archive/oversized checkpoint is tracked or staged;
- runbook is sufficient for another researcher to rerun.

Create:

```text
outputs/ted3_reproduction/delivery/
  FINAL_RESULTS.md
  REPRODUCIBILITY_REPORT.md
  ISSUES_AND_FIXES.md
  RUNBOOK_FA.md
  artifact_manifest.json
```

## Failure-handling policy

For every recoverable failure:

1. capture failing command/logs;
2. identify root cause;
3. patch the smallest defensible fix;
4. add/adjust regression/smoke checks where appropriate;
5. rerun the failed stage;
6. continue downstream only when it passes.

Do not repeatedly retry unchanged commands. Do not weaken scientific checks, fabricate outputs, substitute datasets, or silently reduce the scientific question to force a PASS.

A genuinely external/non-recoverable blocker must be documented precisely with completed partial artifacts and an exact resumable state. Otherwise continue.

## Completion checklist

The agent may stop only when all applicable items are satisfied:

- [ ] `data/train/`, `data/test/`, and `data/TED3-unlabeled-data-15k-pseudo-mask/` were validated as authoritative inputs.
- [ ] No TED3 archive was re-extracted.
- [ ] Dataset audit/leakage/corruption/class mapping completed.
- [ ] Runtime manifests/loaders under `data/processed/ted3/` created and verified.
- [ ] Official checkpoint evaluation completed with measured outputs.
- [ ] Supervised experiment completed and validated.
- [ ] Semi-supervised experiment completed and validated.
- [ ] Per-image and aggregate metrics saved.
- [ ] Required figures and tables generated from real outputs.
- [ ] Evaluation coverage matrix complete.
- [ ] Paper-vs-measured comparison/limitations documented.
- [ ] Persian runbook complete.
- [ ] Final deliverable package built and integrity-checked.
- [ ] Final manifests prove TED3 data paths were used, not `quick_teeth`.
- [ ] Git status confirms datasets/archives/checkpoints are not accidentally tracked/staged.

Final status must be one of:

- `COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION`
- `BLOCKED — EXTERNAL NON-RECOVERABLE` only with exact evidence and resumable state.
