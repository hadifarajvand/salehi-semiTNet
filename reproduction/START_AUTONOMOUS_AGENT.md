# Exact autonomous execution prompt — already-extracted TED3 folders

Copy the prompt below verbatim into the coding/research agent.

---

You are the autonomous research/reproduction agent for this repository. Work directly inside the current `salehi-semiTNet` repository and execute the entire task end-to-end without asking me to supervise intermediate failures.

First, read `AGENTS.md` completely and treat it as the authoritative scientific and execution contract.

## CRITICAL INPUT RULE — DATA ARE ALREADY EXTRACTED

The TED3 datasets have **already been extracted** by the user. Do not unzip, untar, re-extract, duplicate, or overwrite them from any archive.

These three directories are the authoritative dataset inputs:

```text
data/train/
data/test/
data/TED3-unlabeled-data-15k-pseudo-mask/
```

Use these directories directly as the source data for this campaign.

If `.tar`, `.tar.gz`, or `.zip` archives are still present in `data/`, retain them only for provenance/hash verification. **Do not extract them again under any circumstance.**

Do not redownload another substitute dataset.
Do not use the historical 598-image `quick_teeth` dataset.
Do not run `python project.py full` as the TED3 reproduction; that command belongs to the old reduced simulation.

Your first command must be:

```bash
python project.py ted3-preflight
```

The preflight must validate the three already-extracted directories, verify Git safety, inventory their files/extensions/sizes, and optionally record archive hashes if archives still exist. If preflight fails for a locally repairable reason, fix the repository or path handling safely and rerun until it passes.

## DO NOT RE-EXTRACT OR DUPLICATE THE DATASET

After preflight:

- inspect the actual existing folder structures in place;
- do not run `tar -x`, `tarfile.extractall`, `unzip`, or equivalent archive extraction for TED3;
- do not delete or overwrite the source directories;
- do not create another full copy of the multi-GB datasets unless a technically necessary format conversion requires specific derived files.

Build deterministic manifests, indexes, converted annotations, metadata, split definitions, and other derived artifacts under:

```text
data/processed/ted3/
```

The final loaders may read the original extracted files directly through these manifests/indexes. Prefer this over copying all images again.

## REQUIRED DATA PROCESSING AND AUDIT

Inspect the real contents before assuming any annotation or loader format.

Process these roles correctly:

1. `data/train/`
   - labeled training source;
   - identify images, annotations, classes, and tooth-ID mapping;
   - derive validation only from this training source.

2. `data/test/`
   - held-out final evaluation source;
   - never use for training, pseudo-label generation, hyperparameter tuning, or checkpoint selection.

3. `data/TED3-unlabeled-data-15k-pseudo-mask/`
   - semi-supervised unlabeled/pseudo-mask source;
   - identify raw unlabeled images and pseudo masks from the actual folder structure;
   - never treat pseudo masks as human ground-truth test labels.

Run a forensic audit before expensive simulation:

- exact image and annotation counts;
- corrupt/missing files;
- class/category and tooth-ID mapping;
- dimensions and image formats;
- exact duplicate leakage across train/test/unlabeled;
- near-duplicate checks where practical;
- identity manifests and hashes;
- train/test contamination checks;
- provenance of human labels vs pseudo labels vs unlabeled images;
- comparison of TED3 test composition with the paper-reported protocol, without claiming exact TSI15k equivalence unless actually proven.

Write the audit under:

```text
outputs/ted3_reproduction/dataset_audit/
```

Do not continue to expensive training until the dataset is demonstrably loadable, mappings are coherent, leakage/corruption issues are resolved or explicitly documented, and a real-data loader/inference smoke test passes.

## IMPLEMENTATION AND SIMULATION

Use the authors' released SemiTNet/SemiT-SAM implementation and released checkpoint as the primary executable evidence. Resolve paper-vs-code discrepancies explicitly rather than blindly using newer defaults.

The campaign must execute in this order:

### A. Official checkpoint evaluation

- Acquire/verify the pinned official checkpoint if needed.
- Run the actual checkpoint against the TED3 test data originating from `data/test/` using publication-compatible preprocessing, class mapping, post-processing, and evaluator.
- Produce aggregate metrics, per-image metrics, predictions, logs, and comparison against published reference metrics.
- If results are unexpectedly poor, diagnose loader/class mapping/preprocessing/evaluator/checkpoint compatibility first, fix technical mismatches, and rerun before training.

### B. Publication-compatible supervised experiment

- Train the closest defensible publication-compatible architecture using labeled TED3 data originating from `data/train/`.
- Use a fixed validation protocol derived only from training data.
- Keep `data/test/` untouched until final evaluation.
- Save configs, histories, checkpoints/hashes, validation evidence, metrics, per-image metrics, predictions, runtime/resource information, and exact commands.

### C. Semi-supervised TED3 experiment

Use:

```text
data/TED3-unlabeled-data-15k-pseudo-mask/
```

for the semi-supervised teacher/student workflow.

Run the closest defensible publication-compatible pipeline supported by the paper and released code:

- teacher/labeled pretraining;
- unlabeled/pseudo-label processing;
- burn-in;
- pseudo-label filtering;
- student training;
- teacher EMA updates/distillation;
- publication-compatible architecture/losses/optimizer/LR schedule/query-decoder behavior to the closest defensible extent;
- final evaluation using the same held-out TED3 test protocol.

Smoke runs are debugging only. Do not silently reduce the final experiment to `QuickSemiTransformer`, the 60/20/16 quick split, 128x256 inputs, a toy subset, or a short fake-equivalent run and present it as the final result.

If hardware constraints require adaptation, use scientifically defensible changes such as gradient accumulation where possible, preserve effective protocol intent, and document every deviation.

## FAILURE REPAIR LOOP

For every recoverable error:

1. capture the failing command and logs;
2. identify the root cause;
3. patch the smallest defensible fix;
4. add/adjust a regression or smoke check where appropriate;
5. rerun the failed stage;
6. verify the fix;
7. continue the pipeline.

Do not stop at the first error, setup completion, smoke test, checkpoint evaluation, or one successful experiment.

Do not repeatedly retry unchanged commands.
Do not weaken scientific checks to force PASS.
Do not fabricate metrics, predictions, figures, tables, curves, baselines, or confidence intervals.
Do not copy paper numbers into measured-output files.

## OUTPUT REQUIREMENTS

All new measured experiment outputs must be stored under:

```text
outputs/ted3_reproduction/
```

Generate and preserve:

- aggregate metrics;
- per-image metrics;
- training/validation histories;
- predictions;
- checkpoint hashes;
- configs;
- exact commands;
- runtime/resource information;
- experiment manifests;
- checkpoint-vs-trained-model comparison;
- supervised-vs-semi-supervised comparison;
- statistical analysis supported by actual executions.

Every final run manifest must prove the real dataset roots used. It must explicitly show that TED3 data originated from:

```text
data/train/
data/test/
data/TED3-unlabeled-data-15k-pseudo-mask/
```

or deterministic derived manifests/indexes under:

```text
data/processed/ted3/
```

It must also prove that `data/processed/quick_teeth/` was not used for final TED3 experiments.

## RESEARCH ARTICLE EVALUATION/EXPERIMENT OUTPUTS

Inspect the research article/reference material present in the repository.

Identify every experiment-derived figure, table, metric, and quantitative analysis required by the Evaluation/Experiments section.

Create:

```text
outputs/ted3_reproduction/paper_exports/EVALUATION_COVERAGE_MATRIX.csv
outputs/ted3_reproduction/paper_exports/EVALUATION_ANALYSIS.md
```

For every paper experiment artifact, record:

- paper figure/table identifier;
- purpose/metric;
- source experiment;
- generated output path;
- measured vs published-reference provenance;
- completion/fidelity status.

Generate all defensible experiment artifacts from real outputs, including:

- figures in PNG plus vector/PDF where practical;
- experiment tables in CSV plus readable rendered form;
- training curves;
- metrics comparisons;
- qualitative prediction figures;
- supervised vs semi-supervised comparisons;
- official checkpoint comparisons;
- paper-reference vs measured comparisons with dataset/protocol differences explicitly stated;
- ablations/diagnostics only when supported by actual executions.

## FINAL VALIDATION AND DELIVERY

Before stopping verify:

- no TED3 archive was re-extracted;
- original extracted folders were preserved;
- final simulations consumed TED3 sources / `data/processed/ted3/` manifests;
- no final experiment used `quick_teeth`;
- source-directory identities/manifests are recorded;
- train/test leakage checks were performed;
- figures/tables trace to real measured outputs;
- table values match machine-readable metrics;
- paper values remain explicitly labeled as published references;
- no raw dataset/archive/oversized checkpoint is tracked or staged by Git.

Produce at minimum:

```text
outputs/ted3_reproduction/delivery/
  FINAL_RESULTS.md
  REPRODUCIBILITY_REPORT.md
  ISSUES_AND_FIXES.md
  RUNBOOK_FA.md
  artifact_manifest.json
```

Do not stop merely because training or simulation finishes.

Stop only after the TED3 checkpoint evaluation, supervised experiment, semi-supervised experiment, metrics, predictions, figures, tables, Evaluation/Experiments coverage, manuscript-ready analysis, reproducibility documentation, Persian runbook, and final deliverable have all been generated and integrity-checked.

Successful final status:

```text
COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION
```

Only use:

```text
BLOCKED — EXTERNAL NON-RECOVERABLE
```

for a genuinely external blocker that remains after all locally repairable paths are exhausted. Preserve every completed artifact and provide the exact resumable command/state in that case.
