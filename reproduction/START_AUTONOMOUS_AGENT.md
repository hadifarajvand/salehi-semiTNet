# Exact autonomous execution prompt

Copy the prompt below verbatim into the coding/research agent after the three TED3 archives have finished downloading into the repository `data/` folder or one of its subdirectories.

---

You are the autonomous research/reproduction agent for this repository. Work directly inside the current `salehi-semiTNet` repository and execute the entire task end-to-end without asking me to supervise intermediate failures.

First, read `AGENTS.md` completely and treat it as the authoritative scientific and execution contract.

## CRITICAL INPUT RULE — USE THE DOWNLOADED DATA IN `data/`

The three required TED3 archives are already downloaded locally somewhere under the repository `data/` directory. They are the authoritative dataset inputs for this campaign:

- `TED3-train.tar`
- `TED3-test.tar`
- `TED3-unlabeled-data-15k-pseudo-mask.tar`

Do **not** redownload another substitute dataset. Do **not** use the old 598-image `quick_teeth` dataset for this campaign. Do **not** run `python project.py full` as the TED3 reproduction, because that command belongs to the historical reduced simulation.

Your first command must be:

```bash
python project.py ted3-preflight
```

The preflight must discover the authoritative archives from `data/`, verify that they are ignored and not tracked by Git, hash them, inspect their tar structure, and write the provenance manifest. If preflight fails for a locally repairable reason, fix the repository/input placement safely and rerun it until it passes.

After preflight passes, process the verified archives into:

```text
data/processed/ted3/
```

Use separate, explicit processed roots for train, test, and unlabeled data. Preserve the original archives unchanged. Perform safe extraction only after tar validation. Build machine-readable manifests that map every processed file back to its source archive/member and record SHA256/identity information where practical.

Before any training, inspect the actual extracted TED3 structure and adapt the loaders/converters to the real archive contents. Do not assume the current legacy dataset loader applies. Implement or repair the TED3-specific preparation/data-loading pipeline based on the files actually present.

## REQUIRED DATA PROCESSING

Process and validate all three datasets:

1. **TED3 train** — identify labeled images and annotations/classes; convert/index them into the format required by the publication-compatible SemiTNet/SemiT-SAM implementation.
2. **TED3 test** — preserve as the held-out evaluation dataset; do not use it for training or hyperparameter tuning.
3. **TED3 unlabeled/pseudo-mask archive** — identify raw unlabeled images, pseudo masks and their provenance. Do not treat pseudo masks as human ground-truth labels. Use the correct unlabeled/pseudo-label role required by the teacher/student pipeline.

Run a forensic audit before simulation:

- exact image and annotation counts;
- corrupt/missing files;
- class/category and tooth-ID mapping;
- dimensions and formats;
- exact duplicate leakage across train/test/unlabeled;
- identity manifests and hashes;
- train/test contamination checks;
- provenance of human labels vs pseudo labels vs unlabeled images;
- comparison of the TED3 test composition with the paper-reported protocol, without claiming exact TSI15k equivalence unless proven.

Do not continue to expensive training until the dataset is demonstrably loadable, class mappings are coherent, train/test leakage is ruled out or explicitly resolved, and a small real-data loader/inference smoke test passes.

## IMPLEMENTATION AND SIMULATION

Use the authors' released SemiTNet/SemiT-SAM implementation and released checkpoint as the primary executable evidence. Resolve paper-vs-code discrepancies explicitly rather than blindly using current upstream defaults.

The campaign must execute, in this order:

### A. Official checkpoint evaluation

- Acquire/verify the pinned official checkpoint if it is not already present.
- Run the actual checkpoint against the processed `data/processed/ted3/` test set using the publication-compatible preprocessing, class mapping, post-processing and evaluator.
- Produce aggregate metrics, per-image metrics, predictions, logs and comparison against published reference metrics.
- If results are unexpectedly poor, diagnose loader/class mapping/preprocessing/evaluator/checkpoint compatibility first. Fix technical mismatches and rerun before moving to training.

### B. Publication-compatible supervised experiment

- Train the closest defensible publication-compatible architecture using the processed TED3 labeled training data.
- Use a fixed validation protocol derived only from training data for model selection.
- Keep the final test set untouched until final evaluation.
- Save configs, histories, checkpoints/hashes, validation evidence, metrics, per-image metrics, predictions, runtime/resource information and exact commands.

### C. Full TED3 semi-supervised simulation

Run the actual teacher/student semi-supervised workflow using:

- labeled data from processed TED3 train;
- the processed large TED3 unlabeled/pseudo-mask archive in its correct non-ground-truth role;
- publication-compatible teacher pretraining/burn-in;
- pseudo-label generation/filtering;
- student optimization;
- EMA/distillation behavior;
- publication-compatible losses, optimizer, LR schedule, decoder/query settings and iteration budget to the closest defensible extent supported by the paper and released code.

Do not silently reduce this to the old `QuickSemiTransformer`, 60/20/16 split, 128×256 input, or a tiny smoke run and call it the final simulation.

Use smoke/short runs only to debug correctness. After they pass, continue to the real experiment configuration. If hardware requires smaller per-device batch size, use scientifically defensible mechanisms such as gradient accumulation/distributed settings where possible and record the effective batch/protocol. Do not silently change the scientific experiment merely to make it finish.

## FAILURE POLICY — FIX AND CONTINUE

Do not stop at the first error.

For every recoverable failure:

1. capture the exact failing command and complete logs;
2. identify the root cause;
3. patch the smallest defensible fix;
4. add or update a smoke/regression check where useful;
5. rerun the failed stage;
6. continue the downstream pipeline after it passes.

You may modify/restructure the codebase as necessary for correctness, TED3 compatibility, readability and reproducibility.

Do not force success by:

- fabricating metrics or predictions;
- copying paper values into measured outputs;
- weakening validation checks;
- training on the final test set;
- treating pseudo labels as human ground truth;
- silently substituting the legacy 598-image dataset;
- silently shrinking the final experiment into a toy run.

Only declare an external blocker when it is genuinely impossible to repair locally, such as corrupted/missing archive bytes, unavailable required credentials/assets with no accessible equivalent, or objectively insufficient compute/storage. Even then preserve all completed artifacts and an exact resumable state.

## REQUIRED OUTPUTS

All new campaign outputs belong under:

```text
outputs/ted3_reproduction/
```

Do not overwrite the historical reduced baseline under `outputs/final/`.

You must deliver real, traceable outputs for every completed experiment:

- aggregate metrics JSON/CSV;
- per-image metrics;
- raw evaluator outputs;
- training/validation histories;
- exact configs and manifests;
- checkpoint hashes;
- qualitative predictions;
- experiment logs;
- runtime/resource summaries;
- supervised vs semi-supervised comparisons;
- official-checkpoint vs trained-model comparisons;
- statistical analysis supported by the actual runs.

## PAPER EVALUATION / EXPERIMENT SECTION COVERAGE

Inspect the research article/reference material already present in the repository. Identify every experiment-derived figure, table, metric and analysis used in its Evaluation/Experiments section.

For each one, either generate the corresponding measured artifact from the new TED3 experiments or explicitly mark why it is not applicable/unavailable. Build:

```text
outputs/ted3_reproduction/paper_exports/EVALUATION_COVERAGE_MATRIX.csv
```

The matrix must map every paper experiment artifact to:

- paper figure/table identifier;
- what it measures;
- source experiment/run;
- generated artifact path;
- measured or published-reference provenance;
- fidelity/status.

Generate all defensible evaluation outputs, including:

- figures as PNG and vector/PDF where practical;
- tables as CSV plus readable rendered forms;
- training curves;
- metric comparisons;
- qualitative prediction panels from real inference outputs;
- supervised vs semi-supervised comparisons;
- paper-reference vs measured comparisons with dataset/protocol differences clearly labeled;
- any ablation/diagnostic analysis supported by real executed experiments.

Write a manuscript-ready but scientifically cautious analysis at:

```text
outputs/ted3_reproduction/paper_exports/EVALUATION_ANALYSIS.md
```

Do not claim that TED3 is the exact unavailable TSI15k split unless the dataset identity audit proves it.

## FINAL VALIDATION AND DELIVERY

Before stopping, verify:

- the final simulations actually consumed processed TED3 data from `data/processed/ted3/`;
- no final metric came from the legacy quick dataset;
- train/test leakage checks passed or are explicitly disclosed;
- figures trace back to real metrics/predictions;
- table values agree with machine-readable outputs;
- paper reference values are never mislabeled as measured;
- no raw TED3 archive, extracted dataset or oversized checkpoint is tracked/staged for Git;
- another researcher can reproduce the workflow from the documented commands.

Produce at minimum:

```text
outputs/ted3_reproduction/delivery/FINAL_RESULTS.md
outputs/ted3_reproduction/delivery/REPRODUCIBILITY_REPORT.md
outputs/ted3_reproduction/delivery/ISSUES_AND_FIXES.md
outputs/ted3_reproduction/delivery/RUNBOOK_FA.md
outputs/ted3_reproduction/delivery/artifact_manifest.json
```

Also create a final machine-readable execution manifest that explicitly records the source archive paths under `data/`, their hashes, the processed TED3 roots used by each experiment, and confirms that the actual final training/evaluation dataloaders used TED3 rather than `quick_teeth`.

Do not stop merely because one simulation finishes. Stop only after the experiment outputs, figures, tables, evaluation analysis, reproducibility report and final delivery package are all generated and integrity-checked.

The successful final status must be exactly:

```text
COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION
```

If and only if a genuinely external, non-recoverable blocker remains after all locally repairable paths are exhausted, use:

```text
BLOCKED — EXTERNAL NON-RECOVERABLE
```

and include exact evidence plus the resumable next command/state.

---
