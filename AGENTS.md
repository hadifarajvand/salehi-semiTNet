# Autonomous research agent contract — SemiTNet / TED3

This file is authoritative for any coding/research agent operating in this repository.

## Mission

Once the TED3 archives are present locally, execute an end-to-end, defensible SemiTNet reproduction/reimplementation campaign. Do not stop after setup, a smoke test, a partial experiment, or the first error. Diagnose recoverable failures, patch the repository, rerun the affected stage, and continue until the final evidence package passes the completion checklist below.

## Required local input archives

The agent must auto-discover these exact files anywhere under the repository working tree, preferring `reproduction/assets/dataset/incoming/` when multiple copies exist:

- `TED3-train.tar`
- `TED3-test.tar`
- `TED3-unlabeled-data-15k-pseudo-mask.tar`

These archives and all extracted datasets are local research inputs. They must never be committed, added to Git LFS, copied into the client ZIP, or pushed to GitHub.

Before reading/extracting them:

1. Verify `.gitignore` contains the exact filenames and broad dataset/archive rules.
2. Run `git check-ignore -v` on all three archives.
3. Run `git status --short` and confirm no dataset/checkpoint file is staged or tracked.
4. If any dataset file is tracked, remove it from the Git index without deleting the local file, then continue.

## Non-negotiable scientific rules

1. **Measured outputs only.** Never copy paper metrics into measured-output files, never fabricate predictions, curves, tables, baselines, confidence intervals, or missing experiment results.
2. **Paper reference values and measured values must remain separate.** Any paper number used for comparison must be labeled `published_reference` and carry source/provenance.
3. Preserve the historical reduced baseline under `outputs/final/`; do not overwrite it.
4. TED3 is a defensible public-author dataset for the new campaign, but it must not be silently claimed to be byte-identical to the unavailable original TSI15k split. State the dataset/split relationship explicitly in every final report.
5. Do not tune directly on the final test set. Use train/validation partitions or fixed validation identities for model selection; reserve test for final evaluation.
6. Every experiment must have a machine-readable manifest: git SHA, dataset hashes/manifests, code/config, seed, environment, device, checkpoint hash, start/end time, and exact command.
7. Every failure that can be repaired locally must be repaired and rerun. Do not merely document a stack trace and stop.

## Execution order

### Phase 0 — repository and data preflight

- Snapshot `git rev-parse HEAD` and `git status`.
- Discover the three archives.
- Compute SHA256 and byte size for each archive.
- Inspect tar contents before extraction; reject unsafe paths/symlinks/path traversal.
- Extract into ignored paths under `reproduction/assets/dataset/ted3/`.
- Produce `outputs/ted3_reproduction/preflight/` with archive hashes, extraction manifest, disk-usage report, environment report, and preflight status.

### Phase 1 — dataset forensic audit

Audit train/test/unlabeled data before training:

- exact image counts and annotation counts;
- class/category mapping and tooth-ID semantics;
- image dimensions/formats;
- duplicate and near-duplicate checks where computationally practical;
- exact duplicate leakage across train/test/unlabeled using cryptographic hashes;
- missing/corrupt images and annotations;
- train/test identity lists and hashes;
- labeled vs pseudo-labeled vs unlabeled provenance;
- whether the provided test set can be reconciled with the paper-reported 191-case test protocol;
- fully dentate / partially edentulous grouping if recoverable without guessing.

Write all findings under `outputs/ted3_reproduction/dataset_audit/` and create a clear `DATASET_AUDIT.md` plus machine-readable JSON/CSV manifests.

### Phase 2 — publication-compatible implementation freeze

Use the authors' released SemiTNet/SemiT-SAM code and checkpoint as primary implementation evidence, while documenting paper-vs-code discrepancies already recorded in `reproduction/PAPER_CODE_DISCREPANCIES.md`.

Freeze and record:

- backbone/model architecture;
- input preprocessing and 1024-size policy;
- class mapping;
- 100-query decoder behavior;
- losses and weights;
- teacher/student initialization;
- burn-in, pseudo-label filtering, EMA/distillation behavior;
- optimizer, LR schedule, iteration budget;
- evaluator and post-processing thresholds;
- dependency versions.

Do not blindly use a newer upstream default when it contradicts the paper. Resolve each discrepancy explicitly and record the chosen configuration in a completed method manifest derived from `reproduction/paper_method_manifest.template.json`.

### Phase 3 — official checkpoint evaluation first

Before expensive retraining:

- acquire/verify the pinned author checkpoint;
- run it on the audited TED3 test data using the publication-compatible pipeline;
- generate per-image predictions and metrics;
- compare with published reference metrics without treating TED3 results as exact TSI15k reproduction unless identity equivalence is established.

Required outputs under `outputs/ted3_reproduction/checkpoint_eval/`:

- `metrics.json`
- `per_image_metrics.csv`
- `run_manifest.json`
- `checkpoint_sha256.txt`
- prediction visualizations
- raw evaluator output/logs
- `CHECKPOINT_EVALUATION.md`

If this stage fails technically, debug and continue. If performance is poor, diagnose dataset/class/preprocessing/evaluator mismatch before changing training code.

### Phase 4 — supervised controlled experiment

Train the publication-compatible architecture on labeled TED3 training data with a fixed validation protocol and evaluate once on test.

Save:

- complete configs;
- training history;
- checkpoints and hashes;
- validation selection evidence;
- final metrics/per-image metrics;
- qualitative predictions;
- runtime/resource statistics.

Output: `outputs/ted3_reproduction/supervised/`.

### Phase 5 — semi-supervised TED3 experiment

Use the large unlabeled TED3 archive for the teacher/student experiment. Preserve label hiding/provenance; pseudo masks supplied by the archive must be identified as such and must not be accidentally treated as ground-truth test labels.

Implement and run the closest defensible SemiTNet workflow supported by the publication and released code:

- teacher/labeled pretraining;
- unlabeled/pseudo-label processing;
- burn-in;
- student training;
- teacher EMA updates/distillation;
- final evaluation with the same evaluator used for the supervised experiment.

Output: `outputs/ted3_reproduction/semi_supervised/` with the same provenance requirements as Phase 4.

### Phase 6 — experiment comparison and statistical analysis

Compare, on the same evaluation protocol:

1. historical reduced run (context only);
2. released author checkpoint on TED3;
3. supervised publication-compatible run;
4. semi-supervised TED3 run.

Where repeated seeds are computationally feasible, report mean/std and confidence intervals. If only one authoritative full run is feasible, state that limitation explicitly and do not invent uncertainty estimates.

Create:

- `outputs/ted3_reproduction/comparison/metrics_comparison.csv`
- `outputs/ted3_reproduction/comparison/per_experiment_summary.json`
- `outputs/ted3_reproduction/comparison/STATISTICAL_ANALYSIS.md`
- ablation/diagnostic tables when supported by actual runs.

### Phase 7 — paper/evaluation-section artifact coverage

Inspect the research article/reference material already present in the repository and build an artifact coverage matrix for every experiment-derived figure, table, and quantitative statement in the evaluation/experiments section.

For each paper artifact, record:

- paper figure/table identifier;
- purpose/metric;
- source experiment;
- generated replacement/comparison artifact;
- measured vs published-reference provenance;
- fidelity/status (`generated`, `not applicable`, `blocked by unavailable source output`).

Do not generate architectural/conceptual figures as if they were experiment outputs unless explicitly needed for the deliverable.

Required outputs:

- `outputs/ted3_reproduction/paper_exports/EVALUATION_COVERAGE_MATRIX.csv`
- all measured figures in PNG plus vector/PDF form where practical;
- all experiment tables in CSV plus a human-readable rendered form;
- `outputs/ted3_reproduction/paper_exports/EVALUATION_ANALYSIS.md` explaining the results and gaps in language suitable for the manuscript evaluation/experiment section;
- a paper-vs-measured comparison report that clearly separates datasets and protocols.

### Phase 8 — final validation and delivery

Run integrity checks over every final artifact:

- all required metrics finite and sourced from real outputs;
- figures readable and traceable to source data;
- table values agree with machine-readable metrics;
- no paper reference value is mislabeled as measured;
- no train/test leakage detected or left unexplained;
- no large dataset/archive/checkpoint accidentally tracked by Git;
- deterministic commands/runbook are complete enough for another researcher to rerun;
- repository remains readable to a user unfamiliar with the codebase.

Create a final package under:

`outputs/ted3_reproduction/delivery/`

containing only code/config/report/result artifacts appropriate for delivery, not raw datasets or oversized checkpoints unless explicitly required and legally redistributable.

Required final reports:

- `FINAL_RESULTS.md`
- `REPRODUCIBILITY_REPORT.md`
- `ISSUES_AND_FIXES.md`
- `RUNBOOK_FA.md` (Persian execution/runbook document)
- `artifact_manifest.json` with hashes and provenance

## Failure-handling policy

The agent must operate as a repair loop:

1. capture the failing command and logs;
2. identify the root cause;
3. patch the smallest defensible fix;
4. add/adjust a regression or smoke check when appropriate;
5. rerun the failed stage;
6. continue downstream only after the stage passes.

Do not repeatedly retry unchanged commands. Do not hide failures by weakening validation, deleting checks, substituting fabricated outputs, or changing the scientific question.

A truly external/non-recoverable blocker (for example corrupted/unreadable local archives with no valid copy, unavailable required credentials, or insufficient storage/compute that makes execution impossible) must be documented precisely. Preserve all completed artifacts and leave an exact resumable command/state. Otherwise continue until completion.

## Completion checklist — the agent may stop only when all applicable items are satisfied

- [ ] All three local archives discovered, ignored by Git, hashed, safely extracted and audited.
- [ ] Dataset leakage/corruption/class mapping audit completed.
- [ ] Publication-compatible method manifest frozen.
- [ ] Official checkpoint evaluation completed with measured outputs.
- [ ] Supervised experiment completed and validated.
- [ ] Semi-supervised experiment completed and validated.
- [ ] Per-image and aggregate metrics saved for every completed experiment.
- [ ] Required evaluation/experiment figures generated from real outputs.
- [ ] Required evaluation/experiment tables generated from real outputs.
- [ ] Evaluation coverage matrix complete.
- [ ] Paper-vs-measured comparison and methodological limitations documented.
- [ ] Persian runbook complete.
- [ ] Final deliverable package built and integrity-checked.
- [ ] `git status` confirms no dataset archives/raw data/checkpoints are accidentally staged or tracked.

The final status must be one of:

- `COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION` when all required local experiments and deliverables above are complete; or
- `BLOCKED — EXTERNAL NON-RECOVERABLE` only with exact evidence, completed partial artifacts, and a resumable next command.
