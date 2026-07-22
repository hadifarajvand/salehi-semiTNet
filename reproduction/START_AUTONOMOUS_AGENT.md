# One-shot autonomous execution handoff

Use this only after all three TED3 archives have finished downloading and are present somewhere inside the local repository working tree.

## Instruction to the coding/research agent

Read `AGENTS.md` in the repository root and treat it as the authoritative execution contract. Execute the complete TED3 reproduction/reimplementation campaign end-to-end.

Start with:

```bash
python project.py ted3-preflight
```

Then continue through every phase in `AGENTS.md`: dataset audit, publication-compatible method freeze, official checkpoint evaluation, supervised experiment, semi-supervised experiment, comparison/statistical analysis, evaluation-section figure/table coverage, final validation, Persian runbook, and delivery packaging.

Do not stop at setup, smoke tests, a partial run, or the first failure. For every locally recoverable issue, capture the logs, identify the root cause, patch the smallest defensible fix, rerun the failed stage, and continue. Do not weaken scientific checks or fabricate missing results to force a PASS.

All final measured numbers, figures, tables, predictions, histories, and statistical analyses must originate from real executions. Published paper values may only appear as explicitly labeled reference values.

Do not commit or push raw TED3 archives, extracted datasets, oversized checkpoints, secrets, or generated caches. Verify Git ignore/tracking state before and after the campaign.

Do not overwrite the historical reduced baseline in `outputs/final/`.

The authoritative new campaign output root is:

```text
outputs/ted3_reproduction/
```

Before stopping, satisfy every applicable checkbox in the `AGENTS.md` completion checklist and produce the final status:

```text
COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION
```

Only use `BLOCKED — EXTERNAL NON-RECOVERABLE` for a genuinely external blocker that cannot be fixed in the repository or execution environment; preserve completed outputs and provide an exact resumable next command/state in that case.
