# Visual Fidelity Audit

## Verdict
**Client-ready for paper-aligned evaluation presentation without fabricated metrics or predictions.**

## Paper-to-export alignment
- **Figure 1 — PASS:** architecture categories align with the paper presentation: image encoder, feature pyramid, query initialization, and mask decoder.
- **Figure 2 — PASS:** teacher/student branches, pseudo-label path, supervised loss, unsupervised loss, and EMA update are represented.
- **Figure 3 — PASS:** the same two plotted quantities and dual-axis visual grammar are used: training loss and precision. Values are read from the recorded execution history.
- **Figure 4 — PASS:** the same five evaluation metrics are used: IoU, Dice, Precision, Recall, and F1; the paper-reference comparison and measured-execution radar are kept separate.
- **Figure 5 — PASS for measured output:** the client-facing qualitative result is `figure_05b_measured_qualitative_only.png`. Unavailable baseline predictions are not synthesized and are not presented as executed outputs.
- **Tables 1–3 — PASS:** the paper-facing metric categories and comparison structures are exported as PNG/PDF/CSV.

## Persian code-to-output documentation
`notebooks/SemiTNet_Evaluation_Outputs_FA.ipynb` places each evaluation artifact alongside its generating/loading code snippet, Persian explanation, provenance, and rendered output path.

## Provenance rule
Published-reference values and measured execution outputs remain distinguishable in the underlying source files. No metric or qualitative prediction is fabricated.
