# Visual Fidelity Audit

## Verdict
**Client-ready for paper-aligned presentation, with one explicit limitation: Figure 5 baseline qualitative predictions were not executed and are not fabricated.**

## Paper-to-export alignment
- **Figure 1 — PASS:** same architecture categories: image encoder, feature pyramid, query initialization, mask decoder.
- **Figure 2 — PASS:** teacher/student branches, pseudo-label path, supervised loss, unsupervised loss, and EMA update are represented.
- **Figure 3 — PASS:** same two metrics and visual grammar as the paper: red training-loss axis + blue precision axis. The x-axis truthfully uses recorded reduced-run steps rather than pretending to be 26,250 paper iterations.
- **Figure 4 — PASS:** same five metrics, six model references, and top-one/bottom-two radar arrangement.
- **Figure 5 — PARTIAL:** corrected to the paper's 2×7 panel geometry: Mask R-CNN, MPFormer, Mask2Former, MaskDINO, GEM, SemiTNet, Ground Truth. Missing baseline predictions are explicitly marked unavailable. The measured reduced qualitative source remains `outputs/final/figures/predictions.png`.
- **Tables 1–3 — PASS:** same metric categories and paper-facing comparison structure are exported as PNG/PDF/CSV.

## Scientific claim
This package is a **paper-aligned visualization/export package**, not a numerical reproduction of the full paper experiment. Published-reference values and measured reduced-run values remain distinguishable.
