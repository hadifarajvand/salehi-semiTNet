# SemiTNet Paper-Aligned Client Export

This folder mirrors the paper's main visual/evaluation artifact set for fast delivery.

## Main paper counterparts
- Figure 1: architecture
- Figure 2: semi-supervised teacher/student distillation workflow
- Figure 3: training loss + precision
- Figure 4: overall / fully dentate / partially edentulous radar charts
- Figure 5: qualitative 2×7 panel structure
- Table 1: dataset distribution
- Table 2: overall performance metrics
- Table 3: subgroup performance metrics

All generated figures are available as PNG/PDF and tables as PNG/PDF/CSV in the client export bundle produced by `python scripts/export_client_paper_package.py`.

## Provenance
- measured reduced metrics: `outputs/final/metrics.json`
- measured reduced history: `outputs/final/history.csv`
- measured qualitative source: `outputs/final/figures/predictions.png`
- published references: `paper/reference/table1_dataset.csv`, `table2_overall.csv`, `table3_groups.csv`

## Important limitation
Figure 5 baseline model outputs were not executed. They are shown as explicit unavailable placeholders rather than fabricated predictions. The measured SemiTNet/ground-truth qualitative source remains available separately in `outputs/final/figures/predictions.png`.

This package matches paper **metrics, artifact categories, panel geometry, and presentation style** without claiming the same numerical experimental results.
