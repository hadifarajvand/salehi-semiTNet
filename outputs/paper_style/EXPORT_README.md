# SemiTNet Paper-Aligned Evaluation Export

This folder mirrors the paper's main evaluation artifact categories for client delivery while keeping published reference values and measured execution outputs provenance-separated.

## Main paper counterparts
- Figure 1: architecture
- Figure 2: semi-supervised teacher/student workflow
- Figure 3: training loss + precision
- Figure 4: overall / fully dentate / partially edentulous radar presentation
- Figure 5: measured qualitative model output
- Table 1: dataset distribution
- Table 2: overall performance metrics
- Table 3: subgroup performance metrics

## Persian evaluation notebook
- `notebooks/SemiTNet_Evaluation_Outputs_FA.ipynb`
- Each evaluation artifact is presented with its generation/loading code, Persian explanation, source path, and rendered output reference.

## Primary Figure 5 result
Use:
- `outputs/paper_style/figures/figure_05b_measured_qualitative_only.png`

This is the measured qualitative output intended for the evaluation handoff. Baseline qualitative predictions are not synthesized or attributed to models that were not actually executed.

## Provenance
- measured metrics: `outputs/final/metrics.json`
- measured history: `outputs/final/history.csv`
- measured qualitative source: `outputs/final/figures/predictions.png`
- published references: `paper/reference/table1_dataset.csv`, `table2_overall.csv`, `table3_groups.csv`

The package preserves the paper's metric names, evaluation categories, and visual presentation while keeping the source of every value and image explicit.
