# Figure comparison

| Figure | Paper structure | Generated structure | Fidelity |
|---|---|---|---|
| Figure 1 | published architecture | `figure_01_semitnet_architecture.png + figure_01b_implemented_reduced_architecture.png` | HIGH structural reference match after fix; implementation truth documented in Figure 1b |
| Figure 2 | distillation workflow | `figure_02_teacher_student_workflow.png` | HIGH workflow-layout match after fix |
| Figure 3 | dual-axis loss + precision | `figure_03_training_loss_precision.png` | HIGH visual-format match; measured series shorter than paper |
| Figure 4 | top + bottom radar geometry | `figure_04_performance_radar.png + figure_04b_measured_reduced_simulation.png` | HIGH paper-reference visual match; measured reduced results separated into Figure 4b |
| Figure 5 | 2-row qualitative comparison | `figure_05_qualitative_outputs.png + figure_05b_measured_qualitative_only.png` | PARTIAL; baselines unavailable and not fabricated; paper-aligned layout plus measured-only Figure 5b supplied |

## Metric comparison

| Metric | Paper SemiTNet | Reduced simulation | Difference |
|---|---:|---:|---:|
| IOU | 94.41 | 33.05651614 | -61.35348386 |
| DICE | 95.45 | 49.68793277 | -45.76206723 |
| PRECISION | 94.74 | 12.52662424 | -82.21337576 |
| RECALL | 97.10 | 12.02895676 | -85.07104324 |
| F1 | 95.90 | 12.27274689 | -83.62725311 |

The reduced simulation is not numerically equivalent to the full-scale paper experiment. The exported figures reproduce the paper’s presentation structure and experimental visualization categories using the available measured outputs.
