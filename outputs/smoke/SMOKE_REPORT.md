# Executed smoke report

This is an actual CPU execution of the package's small teacher/student Transformer validation path. It is not a TSI15k or journal-paper result.

| Metric | Result |
|---|---:|
| IoU | 69.71% |
| Dice | 82.13% |
| Precision | 70.34% |
| Recall | 98.72% |
| F1 | 82.13% |
| Runtime | 3.80 seconds (training/evaluation core) |

Configuration: 24 labeled synthetic panoramics, 24 unlabeled synthetic panoramics, 8 validation images, six teacher epochs and four student epochs on CPU.
