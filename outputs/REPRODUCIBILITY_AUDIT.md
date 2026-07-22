# SemiTNet Reproducibility Audit

Overall classification: **REDUCED_BASELINE_VALID_NOT_PAPER_EQUIVALENT**

This audit separates artifact integrity from paper equivalence. The existing PASS status remains valid for the reduced measured simulation and its packaging, but it is not evidence that the published SemiTNet experiment has been numerically reproduced.

## Evidence gates

| Gate | Status | Meaning |
|---|---|---|
| G0 — reduced baseline integrity | **PASS** | Existing canonical measured outputs and manifests are internally consistent. |
| G1 — official checkpoint equivalence | **BLOCKED** | No evaluation of the exact released checkpoint on the exact 191-image paper test set is present. |
| G2 — methodological faithfulness | **BLOCKED** | No frozen publication-faithful method manifest exists yet; the current reduced run explicitly fails paper equivalence. |
| G3 — full training reproduction | **BLOCKED** | No paper-faithful full training output exists, and G1/G2 are prerequisites. |

## Current defensible statement

> Measured reduced-scale re-simulation; not numerically equivalent to the published full-scale experiment.

## Verified current baseline

- dataset: `Teeth Segmentation on Dental X-ray Images`
- verified source images: 598
- actual execution split: 60 labeled + 20 label-hidden pseudo + 16 held-out test
- model: `QuickSemiTransformer`
- input: 128×256
- pseudo threshold: 0.35
- unsupervised weight: 0.05
- confident pseudo-label pixels used in the canonical run: 0

## Numerical gap versus published SemiTNet

| Metric | Paper | Reduced measured run | Difference (percentage points) |
|---|---:|---:|---:|
| IoU | 94.41 | 33.06 | -61.35 |
| Dice | 95.45 | 49.69 | -45.76 |
| Precision | 94.74 | 12.53 | -82.21 |
| Recall | 97.10 | 12.03 | -85.07 |
| F1 | 95.90 | 12.27 | -83.63 |

These differences are too large to treat as ordinary run variance. The current run is a materially different experiment and should not be tuned as though it were the publication implementation.

## Why the current run is not paper-equivalent

1. The paper protocol uses TSI15k with 16,317 images: 1398 labeled training, 14,728 unlabeled training and 191 test images. The reduced run uses a different 598-image dataset and only 96 images in the actual experiment.
2. The reduced runner converts images to grayscale and resizes them to 128×256. The publication-compatible pipeline operates at 1024 input size with RGB preprocessing.
3. `QuickSemiTransformer` is a lightweight surrogate. It is not the publication MobileSAM/TinyViT + multi-scale/query/mask-decoder system.
4. The reduced training budget is 8 teacher epochs and 2 student passes. The paper reports a 26,250-iteration schedule.
5. The reduced pseudo-label settings (`0.35`, unsupervised weight `0.05`) are not the publication protocol.
6. The reduced-run evaluator has not been proven identical to the publication evaluator.

## Exact author checkpoint provenance pinned for G1

- repository: `Bryceee/SemiTNet`
- revision: `2d79fa571467b159fbcf279d8676bd07fd0dcc9a`
- file: `SemiTNet_Tooth_Instance_Segmentation_32Classes.pth`
- expected size: `338584301` bytes
- SHA256: `8364853c7632a491fd66108e23a536fa68e9a9c9b416b21c69143a4d02a26c0a`

This checkpoint must be evaluated before retraining. If it cannot reproduce the paper metrics within the declared G1 tolerances, the problem is in dataset identity, class mapping, preprocessing, checkpoint loading, post-processing or evaluation—not lack of training time.

## Highest-priority experiment

Run the exact author checkpoint on the exact 191-image paper test set with a publication-compatible evaluator.

Acceptance tolerances:

- IoU: ±1.0 point
- Dice: ±1.0 point
- Precision: ±2.0 points
- Recall: ±2.0 points
- F1: ±2.0 points

If this passes, the repository may claim that the published checkpoint evaluation has been reproduced within declared tolerance. Only then should G2 be frozen and full paper-faithful retraining begin.

## Stop rules

Do not substitute a different dataset if the exact TSI15k asset or test identities cannot be recovered. Do not use today's upstream `main` as proof of the 2024 publication configuration without documenting discrepancies. Do not overwrite the reduced baseline with new runs. Do not fabricate missing baseline predictions or infer numerical equivalence from visual figure similarity.
