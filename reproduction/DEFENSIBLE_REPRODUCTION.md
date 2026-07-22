# Defensible SemiTNet reproduction protocol

This repository has two deliberately separate evidence tracks.

## Track A — reduced measured simulation

The existing `outputs/final/` run is preserved as a deterministic, measured, reduced-scale teacher/student simulation. It is useful for execution validation, packaging, provenance, and demonstration. It is **not** a numerical reproduction of the published SemiTNet experiment.

Canonical baseline commit:

`aad3a6a86f8e5edb846ceb92dc38999cd88596ce`

Canonical reduced metrics:

- IoU: 33.05651614205745
- Dice: 49.68793276800474
- Precision: 12.52662423999735
- Recall: 12.02895676318772
- F1: 12.272746893320026

The baseline uses a different 598-image public dataset and an execution split of 60 labeled + 20 label-hidden pseudo + 16 held-out test images, downsampled to 128×256, with a lightweight `QuickSemiTransformer`.

The only defensible claim for this track is:

> Measured reduced-scale re-simulation; not numerically equivalent to the published full-scale experiment.

## Track B — paper-equivalence evidence

No result is called a paper reproduction until the following gates pass in order.

### G0 — reduced baseline integrity

Run:

```bash
python project.py audit
```

Expected state for the current repository: G0 PASS. This confirms internal provenance and canonical metric consistency only.

### G1 — official checkpoint equivalence

This is the highest-priority experiment and must be completed before retraining.

1. Acquire the exact author-released checkpoint.
2. Verify its SHA256 and byte size.
3. Acquire the exact paper-cited TSI15k/TISI15k archive. Do not substitute another dental dataset.
4. Reconstruct the exact 191-image test identity list and hash it.
5. Freeze publication-compatible class mapping, preprocessing and evaluator.
6. Run the released author checkpoint on exactly those 191 test images.
7. Write:
   - `outputs/paper_reproduction/pretrained/metrics.json`
   - `outputs/paper_reproduction/pretrained/run_manifest.json`
8. Run `python project.py audit` again.

The released checkpoint provenance pinned by this repository is:

- Hugging Face repository: `Bryceee/SemiTNet`
- revision: `2d79fa571467b159fbcf279d8676bd07fd0dcc9a`
- file: `SemiTNet_Tooth_Instance_Segmentation_32Classes.pth`
- expected bytes: `338584301`
- SHA256: `8364853c7632a491fd66108e23a536fa68e9a9c9b416b21c69143a4d02a26c0a`

Acquire/verify assets with:

```bash
python scripts/download_paper_assets.py --checkpoint
python scripts/download_paper_assets.py --dataset
```

If the exact paper dataset location is unavailable, **stop**. Do not silently use the current 598-image dataset, a newer dataset, or a reconstructed random split for a paper-equivalence claim.

G1 metric tolerances are intentionally narrow enough to detect an evaluator/data mismatch:

- IoU: ±1.0 percentage point
- Dice: ±1.0
- Precision: ±2.0
- Recall: ±2.0
- F1: ±2.0

Reference values:

- IoU: 94.41
- Dice: 95.45
- Precision: 94.74
- Recall: 97.10
- F1: 95.90

If G1 fails, do **not** start full retraining. Fix dataset identities, class mapping, evaluator, transforms, checkpoint loading, or publication-version code first.

### G2 — methodological faithfulness

Freeze and record all of the following before training:

- exact labeled training identities: 1398
- exact unlabeled identities: 14728
- exact test identities: 191
- 40 fully dentate and 151 partially edentulous test cases where recoverable from publication artifacts
- publication-compatible 32-tooth class mapping
- RGB input and 1024 processing size
- MobileSAM/TinyViT-based backbone and publication-compatible multi-scale/query/mask-decoder implementation
- 100 object queries
- teacher pretraining initialization
- burn-in and teacher/student distillation schedule
- pseudo-label class and mask filtering rules
- EMA update behavior
- AdamW optimizer
- base LR 1e-4
- LR steps 24000 and 25000
- 26250 iterations
- exact evaluator and post-processing thresholds
- package versions, CUDA/cuDNN/PyTorch/Detectron2 versions
- seeds and distributed-training settings

Any conflict between paper text and current upstream code must be written to a discrepancy table and tested explicitly. Current upstream `main` is not automatically considered the publication configuration.

### G3 — full training reproduction

Only after G1 and G2:

1. Run one full paper-faithful training campaign.
2. Evaluate with the already validated G1 evaluator.
3. Save the complete manifest and immutable raw outputs.
4. For claims about stability/repeatability rather than one-run reproduction, run at least three declared seeds.

Required output location:

```text
outputs/paper_reproduction/training/
  metrics.json
  run_manifest.json
  history.*
  per_image_metrics.*
  test_identity_manifest.*
  environment.*
  checkpoint_sha256.*
```

Do not overwrite `outputs/final/`; that directory is the historical reduced baseline.

## Mandatory stop rules

Stop and report BLOCKED rather than manufacturing progress when any of these occur:

- exact paper dataset cannot be obtained or its identities cannot be reconstructed;
- released checkpoint hash does not match;
- author checkpoint does not reproduce paper metrics within the G1 tolerances;
- evaluator/class mapping is uncertain;
- current upstream implementation contradicts the paper and no publication-era state can be justified;
- measured predictions or baseline outputs required by a figure are unavailable.

## Figure policy

Paper-style figures may use published reference values only when labeled as published reference content. Measured panels must use measured outputs only. Missing baseline predictions must remain missing/unavailable rather than being synthesized.

## Claim ladder

- G0 only: **reduced measured re-simulation**
- G0 + G1: **published checkpoint evaluation reproduced within declared tolerance**
- G0 + G1 + G2: **methodologically faithful reproduction setup**
- G0 + G1 + G2 + G3: **training reproduction evidence**

This ladder is intentionally conservative. It prevents a visually similar figure set or a successful smoke run from being mistaken for numerical reproduction of the paper.
