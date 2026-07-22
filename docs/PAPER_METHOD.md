# Paper method summary

SemiTNet is a semi-supervised Transformer-based instance segmentation and identification model for panoramic radiographs.

## Data

- Total panoramic radiographs: 16,317
- Labeled: 1,589
- Unlabeled: 14,728
- Labeled training: 1,398
- Test: 191
- Test groups: 40 fully dentate and 151 partially edentulous

## Architecture

- MobileSAM/TinyViT-style image encoder
- Simple feature pyramid
- GEM/MaskDINO-like query-based instance segmentation
- 100 object queries
- 9 Transformer decoder layers
- 32 tooth classes

## Training

1. Train a teacher on labeled images.
2. Initialize teacher/student semi-supervised training.
3. Generate pseudo-labels on unlabeled images.
4. Apply strong perturbation to student inputs.
5. Update the teacher through EMA.

## Reported compute and optimization

- 8 × NVIDIA V100 32GB
- total batch size 16
- 26,250 iterations
- AdamW, learning rate 1e-4
- LR drops at 24,000 and 25,000
- 1024-pixel input setting
- approximately 6 hours reported training time
