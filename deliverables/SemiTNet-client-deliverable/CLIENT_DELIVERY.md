# Client Delivery Summary

This package contains the executable SemiTNet reduced-scale simulation code, verified dataset manifests, and the latest validated measured outputs.

## Dataset
- Dataset: Teeth Segmentation on Dental X-ray Images
- Verified source images: 598
- Classes: 32
- Execution split: 60 labeled / 20 pseudo-label / 16 test

## Measured outputs
- IoU: 33.05651614205745
- Dice: 49.68793276800474
- Precision: 12.52662423999735
- Recall: 12.02895676318772
- F1: 12.272746893320026
- Selected checkpoint: student

## Run
```bash
python project.py install
python project.py download
python project.py full
```
