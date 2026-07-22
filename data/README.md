# Data directory

Large medical image data are downloaded at runtime and are not committed to Git.

```bash
python scripts/download_assets.py --dataset
python scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k
```

Dataset source: `Bryceee/TISI15k-Dataset` on Hugging Face.
