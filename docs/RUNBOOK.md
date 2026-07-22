# راهنمای اجرای SemiTNet

## ۱. حالت‌های اجرا

این پروژه سه حالت دارد:

1. **Smoke:** تست سریع روی دیتاست مصنوعی؛ مناسب بررسی کد روی CPU.
2. **Inference:** استفاده از checkpoint رسمی برای بازتولید نزدیک‌تر به خروجی مقاله.
3. **Full:** آموزش دوباره‌ی teacher و student روی کل TSI15k.

## ۲. Smoke Test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
python scripts/run_smoke.py
python scripts/acceptance_check.py --mode smoke
```

این اجرا یک مدل کوچک Transformer را ابتدا با داده‌ی برچسب‌دار آموزش می‌دهد، سپس یک مرحله‌ی teacher/student با pseudo-label انجام می‌دهد و معیارهای واقعی همان اجرا را ذخیره می‌کند.

## ۳. دریافت و اصلاح کد نویسندگان

```bash
python scripts/bootstrap_upstream.py
```

نسخه‌ی ثابت زیر دریافت می‌شود:

```text
repository: isjinghao/SemiT-SAM
commit: bf66074bee9da9b37fd68454bcbac9140c4f59e2
```

سپس `scripts/patch_upstream.py` این موارد را اصلاح می‌کند:

- حذف مسیر خصوصی `/root/paddlejob/...`
- حذف override اجباری checkpoint و حالت Teacher
- ثبت runtime دیتاست TSI15k با متغیر `TSI15K_ROOT`
- تنظیم 32 کلاس
- تنظیم batch کلی مقاله روی 16
- حفظ 26,250 iteration و LR stepهای 24,000 و 25,000
- افزودن اسکریپت‌های قابل تنظیم برای train/eval

## ۴. نصب محیط کامل

پیشنهاد اصلی Linux + NVIDIA CUDA است:

```bash
bash scripts/setup_full_env.sh
```

این اسکریپت محیط upstream را بر اساس Python 3.10، PyTorch 1.13.1 و CUDA 11.7 آماده می‌کند و extensionهای Mask2Former را می‌سازد.

## ۵. دانلود دیتاست و checkpoint

```bash
python scripts/download_assets.py --all
```

مسیرها:

```text
data/raw/TSI15k/
assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
```

برای فقط checkpoint:

```bash
python scripts/download_assets.py --checkpoint
```

برای فقط دیتاست:

```bash
python scripts/download_assets.py --dataset
```

## ۶. بررسی و آماده‌سازی دیتاست

```bash
python scripts/inspect_dataset.py --root data/raw/TSI15k
python scripts/prepare_dataset.py \
  --source data/raw/TSI15k \
  --dest data/processed/TSI15k
```

اسکریپت ابتدا به دنبال split و annotation رسمی می‌گردد. اگر split اصلی در archive وجود داشته باشد همان را حفظ می‌کند. اگر فقط یک مجموعه‌ی labeled موجود باشد، split 7:1 را با seed ثابت می‌سازد و این موضوع را در manifest ثبت می‌کند؛ چنین splitی دقیقاً split نویسندگان محسوب نمی‌شود.

ساختار نهایی:

```text
data/processed/TSI15k/
├── labeled/train/images/
├── labeled/train/annotations.json
├── labeled/test/images/
├── labeled/test/annotations.json
├── unlabeled/images/
├── unlabeled/images.json
└── dataset_manifest.json
```

## ۷. اجرای checkpoint رسمی

```bash
bash run_inference.sh
```

یا:

```bash
python scripts/run_official_inference.py \
  --dataset data/processed/TSI15k \
  --checkpoint assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
```

خروجی:

```text
outputs/inference/
├── metrics.json
├── predictions/
├── figures/
├── raw_evaluator_output.json
└── PAPER_COMPARISON.md
```

## ۸. آموزش کامل

```bash
bash run_full.sh
```

مرحله‌ی اول teacher را با داده‌های labeled آموزش می‌دهد. مرحله‌ی دوم student را با labeled + unlabeled و EMA teacher آموزش می‌دهد.

پارامترهای اصلی:

```text
input size:       1024
classes:          32
queries:          100
decoder layers:   9
optimizer:        AdamW
learning rate:    1e-4
iterations:       26250
lr steps:         24000, 25000
total batch:      16
EMA decay:        0.9996
pseudo class threshold: 0.7
pseudo mask threshold: 0.5
unsupervised loss weight: 2.0
```

مقاله iteration دقیق شروع distillation را منتشر نکرده است. wrapper مقدار `3000` را به‌عنوان مقدار inferred از کد debug باقی‌مانده در upstream قرار می‌دهد و آن را در manifest ثبت می‌کند. این مقدار با متغیر `BURNIN_ITER` قابل تغییر است.

## ۹. مقایسه با مقاله

```bash
python scripts/compare_to_paper.py --results outputs/inference/metrics.json
```

مقادیر مرجع مقاله:

```text
IoU:       94.41
Dice:      95.45
Precision: 94.74
Recall:    97.10
F1:        95.90
```

## ۱۰. بسته‌بندی

```bash
make package
```

دیتاست خام و checkpoint بزرگ داخل ZIP کد قرار نمی‌گیرند؛ downloader و checksum آن‌ها داخل بسته قرار دارد.

## ۱۱. اجرای Docker روی GPU

روی سیستمی که NVIDIA Container Toolkit نصب است:

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm semitnet bash
```

داخل container:

```bash
python scripts/download_assets.py --all
python scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k
bash run_inference.sh
```

### وزن اولیه در آموزش کامل

مسیر آموزش کامل به‌صورت پیش‌فرض با `MODEL.WEIGHTS=""` شروع می‌شود تا از checkpoint نهایی مقاله به‌عنوان نقطه‌ی شروع استفاده نشود. اگر وزن اولیه‌ی سازگار با TinyViT/Detectron2 دارید، آن را مشخص کنید:

```bash
python scripts/run_full.py \
  --dataset data/processed/TSI15k \
  --num-gpus 8 \
  --batch-size 16 \
  --init-weights /path/to/compatible_initial_weights.pth
```

checkpoint منتشرشده‌ی SemiTNet برای **inference و بازتولید نتایج نهایی** در نظر گرفته شده است؛ استفاده از آن برای شروع آموزش از ابتدا باعث data leakage می‌شود.
