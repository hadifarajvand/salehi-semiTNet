# راهنمای اجرای پروژه SemiTNet

## ۱. حالت‌های اجرا

این پروژه سه حالت اصلی دارد:

1. **Smoke:** تست سریع روی دیتاست مصنوعی برای بررسی سالم بودن کد روی CPU.
2. **Inference:** اجرای مدل روی test set آماده‌شده با checkpoint منتشرشده.
3. **Full:** آموزش teacher و سپس آموزش نیمه‌نظارتی student روی TSI15k.

## ۲. Smoke Test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
python scripts/run_smoke.py
python scripts/acceptance_check.py --mode smoke
```

این اجرا یک مدل کوچک Transformer را ابتدا با داده‌ی برچسب‌دار آموزش می‌دهد، سپس مرحله‌ی teacher/student با pseudo-label را اجرا می‌کند و معیارها و خروجی‌های واقعی همان اجرا را ذخیره می‌کند.

## ۳. آماده‌سازی هسته‌ی مدل

```bash
make bootstrap
```

این مرحله وابستگی‌های لازم برای اجرای معماری اصلی را در `vendor/` آماده می‌کند و تنظیمات مسیرها، کلاس‌ها و اجرای محلی را برای محیط پروژه اعمال می‌کند.

## ۴. نصب محیط کامل

برای اجرای اصلی، Linux + NVIDIA CUDA پیشنهاد می‌شود:

```bash
make setup-full
```

یا:

```bash
bash scripts/setup_full_env.sh
```

محیط سازگار پروژه بر اساس Python 3.10، PyTorch 1.13.1 و CUDA 11.7 آماده می‌شود و extensionهای لازم برای Detectron2/Mask2Former ساخته می‌شوند.

## ۵. دانلود دیتاست و checkpoint

```bash
make download
```

یا:

```bash
python scripts/download_assets.py --all
```

مسیرهای اصلی:

```text
data/raw/TSI15k/
assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
```

فقط checkpoint:

```bash
python scripts/download_assets.py --checkpoint
```

فقط دیتاست:

```bash
python scripts/download_assets.py --dataset
```

پس از دانلود، checksum فایل checkpoint بررسی می‌شود.

## ۶. بررسی و آماده‌سازی دیتاست

```bash
python scripts/inspect_dataset.py --root data/raw/TSI15k
python scripts/prepare_dataset.py \
  --source data/raw/TSI15k \
  --dest data/processed/TSI15k
```

اسکریپت ابتدا splitها و annotationهای موجود را بررسی می‌کند. اگر split اصلی در archive وجود داشته باشد همان ساختار حفظ می‌شود. اگر فقط مجموعه‌ی labeled موجود باشد، split 7:1 با seed ثابت ساخته و در manifest ثبت می‌شود.

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

## ۷. اجرای مدل با checkpoint آماده

```bash
make inference
```

یا:

```bash
python scripts/run_official_inference.py \
  --dataset data/processed/TSI15k \
  --checkpoint assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
```

خروجی‌ها:

```text
outputs/inference/
├── metrics.json
├── coco_instances_results.json
├── raw_evaluator_output.json
├── PAPER_COMPARISON.md
└── figures/
```

این مرحله predictionها را اجرا می‌کند، معیارهای segmentation و identification را محاسبه می‌کند، تصاویر نمونه را تولید می‌کند و خروجی‌ها را با مقادیر مرجع مقاله مقایسه می‌کند.

## ۸. آموزش کامل

```bash
make full
```

یا:

```bash
bash run_full.sh
```

مراحل:

1. آموزش teacher با داده‌های labeled.
2. آموزش student با داده‌های labeled و unlabeled.
3. استفاده از pseudo-label و EMA teacher در مرحله‌ی نیمه‌نظارتی.
4. ذخیره‌ی checkpointها و manifest اجرای آزمایش.
5. ارزیابی checkpoint نهایی با pipeline inference.

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

مقدار پیش‌فرض `BURNIN_ITER=3000` در تنظیمات پروژه قابل تغییر است:

```bash
BURNIN_ITER=3000 NUM_GPUS=8 BATCH_SIZE=16 make full
```

## ۹. تولید و مقایسه‌ی نتایج

برای تولید جدول‌ها و نمودارهای مقادیر مرجع:

```bash
make reference
```

برای مقایسه‌ی خروجی اجرای مدل:

```bash
python scripts/compare_to_paper.py --results outputs/inference/metrics.json
```

مقادیر مرجع ثبت‌شده:

```text
IoU:       94.41
Dice:      95.45
Precision: 94.74
Recall:    97.10
F1:        95.90
```

## ۱۰. بررسی سیستم قبل از اجرا

```bash
make preflight
```

خروجی در:

```text
outputs/PREFLIGHT.json
```

این گزارش وضعیت GPU، CUDA، فضای دیسک، دیتاست و checkpoint را بررسی می‌کند.

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

## ۱۲. بسته‌بندی خروجی

```bash
make package
```

دیتاست خام و checkpoint بزرگ داخل بسته‌ی کد قرار نمی‌گیرند و از طریق downloader پروژه دریافت می‌شوند.

## ۱۳. نکته درباره وزن اولیه

مسیر Full Training به‌صورت پیش‌فرض با وزن نهایی checkpoint شروع نمی‌شود. برای تعیین وزن اولیه‌ی سازگار:

```bash
python scripts/run_full.py \
  --dataset data/processed/TSI15k \
  --num-gpus 8 \
  --batch-size 16 \
  --init-weights /path/to/compatible_initial_weights.pth
```

## ۱۴. اطلاعات وابستگی‌ها

اطلاعات مربوط به کتابخانه‌ها، اجزای ثالث و مجوزهای آن‌ها در فایل زیر نگهداری می‌شود:

```text
docs/THIRD_PARTY_NOTICES.md
```