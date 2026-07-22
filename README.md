# پیاده‌سازی و شبیه‌سازی SemiTNet برای سگمنتیشن و شناسایی دندان

**زبان:** **فارسی** | [English](README.en.md)

این پروژه یک پایپ‌لاین کامل برای آموزش، ارزیابی و اجرای مدل‌های سگمنتیشن و شناسایی دندان روی تصاویر پانورامیک X-ray فراهم می‌کند و بر اساس روش مطرح‌شده در مقاله‌ی زیر طراحی شده است:

> **A Semi-Supervised Transformer-Based Deep Learning Framework for Automated Tooth Segmentation and Identification on Panoramic Radiographs**  
> *Diagnostics*, 2024, 14(17), 1948 — DOI: `10.3390/diagnostics14171948`

کدهای این مخزن برای مدیریت دیتاست، آماده‌سازی داده، اجرای Smoke Test، آموزش teacher/student، ارزیابی، محاسبه‌ی معیارها، تولید جدول‌ها و خروجی‌های تصویری و اجرای کامل آزمایش‌ها آماده شده‌اند.

## اجرای سریع Smoke Test

Smoke Test بدون نیاز به دیتاست اصلی و روی CPU اجرا می‌شود و سالم بودن مسیر آموزش teacher/student، محاسبه‌ی معیارها و تولید خروجی را بررسی می‌کند.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-smoke.txt
make smoke
```

خروجی در مسیر زیر ذخیره می‌شود:

```text
outputs/smoke/
├── metrics.json
├── history.csv
├── run_manifest.json
└── figures/
```

> اعداد Smoke Test مربوط به تست سریع نرم‌افزار هستند و با نتایج اجرای کامل روی TSI15k تفاوت دارند.

## اجرای مدل روی دیتاست اصلی

برای آماده‌سازی محیط و اجرای مدل با checkpoint منتشرشده:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

برای آموزش کامل teacher و student روی دیتاست TSI15k:

```bash
make full
```

### مراحل اصلی

```text
bootstrap
  آماده‌سازی وابستگی‌های هسته‌ی مدل

setup-full
  نصب محیط CUDA / Detectron2 / Mask2Former

download
  دریافت دیتاست TSI15k و checkpoint مدل

prepare
  بررسی ساختار فایل‌ها، annotationها و splitها

inference
  اجرای مدل روی test set
  تولید معیارها، predictionها و گزارش نتایج

full
  مرحله ۱: آموزش teacher با داده‌های برچسب‌دار
  مرحله ۲: آموزش نیمه‌نظارتی student با داده‌های برچسب‌دار و بدون برچسب
  ارزیابی و تولید خروجی‌های نهایی
```

## سخت‌افزار پیشنهادی

تنظیم مرجع مقاله از ۸ عدد NVIDIA V100 32GB، batch کلی 16 و 26,250 iteration استفاده می‌کند. اجرای inference روی یک GPU مناسب امکان‌پذیر است، اما آموزش کامل روی CPU یا لپ‌تاپ توصیه نمی‌شود.

- inference: یک GPU با حداقل 16 تا 24 گیگابایت VRAM
- full training: چند GPU یا یک GPU پرحافظه با batch کوچک‌تر
- سیستم‌عامل: Linux
- محیط سازگار: Python 3.10، PyTorch 1.13.1، CUDA 11.7

## خروجی‌های پروژه

- IoU و Dice برای segmentation
- Precision، Recall و F1-score برای identification
- نتایج جداگانه برای fully dentate و partially edentulous
- جدول مقایسه‌ی مدل‌ها
- prediction و overlayهای تصویری
- نمودارهای آموزش و validation
- گزارش اختلاف نتایج اجرا با مقادیر مرجع مقاله

مقادیر مرجع مقاله در `paper/reference/` نگهداری می‌شوند و با دستور زیر به جدول و نمودار تبدیل می‌شوند:

```bash
make reference
```

## ساختار پروژه

```text
salehi-semiTNet/
├── README.md
├── README.en.md
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── paper/reference/
├── scripts/
├── tests/
└── vendor/
```

## بررسی سیستم قبل از اجرا

```bash
make preflight
```

گزارش سخت‌افزار، CUDA، وضعیت دیتاست، checkpoint و فضای دیسک در `outputs/PREFLIGHT.json` ذخیره می‌شود.

## راهنمای کامل

راهنمای مرحله‌به‌مرحله‌ی فارسی در [docs/RUNBOOK.md](docs/RUNBOOK.md) قرار دارد.

جزئیات وابستگی‌های نرم‌افزاری و مجوزهای اجزای ثالث در `docs/THIRD_PARTY_NOTICES.md` نگهداری می‌شود.