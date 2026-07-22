# بازتولید مقاله SemiTNet برای سگمنتیشن و شناسایی دندان

**زبان:** **فارسی** | [English](README.en.md)

این مخزن برای بازتولید مقاله‌ی زیر آماده شده است:

> **A Semi-Supervised Transformer-Based Deep Learning Framework for Automated Tooth Segmentation and Identification on Panoramic Radiographs**  
> *Diagnostics*, 2024, 14(17), 1948 — DOI: `10.3390/diagnostics14171948`

هدف پروژه این است که روند مقاله از دانلود دیتاست تا اجرای مدل، ارزیابی، مقایسه با اعداد مقاله و تولید خروجی‌ها با فرمان‌های مشخص قابل انجام باشد. کد عمومی نویسندگان به یک پروژه‌ی جدیدتر با نام **SemiT-SAM** منتقل شده و در نسخه‌ی عمومی دارای چند مسیر خصوصی و تنظیم دستی است. اسکریپت‌های این مخزن نسخه‌ی مشخصی از کد upstream را دریافت می‌کنند و اصلاحات لازم را به‌صورت خودکار اعمال می‌کنند.

## اجرای سریع Smoke Test

Smoke Test بدون دانلود دیتاست اصلی و روی CPU اجرا می‌شود. این مرحله فقط سالم بودن مسیر آموزش teacher/student، محاسبه‌ی معیارها و تولید خروجی را بررسی می‌کند.

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

> اعداد Smoke Test نتیجه‌ی مقاله نیستند؛ این اعداد از یک دیتاست مصنوعی کوچک برای بررسی نرم‌افزار به‌دست می‌آیند.

## مسیر پیشنهادی برای اجرای مقاله

برای اجرای inference با checkpoint منتشرشده‌ی نویسندگان:

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

### این فرمان‌ها چه کاری انجام می‌دهند؟

```text
bootstrap
  دریافت نسخه‌ی ثابت کد نویسندگان
  اعمال patchهای بازتولیدپذیری

setup-full
  نصب محیط CUDA/Detectron2/Mask2Former

download
  دریافت دیتاست TSI15k و checkpoint رسمی

prepare
  بررسی ساختار فایل‌ها، COCO annotations و splitها

inference
  اجرای checkpoint رسمی روی test set
  تولید معیارها، predictionها و مقایسه با مقاله

full
  مرحله ۱: آموزش teacher با داده‌های برچسب‌دار
  مرحله ۲: آموزش نیمه‌نظارتی student با داده‌های برچسب‌دار و بدون برچسب
  ارزیابی و تولید گزارش نهایی
```

## سخت‌افزار موردنیاز

اجرای رسمی مقاله روی **۸ عدد NVIDIA V100 32GB**، با batch کلی 16 و 26,250 iteration گزارش شده است. اجرای inference با checkpoint آماده روی یک GPU مناسب ممکن است، اما آموزش کامل روی لپ‌تاپ یا CPU عملی نیست.

پیشنهاد اجرا:

- inference: یک GPU با حداقل 16 تا 24 گیگابایت VRAM
- full training: چند GPU یا یک GPU پرحافظه با batch کوچک‌تر و gradient accumulation
- سیستم‌عامل: Linux
- محیط مرجع upstream: Python 3.10، PyTorch 1.13.1، CUDA 11.7

## خروجی‌های مقاله که در این پروژه پوشش داده می‌شوند

- معیارهای segmentation: IoU و Dice
- معیارهای identification: Precision، Recall و F1-score
- نتیجه‌ی جداگانه برای fully dentate و partially edentulous
- جدول مقایسه با Mask R-CNN، MPFormer، Mask2Former، MaskDINO و GEM
- نمونه‌های prediction و overlay
- نمودارهای آموزش و validation
- گزارش اختلاف خروجی اجرا با مقادیر منتشرشده

مقادیر مرجع مقاله در `paper/reference/` ثبت شده‌اند و با اجرای زیر به نمودار و جدول تبدیل می‌شوند:

```bash
make reference
```

## ساختار مهم پروژه

```text
salehi-semiTNet/
├── README.md                     # راهنمای اصلی فارسی
├── README.en.md                  # راهنمای انگلیسی
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── paper/reference/              # اعداد منتشرشده در مقاله
├── scripts/
├── tests/
└── vendor/                       # کد upstream پس از bootstrap
```

## نکته درباره بازتولید دقیق

این پروژه از کد عمومی، دیتاست عمومی و checkpoint منتشرشده‌ی نویسندگان استفاده می‌کند. نزدیک‌ترین روش به خروجی مقاله، اجرای `make inference` با checkpoint رسمی و test split اصلی دیتاست است. آموزش دوباره ممکن است به دلیل seed، نسخه‌ی CUDA، ترتیب batchها و بعضی جزئیات منتشرنشده، اعداد کمی متفاوت تولید کند. گزارش پروژه این تفاوت را شفاف ثبت می‌کند و نتیجه‌ی Smoke را با نتیجه‌ی مقاله مخلوط نمی‌کند.

راهنمای مرحله‌به‌مرحله‌ی کامل در [docs/RUNBOOK.md](docs/RUNBOOK.md) قرار دارد.

## بررسی سیستم قبل از اجرا

پیش از دانلود یا آموزش می‌توانید وضعیت سیستم را بررسی کنید:

```bash
make preflight
```

گزارش سخت‌افزار، CUDA، وضعیت دیتاست، checkpoint و فضای دیسک در `outputs/PREFLIGHT.json` ذخیره می‌شود.

### نکته درباره وزن اولیه

checkpoint رسمی منتشرشده برای inference و نزدیک‌ترین بازتولید خروجی نهایی مقاله استفاده می‌شود. مسیر Full Training به‌صورت پیش‌فرض از checkpoint نهایی شروع نمی‌کند، چون چنین کاری نتیجه‌ی آموزش را آلوده می‌کند. وزن اولیه‌ی سازگار را می‌توان با گزینه‌ی `--init-weights` مشخص کرد.
