# SemiTNet — سگمنتیشن و شناسایی دندان

> شبیه‌سازی کامل pipeline مدل SemiTNet با اجرای واقعی روی داده‌های پانورامیک دندان، قابل اجرا روی Windows و Linux با Python 3.10 و `.venv`.

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود و آماده‌سازی دیتاست

```bash
python project.py download
```

این دستور دقیقاً دیتاست `Teeth Segmentation on Dental X-ray Images` را از Dataset Ninja/ Humans in the Loop دریافت می‌کند و قبل از استفاده بررسی می‌کند که دیتاست شامل 598 تصویر و کلاس‌های دندان 1 تا 32 باشد.

پس از دانلود:

- نسخه خام در `data/raw/quick_teeth/` نگهداری می‌شود.
- اطلاعات اعتبارسنجی در `data/raw/quick_teeth/download_manifest.json` ذخیره می‌شود.
- split ثابت شبیه‌سازی به‌صورت خودکار در `data/processed/quick_teeth/` ساخته می‌شود.
- مشخصات دقیق split در `data/processed/quick_teeth/split_manifest.json` ذخیره می‌شود.
- split اجرا: 60 تصویر labeled، 20 تصویر label-hidden برای pseudo-label و 16 تصویر test.
- annotation تصاویر pseudo-label وارد ورودی آموزش نمی‌شود تا leakage رخ ندهد.

اگر فایل‌های موجود ناقص یا مربوط به دیتاست دیگری باشند، دستور download متوقف می‌شود و اجازه اجرای simulation را نمی‌دهد.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. شبیه‌سازی کامل

```bash
python project.py full
```

اجرای `full` فقط از دیتاست آماده‌شده و manifest تأییدشده استفاده می‌کند و مراحل teacher training، pseudo-label generation، student training، EMA update و held-out evaluation را اجرا می‌کند. اگر دیتاست آماده نشده باشد، `full` ابتدا فرآیند download/setup را خودکار اجرا می‌کند. تمام معیارها و شکل‌های نهایی در `outputs/final/` ذخیره می‌شوند.
