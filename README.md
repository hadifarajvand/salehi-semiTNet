# SemiTNet — سگمنتیشن و شناسایی دندان

> شبیه‌سازی end-to-end پایپ‌لاین SemiTNet روی داده‌های واقعی پانورامیک دندان، قابل اجرا با Python 3.10 و محیط محلی `.venv`.

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود و آماده‌سازی دیتاست

```bash
python project.py download
```

این دستور دیتاست `Teeth Segmentation on Dental X-ray Images` را دریافت می‌کند و فقط در صورتی ادامه می‌دهد که دقیقاً 598 تصویر و کلاس‌های دندان 1 تا 32 تأیید شوند.

پس از دانلود:
- داده خام: `data/raw/quick_teeth/`
- manifest دانلود: `data/raw/quick_teeth/download_manifest.json`
- داده آماده: `data/processed/quick_teeth/`
- manifest تقسیم‌بندی: `data/processed/quick_teeth/split_manifest.json`
- split ثابت: 60 labeled + 20 pseudo-label بدون annotation + 16 held-out test
- annotation تصاویر pseudo-label وارد training input نمی‌شود تا leakage رخ ندهد.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. اجرای کامل شبیه‌سازی

```bash
python project.py full
```

اجرای `full` مراحل teacher training، pseudo-label generation، student training، EMA update، محافظت در برابر collapse، ارزیابی test، اعتبارسنجی سخت‌گیرانه خروجی و ساخت بسته تحویل را اجرا می‌کند.

پیکسل‌های pseudo-label با confidence پایین به‌جای تبدیل‌شدن به background نادیده گرفته می‌شوند. health check مانع جایگزینی teacher سالم با student collapse‌شده می‌شود. اگر تمام معیارهای نهایی صفر باشند، اجرا PASS اعلام نمی‌شود و بسته نهایی ساخته نمی‌شود.

خروجی معتبر در `outputs/final/` و ZIP تحویل در مسیر زیر ساخته می‌شود:

```text
SemiTNet-client-deliverable.zip
```
