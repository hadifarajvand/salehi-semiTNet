# راهنمای اجرای SemiTNet

> شبیه‌سازی کامل pipeline مدل SemiTNet با داده‌های واقعی — Windows و Linux — Python 3.10 و `.venv`

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود دیتاست

```bash
python project.py download
```

داده‌ی اجرا از یک منبع عمومی پانورامیک دندان هم‌راستا با منابع سازنده‌ی TSI15k تهیه می‌شود و شامل 32 کلاس دندان و annotation پیکسلی است.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. شبیه‌سازی کامل

```bash
python project.py full
```

این دستور تمام pipeline آزمایش را اجرا می‌کند: teacher training، pseudo-label generation، student training، EMA teacher update و evaluation نهایی. برای اجرای سریع، یک subset ثابت 96 تصویری استفاده می‌شود: 60 labeled، 20 pseudo-label و 16 test. تمام معیارها، predictionها و نمودارهای `outputs/final/` از همین اجرای واقعی تولید می‌شوند.
