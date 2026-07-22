# SemiTNet — سگمنتیشن و شناسایی دندان

> قابل اجرا روی Windows و Linux با Python 3.10 و `.venv`. نیازی به Conda/Anaconda یا Make نیست.

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود دیتاست

```bash
python project.py download
```

دیتاست سریع جایگزین شامل 598 تصویر پانورامیک، 32 کلاس دندان و annotation پیکسلی است و حجم دانلود آن حدود 464 MB است.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. اجرای واقعی سریع

```bash
python project.py full
```

اجرای `full` روی یک subset ثابت 96 تصویری از دیتاست واقعی انجام می‌شود: 60 تصویر labeled، 20 تصویر برای pseudo-label و 16 تصویر test. خروجی‌های اندازه‌گیری‌شده در `outputs/final/` ذخیره می‌شوند.
