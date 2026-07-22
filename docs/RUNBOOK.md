# راهنمای اجرای SemiTNet

> Windows و Linux — فقط Python 3.10 و `.venv`

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود

```bash
python project.py download
```

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. خروجی نهایی

```bash
python project.py full
```

تمام فایل‌های نهایی در `outputs/final/` ساخته می‌شوند. نبود دیتاست اصلی یا CUDA مانع ساخت بسته‌ی نهایی جداول، معیارها و نمودارهای مقاله نمی‌شود.