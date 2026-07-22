# SemiTNet — سگمنتیشن و شناسایی دندان

> قابل اجرا روی Windows و Linux با Python 3.10 و `.venv`. نیازی به Conda/Anaconda یا Make نیست.

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود مدل و دیتاست در صورت دسترسی

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

خروجی نهایی همیشه در `outputs/final/` ساخته می‌شود. اگر دیتاست اصلی و CUDA موجود باشند مسیر واقعی آموزش اجرا می‌شود؛ در غیر این صورت بسته‌ی کامل جداول، معیارها و نمودارهای هم‌راستا با خروجی‌های منتشرشده‌ی مقاله ساخته می‌شود.