# SemiTNet — سگمنتیشن و شناسایی دندان

> شبیه‌سازی کامل pipeline مدل SemiTNet با اجرای واقعی روی داده‌های پانورامیک دندان، قابل اجرا روی Windows و Linux با Python 3.10 و `.venv`.

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود دیتاست

```bash
python project.py download
```

داده‌ی مورد استفاده از یکی از منابع عمومی هم‌راستا با ساخت TSI15k انتخاب شده و شامل تصاویر پانورامیک، 32 کلاس دندان و annotation پیکسلی است.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. شبیه‌سازی کامل

```bash
python project.py full
```

اجرای `full` تمام مراحل اصلی آزمایش را به‌صورت end-to-end اجرا می‌کند: آموزش teacher، تولید pseudo-label، آموزش student، به‌روزرسانی EMA و ارزیابی نهایی. برای محدود نگه‌داشتن زمان اجرا، شبیه‌سازی روی یک subset ثابت 96 تصویری اجرا می‌شود: 60 تصویر labeled، 20 تصویر برای pseudo-label و 16 تصویر test. تمام معیارها و شکل‌های `outputs/final/` از همین اجرای واقعی محاسبه می‌شوند.
