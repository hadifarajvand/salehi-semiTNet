# راهنمای اجرای SemiTNet

> Windows و Linux — فقط Python 3.10 و `.venv`

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود دیتاست واقعی سریع

```bash
python project.py download
```

دیتاست: 598 تصویر پانورامیک، 32 کلاس دندان، annotation پیکسلی، حدود 464 MB.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. اجرای واقعی سریع

```bash
python project.py full
```

`full` از 96 تصویر واقعی استفاده می‌کند: 60 labeled، 20 pseudo-label و 16 test. تمام معیارها و شکل‌های `outputs/final/` از همین اجرای واقعی محاسبه می‌شوند.
