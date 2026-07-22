# راهنمای اجرای SemiTNet

> Windows و Linux — Python 3.10 و `.venv`

## ۱. نصب

```bash
python project.py install
```

## ۲. دانلود و آماده‌سازی دیتاست

```bash
python project.py download
```

این دستور دیتاست دقیق `Teeth Segmentation on Dental X-ray Images` را دریافت می‌کند، سپس موارد زیر را بررسی می‌کند:

- تعداد کل تصاویر: 598
- کلاس‌ها: 1 تا 32
- ساختار annotation پیکسلی Supervisely

ساختار خروجی:

```text
data/raw/quick_teeth/
    download_manifest.json

data/processed/quick_teeth/
    split_manifest.json
    train_labeled/
    pseudo_unlabeled/
    test/
```

split ثابت اجرای simulation:

- 60 تصویر labeled برای teacher/student supervised training
- 20 تصویر بدون annotation در ورودی برای pseudo-label generation
- 16 تصویر held-out test با annotation

اگر دیتاست ناقص یا اشتباه باشد، فرآیند متوقف می‌شود.

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. اجرای simulation

```bash
python project.py full
```

`full` فقط از `data/processed/quick_teeth/split_manifest.json` استفاده می‌کند. اگر manifest موجود نباشد، ابتدا download و preparation به‌صورت خودکار انجام می‌شود.

خروجی‌ها:

```text
outputs/final/
    metrics.json
    history.csv
    run_manifest.json
    RESULTS.md
    figures/
        metrics.png
        predictions.png
        training_curves.png
```

## Paper-style figures

```bash
python scripts/export_paper_style_figures.py
```

این خروجی در `outputs/paper_style/figures/` figureهای paper-style را می‌سازد و فقط از `outputs/final/` و referenceهای `paper/reference/` استفاده می‌کند. هیچ training یا inference جدیدی اجرا نمی‌شود.
