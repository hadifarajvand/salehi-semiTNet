# خروجی‌های تولیدشده

این پوشه برای نگهداری خروجی‌های Smoke Test، خروجی‌های مرجع مقاله و نتیجه‌ی اجرای واقعی روی TSI15k استفاده می‌شود.

## Smoke Test

برای بازتولید خروجی‌های Smoke اجرا کنید:

```bash
make smoke
```

خروجی‌های اصلی:

```text
outputs/smoke/
├── metrics.json
├── history.csv
├── run_manifest.json
└── figures/
    ├── architecture.png
    ├── predictions.png
    ├── teacher_student_workflow.png
    └── training_curves.png
```

اعداد `metrics.json` نتیجه‌ی Smoke Test مصنوعی هستند و نباید به‌عنوان نتیجه‌ی مقاله گزارش شوند.

## نمودارهای مرجع مقاله

برای تولید دوباره‌ی نمودارهای مقایسه‌ای از اعداد منتشرشده در مقاله:

```bash
make reference
```

این فرمان نمودارهای Dice، IoU، Precision، Recall و F1 را در `outputs/reference/` ایجاد می‌کند.

## GitHub Actions

Workflow با نام `smoke-and-reference` در هر push یا اجرای دستی موارد زیر را انجام می‌دهد:

1. اجرای unit test؛
2. اجرای Smoke Test؛
3. تولید دوباره‌ی تمام شکل‌های Smoke؛
4. تولید دوباره‌ی نمودارهای مرجع مقاله؛
5. ذخیره‌ی همه‌ی خروجی‌ها در artifact با نام:

```text
semitnet-generated-outputs
```

بنابراین لازم نیست تمام فایل‌های PNG تولیدی به‌صورت تکراری داخل Git نگهداری شوند؛ خروجی کامل با اجرای `make smoke` و `make reference` یا از artifact مربوط به GitHub Actions قابل دریافت است.

## اجرای واقعی مقاله

پس از دانلود TSI15k و checkpoint رسمی:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

نتایج واقعی در مسیر زیر ذخیره می‌شوند:

```text
outputs/inference/
```

برای آموزش کامل teacher/student از ابتدا:

```bash
make full
```
