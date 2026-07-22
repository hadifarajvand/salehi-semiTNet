# SemiTNet — سگمنتیشن و شناسایی دندان

این مخزن اکنون دو مسیر کاملاً جدا دارد تا خروجی‌ها از نظر علمی قابل دفاع باشند:

1. **Reduced measured simulation** — اجرای سبک و قابل تکرار برای بررسی end-to-end پایپ‌لاین، provenance، شکل‌ها و بسته تحویل.
2. **Paper-equivalence reproduction** — مسیر سخت‌گیرانه برای نزدیک‌شدن به نتایج مقاله که تا زمان عبور از gateهای شواهد، اجازه ادعای «بازتولید نتایج مقاله» نمی‌دهد.

> وضعیت فعلی: reduced baseline معتبر است، اما هنوز paper-equivalent نیست.

## ۱. نصب محیط reduced simulation

```bash
python project.py install
```

## ۲. دانلود دیتاست reduced simulation

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

**این دیتاست و split با TSI15k مقاله یکسان نیستند و نباید برای ادعای numerical reproduction استفاده شوند.**

## ۳. Smoke Test

```bash
python project.py smoke
```

## ۴. اجرای reduced measured simulation

```bash
python project.py full
```

این دستور teacher training، pseudo-label generation، student training، EMA update، collapse guard، ارزیابی test، اعتبارسنجی خروجی و ساخت بسته تحویل را اجرا می‌کند.

پیکسل‌های pseudo-label با confidence پایین به‌جای تبدیل‌شدن به background نادیده گرفته می‌شوند. اگر تمام معیارهای نهایی صفر باشند، اجرا PASS اعلام نمی‌شود و بسته نهایی ساخته نمی‌شود.

خروجی reduced baseline در:

```text
outputs/final/
SemiTNet-client-deliverable.zip
```

ادعای مجاز برای این خروجی:

> Measured reduced-scale re-simulation; not numerically equivalent to the published full-scale experiment.

## ۵. ممیزی reproducibility

```bash
python project.py audit
```

خروجی:

```text
outputs/reproducibility_audit.json
outputs/REPRODUCIBILITY_AUDIT.md
```

Gateها:

- **G0:** سلامت و consistency خروجی reduced baseline
- **G1:** اجرای checkpoint رسمی نویسندگان روی exact 191-image paper test set و تطبیق evaluator
- **G2:** تطبیق روش، معماری، preprocessing، class mapping، split و training protocol با مقاله
- **G3:** اجرای full paper-faithful training و ارزیابی با evaluator تأییدشده

تا وقتی G1 عبور نکرده است، عبارت‌هایی مثل «same exact paper results» یا «paper-equivalent metrics» مجاز نیستند.

## ۶. دریافت assetهای رسمی مقاله

Checkpoint رسمی منتشرشده توسط نویسندگان با revision و SHA256 ثابت pin شده است:

```bash
python project.py paper-assets-checkpoint
```

برای تلاش جهت دریافت exact datasetی که خود مقاله معرفی کرده است:

```bash
python project.py paper-assets-dataset
```

قانون مهم: اگر exact TSI15k/TISI15k archive یا test identities قابل بازیابی نباشد، فرایند باید **BLOCKED** گزارش شود. جایگزین‌کردن دیتاست 598 تصویری یا هر دیتاست دیگری برای paper-equivalence ممنوع است.

جزئیات کامل:

```text
reproduction/DEFENSIBLE_REPRODUCTION.md
reproduction/reference_contract.json
```

## ۷. Paper-style figures

```bash
python scripts/export_paper_style_figures.py
```

figureهای paper-style در `outputs/paper_style/figures/` ساخته می‌شوند. Published-reference panels و measured reduced-simulation panels باید از هم جدا و صریحاً برچسب‌گذاری شوند. هیچ مقدار یا prediction گمشده‌ای نباید ساخته یا حدس زده شود.

## وضعیت فعلی قابل دفاع

Canonical reduced metrics:

- IoU: `33.05651614205745`
- Dice: `49.68793276800474`
- Precision: `12.52662423999735`
- Recall: `12.02895676318772`
- F1: `12.272746893320026`

Published SemiTNet reference metrics:

- IoU: `94.41`
- Dice: `95.45`
- Precision: `94.74`
- Recall: `97.10`
- F1: `95.90`

این فاصله نباید با tuning روی `QuickSemiTransformer` پنهان شود. قدم بعدی علمی، **G1: official-checkpoint equivalence** است، نه افزایش epoch روی reduced surrogate.
