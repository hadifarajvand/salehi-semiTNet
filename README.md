# SemiTNet — سگمنتیشن و شناسایی دندان

این مخزن اکنون سه مسیر جدا دارد تا خروجی‌ها از نظر علمی قابل دفاع باشند:

1. **Reduced measured simulation** — اجرای سبک و قابل تکرار برای بررسی end-to-end پایپ‌لاین، provenance، شکل‌ها و بسته تحویل.
2. **Strict paper-equivalence reproduction** — مسیر سخت‌گیرانه برای exact TSI15k/test identities؛ تا زمان عبور از gateهای شواهد اجازه ادعای «بازتولید نتایج مقاله» نمی‌دهد.
3. **TED3 defensible reimplementation** — اجرای بزرگ‌مقیاس با دیتاست رسمی/عمومی TED3 برای checkpoint evaluation، supervised و semi-supervised experiments، بدون ادعای اینکه TED3 دقیقاً همان split اصلی TSI15k است مگر اینکه identity-equivalence اثبات شود.

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

قانون مهم: اگر exact TSI15k/TISI15k archive یا test identities قابل بازیابی نباشد، strict paper-equivalence باید **BLOCKED** گزارش شود. TED3 می‌تواند برای reimplementation بزرگ‌مقیاس استفاده شود، اما نباید بدون اثبات identity-equivalence به‌عنوان exact TSI15k معرفی شود.

جزئیات کامل:

```text
reproduction/DEFENSIBLE_REPRODUCTION.md
reproduction/reference_contract.json
```

## ۷. اجرای autonomous TED3 campaign پس از قرار دادن دیتاست‌ها

سه فایل زیر را در `reproduction/assets/dataset/incoming/` قرار دهید؛ agent همچنین می‌تواند آن‌ها را در سایر مسیرهای داخل working tree پیدا کند:

```text
TED3-train.tar
TED3-test.tar
TED3-unlabeled-data-15k-pseudo-mask.tar
```

این فایل‌ها و تمام داده‌های extractشده توسط `.gitignore` محافظت می‌شوند و نباید commit/push/package شوند.

اولین دستور اجباری:

```bash
python project.py ted3-preflight
```

این preflight:

- هر سه archive را پیدا می‌کند؛
- بررسی می‌کند Git آن‌ها را ignore کرده و tracked نیستند؛
- SHA256 و size را ثبت می‌کند؛
- tar را قبل از extraction از نظر unsafe path/link بررسی می‌کند؛
- manifest را در `outputs/ted3_reproduction/preflight/input_archives.json` می‌نویسد.

برای اجرای agent، دستور authoritative در root repository فایل زیر است:

```text
AGENTS.md
```

Agent باید طبق آن فایل تا پایان این مراحل ادامه دهد:

1. dataset forensic audit؛
2. publication-compatible method freeze؛
3. official checkpoint evaluation؛
4. supervised experiment؛
5. semi-supervised teacher/student experiment با unlabeled TED3؛
6. comparison/statistical analysis؛
7. تولید تمام figure/table/experiment artifacts موردنیاز بخش Evaluation/Experiments؛
8. ساخت `EVALUATION_COVERAGE_MATRIX.csv` تا هیچ artifact مقاله بدون وضعیت باقی نماند؛
9. ساخت final delivery، Persian runbook، reproducibility report و artifact manifest.

Agent نباید با اولین خطا متوقف شود. خطاهای قابل تعمیر باید root-cause شوند، patch شوند، مرحله مربوطه rerun شود و سپس pipeline ادامه پیدا کند. خروجی‌های measured فقط باید از اجرای واقعی تولید شوند؛ copy/fabrication نتایج مقاله ممنوع است.

خروجی اصلی campaign:

```text
outputs/ted3_reproduction/
```

Agent فقط زمانی اجازه status نهایی `COMPLETE — DEFENSIBLE TED3 REIMPLEMENTATION` دارد که completion checklist داخل `AGENTS.md` پاس شده باشد.

## ۸. Paper-style figures

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

این فاصله نباید با tuning روی `QuickSemiTransformer` پنهان شود. مسیر علمی جدید، اجرای TED3 campaign طبق `AGENTS.md` است و strict paper-equivalence همچنان به exact TSI15k identity evidence وابسته می‌ماند.
