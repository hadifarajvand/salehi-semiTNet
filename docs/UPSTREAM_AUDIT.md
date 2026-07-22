# Upstream repository audit

Article code URL redirects from `isbrycee/SemiTNet` to `isjinghao/SemiT-SAM`.

Pinned commit:

```text
bf66074bee9da9b37fd68454bcbac9140c4f59e2
```

## Blocking issues in the public branch

1. `train_net.py` overwrites command-line values with a private `/root/paddlejob/...` checkpoint.
2. It forces `SSL.PERCENTAGE=10` and evaluation target `Teacher`.
3. Labeled and unlabeled dataset registration points to private MICCAI/Paddle paths.
4. The current challenge config uses 52 classes while the TSI15k article uses 32.
5. The public launchers assume eight GPUs and contain local checkpoint paths.
6. The second-stage launcher sets burn-in equal to the full run length.
7. Evaluation launchers reference arbitrary local checkpoint filenames.
8. The public requirements file is incomplete for a clean installation.

## Wrapper policy

The original source is not silently rewritten by hand in this ZIP. `bootstrap_upstream.py` downloads the pinned source archive, and `patch_upstream.py` performs deterministic, reviewable changes and writes `vendor/SemiT-SAM/REPRODUCTION_PATCH_MANIFEST.json`.

## Metric implementation recovered from upstream

The upstream file `tools/compute_metric_fully_partial_box.py` contains the paper-oriented evaluation recipe. The wrapper ports it into `scripts/compute_article_metrics.py`, removes private paths, and writes JSON. It evaluates bounding boxes and masks through COCOeval at IoU 0.5, separates images with 32 teeth from images with fewer than 32 teeth, and derives the reported Dice/F1 values from precision and recall in the same style as the public author utility.
