# SemiTNet execution guide

Modes:

- `make smoke`: CPU-safe synthetic software validation.
- `make inference`: run the prepared test set with the released checkpoint.
- `make full`: teacher pretraining plus semi-supervised student training.

Main sequence:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

Full training:

```bash
make full
```

The project prepares the required model dependencies, configures TSI15k through `TSI15K_ROOT`, uses 32 tooth classes and the paper-scale optimization settings, and provides portable launch scripts for dataset preparation, inference, evaluation, and training.

The default `BURNIN_ITER=3000` can be overridden for experiments.

Third-party software and license notices are documented in `docs/THIRD_PARTY_NOTICES.md`.