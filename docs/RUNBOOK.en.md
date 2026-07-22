# SemiTNet execution runbook

Modes:

- `make smoke`: CPU-safe synthetic software validation.
- `make inference`: released checkpoint on the prepared official test set.
- `make full`: teacher pretraining plus semi-supervised student training.

Core sequence:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

Full retraining:

```bash
make full
```

The wrapper pins upstream commit `bf66074bee9da9b37fd68454bcbac9140c4f59e2`, removes private path/debug overrides, registers TSI15k through `TSI15K_ROOT`, restores 32 classes and paper-scale optimization settings, and adds portable launch scripts.

The exact burn-in iteration is not stated in the paper. The default `3000` is inferred from a commented upstream debug configuration and is explicitly recorded as inferred.
