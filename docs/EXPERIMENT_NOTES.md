# Experiment notes

The project supports four output categories:

- `reference`: target values transcribed from the article for comparison;
- `smoke`: a small synthetic software-validation experiment;
- `inference`: model checkpoint evaluated on the prepared TSI15k test set;
- `full`: teacher/student weights trained through the project pipeline.

Exact numerical equality between separate full-training runs is not guaranteed because random initialization, data ordering, CUDA kernels, hardware, and environment versions can affect deep-learning results.

For the closest comparison with the published results, use the prepared TSI15k test set and the released checkpoint through:

```bash
make bootstrap
make setup-full
make download
make prepare
make inference
```

For a new training run:

```bash
make full
```

The project records run settings and outputs under `outputs/` so each experiment can be inspected independently.