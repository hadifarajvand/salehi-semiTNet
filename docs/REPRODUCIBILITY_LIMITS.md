# Reproducibility limits

The released checkpoint and public dataset make evaluation substantially more reproducible than retraining.

Exact bit-for-bit retraining is not guaranteed because:

- the article does not state the exact random seed;
- the exact labeled train/test file IDs must be recovered from the released archive;
- the exact burn-in transition iteration is not reported in the article;
- GPU kernels and distributed data order can differ;
- the public repository has evolved into SemiT-SAM and contains later challenge edits;
- original training logs are not released.

The package classifies outputs as:

- `reference`: values transcribed from the paper;
- `smoke`: actual small synthetic software test;
- `inference`: released checkpoint evaluated on the prepared dataset;
- `full`: weights trained by the user through this wrapper.
