# Paper vs current upstream discrepancy register

This file prevents current upstream code from being treated automatically as the exact 2024 publication configuration.

## Rule

For every conflict, preserve both sources, choose a declared experiment variant, and record the choice in `run_manifest.json`. Never silently resolve a conflict in favor of whichever file is easier to run.

| Item | Published paper / publication record | Current accessible upstream state | Reproduction decision |
|---|---|---|---|
| Repository identity | Paper cites `https://github.com/isbrycee/SemiTNet` (accessed 4 Aug 2024) | Repository now resolves/evolved as `isjinghao/SemiT-SAM` | Treat current `main` as post-publication evidence, not automatically the publication snapshot. |
| Dataset | TSI15k/TISI15k, 16,317 total; 1,589 labeled and 14,728 unlabeled; 1,398 labeled train + 191 test | Current upstream README references the released TSI15k dataset but the historical dataset asset may not be stably available at the old location | Exact asset/test identities are a hard gate. No substitute dataset. |
| Total batch size | Paper reports total batch size `16` | Current accessible MaskDINO config contains `IMS_PER_BATCH: 32` | First reproduction variant uses paper value 16. A separate upstream-current variant may use 32, but results must not be mixed. |
| Input size | 1024 | Current accessible config also uses `IMAGE_SIZE: 1024` | Match at 1024. |
| Optimizer | AdamW | Current accessible config uses `ADAMW` | Match. |
| Base LR | `1e-4` | Current accessible config uses `0.0001` | Match. |
| LR steps | 24,000 and 25,000 | Current accessible config uses `(24000, 25000)` | Match. |
| Iterations | 26,250 | Current accessible config uses `MAX_ITER: 26250`; second-stage script also uses `SSL.BURNIN_ITER 26250` | Exact stage semantics must be verified; matching numbers alone are not enough. |
| Model initialization | Transformer/MobileSAM-derived publication method | Current second-stage script initializes from `pretrained_models/mobile_sam.pkl`; released author SemiTNet checkpoint exists separately | Use exact declared initialization per stage and hash every checkpoint. |
| Number of classes | Publication task: 32 tooth identities | Current accessible config includes `NUM_CLASSES: 52 # 32`, reflecting broader/evolved use cases | Force a publication-specific 32-class config and verify class-ID mapping before G1. |
| Evaluation | Paper reports IoU, Dice, precision, recall, F1 on 191 cases | Current repository supports teacher/student evaluation but current config/code state is not itself proof of the paper evaluator | Validate evaluator by reproducing the released author checkpoint first (G1). |

## Required experiment variants

### Variant P — paper-faithful

Use the paper-stated settings where explicit, publication-compatible 32-class mapping, exact TSI15k identities, and a frozen evaluator that passes G1.

### Variant U — current-upstream comparison

Optionally run the current upstream configuration exactly as currently accessible. This is useful for diagnosing code drift but must be reported separately from Variant P.

## Stop condition

If a discrepancy materially affects model outputs and cannot be resolved from publication-era artifacts, mark it `UNRESOLVED` in the run manifest. Do not describe the resulting run as exact reproduction.
