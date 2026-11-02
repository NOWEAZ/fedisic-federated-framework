# Fed-ISIC2019 6-client — Kaggle extended experiment A (results, 2026-11-02)

Results of Kaggle notebook version 3 ("expA 18 rounds + FedBN", Save & Run All, GPU T4 x2, ran 29,136 s ≈ 8 h 5 min).
Code: this branch (`feature/v2-extended-experiments`), notebook `fed_isic2019_kaggle_v2_expA.ipynb`.
The earlier 9-round results (Centralized, FedAvg, FedProx mu=0.1; seeds 42/100/2026) are read from branch `results/fed-isic-6client-1101`.

## What was run

| Experiment | Method | Rounds | Seeds | Notes |
| :--- | :--- | :---: | :--- | :--- |
| E4 (more training) | FedAvg | 18 (official: 9) | 42, 100, 2026 | `--tag r18`, `--final_test` |
| E4 (more training) | FedProx mu=0.1 | 18 (official: 9) | 42, 100, 2026 | `--tag r18`, `--final_test` |
| E3 | FedBN (BatchNorm kept per center) | 9 | 42, 100, 2026 | `--final_test` |

All other settings are the same as the baseline (lr 5e-4 constant, Adam, batch size 64, 100 local updates per round, val_fraction 0.1, split_seed 0).
The test set was evaluated once per run (`--final_test`, last-round model).

## Final test results (per seed)

Bal Acc = balanced accuracy over all test samples. Gap / Std = max-min and standard deviation of balanced accuracy across the 6 centers.

| Method | Seed | Bal Acc | Macro F1 | ECE | Center Gap | Center Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FedAvg, 18 rounds | 42 | 0.6752 | 0.5386 | 0.0731 | 0.2953 | 0.1109 |
| FedAvg, 18 rounds | 100 | 0.6803 | 0.5766 | 0.0796 | 0.4245 | 0.1490 |
| FedAvg, 18 rounds | 2026 | 0.6777 | 0.5915 | 0.0570 | 0.3790 | 0.1518 |
| FedProx mu=0.1, 18 rounds | 42 | 0.7030 | 0.5988 | 0.0762 | 0.3190 | 0.1282 |
| FedProx mu=0.1, 18 rounds | 100 | 0.7122 | 0.5875 | 0.0599 | 0.3958 | 0.1492 |
| FedProx mu=0.1, 18 rounds | 2026 | 0.6929 | 0.5804 | 0.0609 | 0.3548 | 0.1290 |
| FedBN, 9 rounds | 42 | 0.6742 | 0.5560 | 0.0885 | 0.2502 | 0.1039 |
| FedBN, 9 rounds | 100 | 0.6455 | 0.5256 | 0.0628 | 0.1687 | 0.0753 |
| FedBN, 9 rounds | 2026 | 0.6105 | 0.4772 | 0.0515 | 0.3297 | 0.1214 |

## Mean over the 3 seeds with 95% bootstrap CI (`analysis_extended/bootstrap_ci_overall.csv`)

| Method | Bal Acc (95% CI) | Macro F1 (95% CI) |
| :--- | :---: | :---: |
| Centralized (9-round era baseline) | 0.7391 (0.7205–0.7579) | 0.6824 (0.6641–0.6986) |
| FedAvg, 9 rounds | 0.6331 (0.6165–0.6482) | 0.5078 (0.4886–0.5259) |
| FedAvg, 18 rounds | 0.6778 (0.6616–0.6931) | 0.5689 (0.5496–0.5866) |
| FedProx mu=0.1, 9 rounds | 0.6695 (0.6522–0.6867) | 0.5303 (0.5115–0.5491) |
| FedProx mu=0.1, 18 rounds | 0.7027 (0.6853–0.7195) | 0.5889 (0.5690–0.6074) |
| FedBN, 9 rounds | 0.6434 (0.6267–0.6609) | 0.5196 (0.5014–0.5374) |

The intervals reflect test-sample variation only (resampling within centers); variation across seeds is shown separately in `analysis_extended/paired_by_seed.csv`.

## Paired differences, same seeds (`analysis_extended/paired_summary.csv`, Bal Acc)

| Comparison | Mean diff | Seeds with positive diff | Bootstrap CI excludes 0 |
| :--- | :---: | :---: | :---: |
| FedAvg 18r − FedAvg 9r | +0.0447 | 3/3 | yes |
| FedProx 18r − FedProx 9r | +0.0332 | 3/3 | yes |
| FedProx 18r − FedAvg 18r | +0.0249 | 3/3 | yes |
| FedBN − FedAvg (9r) | +0.0103 | 2/3 | no |
| FedBN − FedProx mu=0.1 (9r) | −0.0261 | 1/3 | yes (below 0) |

## Caveats

- FedBN is evaluated per center with that center's own BatchNorm state ("global weights + center BN"), while FedAvg/FedProx are evaluated as a single server model. The two are not the same kind of model, so compare them with that in mind.
- The 18-round runs use twice the training amount of the official 9-round setting, so they are not directly comparable to Centralized (20 epochs) at the official budget.
- Only 3 seeds per method. The effects of more rounds (all 3 seeds positive) are consistent; the FedBN comparisons are less clear (FedBN seed 2026 is much lower than the other two seeds).
- Not in this folder: model checkpoints (`*.pt`), per-run environment JSONs, and the raw tar (`results_v2.tar.gz`, 285.6 MB, kept on Kaggle version 3 Output).

## Files

- `<method>_seed<N>_final_test.json` — final test metrics and config of each run
- `<method>_seed<N>_final_test_preds.npz` — test predictions used by `analysis/analyze_extended.py`
- `<method>_seed<N>_metrics.csv` — per-round training/validation metrics
- `analysis_extended/*.csv` — output of `analysis/analyze_extended.py` (bootstrap CIs, paired differences, confusion matrices, per-class recall)
