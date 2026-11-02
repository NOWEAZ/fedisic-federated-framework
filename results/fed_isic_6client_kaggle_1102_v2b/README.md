# Fed-ISIC2019 6-client — Kaggle extended experiment B (results, 2026-11-02)

Results of Kaggle notebook version 4 ("expB combined", Save & Run All, GPU T4 x2, ran 21,456 s ≈ 5 h 58 min, no failed runs).
Code: this branch (`feature/v2-expB-notebooks`), notebook `kaggle/fed_isic2019_kaggle_v2_expB.ipynb`.
Two jobs ran at the same time, one per GPU (job queue in the notebook). Test set was opened once per run (`--final_test`, last-round model).

## What was run

| Part | Method | Setting | Seeds |
| :--- | :--- | :--- | :--- |
| E1 (mu recheck, validation only) | FedAvg, FedProx mu=0.03/0.1/0.3 (and mu=0.03/0.3 on seed 41) | official setting: 100 updates, 9 rounds | 40, 41 |
| Main | FedAvg, FedProx (selected mu), FedBN | `--keep_optimizer --lr_schedule cosine --num_updates 0 --local_epochs 2 --rounds 18` | 42, 100, 2026 |

Common settings: Adam lr 5e-4, batch size 64, val_fraction 0.1, split_seed 0, `--resume`. FedBN keeps BatchNorm per center and is evaluated per center.
Result file tag: `_keepopt_cosine_ep2`.

## mu selection (E1, `e1_mu_recheck/best_mu.json`)

Rule: mean over seeds 40 and 41 of the validation macro-F1 averaged over the last 3 rounds; ties go to the smaller mu. Seed 41 values for the other mu (0.001, 0.01, 1.0) come from the earlier results branch.

| mu | seed 40 | seed 41 | mean |
| :--- | :---: | :---: | :---: |
| 0.03 | 0.4938 | 0.5084 | 0.5011 |
| 0.1 | 0.5206 | 0.5299 | 0.5252 |
| **0.3** | 0.5244 | 0.5307 | **0.5275** |
| (FedAvg) | 0.4859 | 0.5018 | |

Selected mu = 0.3 (previously 0.1). The difference between 0.3 and 0.1 is 0.0023, which is small.

## Final test results (mean of 3 seeds, 95% bootstrap CI over test samples)

| Method | Bal Acc | Macro F1 |
| :--- | :---: | :---: |
| Centralized (9-round era baseline) | 0.7391 (0.7205–0.7579) | 0.6824 (0.6641–0.6986) |
| FedAvg, keepopt+cosine, 2 epochs, 18 rounds | 0.7300 (0.7107–0.7491) | 0.7100 (0.6900–0.7278) |
| FedProx mu=0.3, same setting | 0.7225 (0.7025–0.7418) | 0.6902 (0.6695–0.7086) |
| FedBN, same setting | 0.7674 (0.7486–0.7860) | 0.7489 (0.7302–0.7654) |
| (earlier) FedAvg 18 rounds, 100 updates | 0.6778 | 0.5689 |
| (earlier) FedProx mu=0.1 18 rounds, 100 updates | 0.7027 | 0.5889 |
| (earlier) FedBN 9 rounds | 0.6434 | 0.5196 |

Per seed (Bal Acc): FedAvg 0.7265 / 0.7401 / 0.7234, FedProx 0.7098 / 0.7295 / 0.7281, FedBN 0.7592 / 0.7730 / 0.7700 (seeds 42 / 100 / 2026).

## Paired differences, same seeds (`analysis_extended/paired_summary.csv`, Bal Acc)

| Comparison | Mean diff | Seeds positive | Bootstrap CI excludes 0 |
| :--- | :---: | :---: | :---: |
| FedAvg new setting − FedAvg 18 rounds (100 updates) | +0.0523 | 3/3 | yes |
| FedProx mu=0.3 new − FedProx mu=0.1 18 rounds (100 updates) | +0.0198 | 3/3 | yes |
| FedProx − FedAvg (new setting) | −0.0076 | 1/3 | no (macro-F1: −0.0198, 0/3, yes) |
| FedBN − FedAvg (new setting) | +0.0374 | 3/3 | yes |
| FedBN − FedProx (new setting) | +0.0449 | 3/3 | yes |

## Per center (Bal Acc, mean of 3 seeds; test samples 2483 / 791 / 672 / 452 / 164 / 88)

| Center | FedAvg | FedProx | FedBN |
| :--- | :---: | :---: | :---: |
| 0 | 0.804 | 0.789 | 0.812 |
| 1 | 0.686 | 0.712 | 0.825 |
| 2 | 0.649 | 0.650 | 0.653 |
| 3 | 0.479 | 0.485 | 0.599 |
| 4 | 0.422 | 0.447 | 0.603 |
| 5 | 0.753 | 0.789 | 0.760 |

FedBN − FedAvg: center 4 +0.181 (CI 0.124–0.239), center 5 +0.007 (−0.132–0.144). FedBN − FedProx: center 4 +0.156 (0.098–0.216), center 5 −0.029 (−0.159–0.097).

Melanoma (MEL) recall, mean of 3 seeds (`analysis_extended/per_class_recall.csv`): Centralized 0.742, FedBN 0.728, FedAvg 0.651, FedProx 0.608.

## Caveats

- Several factors changed together (keep_optimizer, cosine schedule, epoch-based local training, 18 rounds), so this experiment cannot say how much each one contributed.
- FedBN is evaluated per center with that center's own BatchNorm state, while FedAvg/FedProx are evaluated as a single server model. The evaluated objects differ.
- Only 3 seeds. The bootstrap intervals reflect test-sample variation only; seed-to-seed variation is in `analysis_extended/paired_by_seed.csv`.
- Center 5 has 88 test samples (center 4: 164), so per-center differences there are noisy.
- The selected mu=0.3 beat 0.1 by a small margin (0.0023 in validation macro-F1).

## Files

- `<method>_keepopt_cosine_ep2_seed<seed>_final_test.json / _final_test_preds.npz / _metrics.csv / _env.json`: main runs (3 methods x 3 seeds; FedProx files carry `mu0.3` in the name).
- `e1_mu_recheck/`: E1 metrics CSVs, env files and `best_mu.json`.
- `analysis_extended/`: output of `analysis/analyze_extended.py` (all pairs, same numbers as the notebook's own analysis plus FedBN − FedProx).
- `kaggle_version4_full_log.txt`: full Kaggle log of version 4.
- Checkpoints (`.pt`) and the Kaggle output archives are not included.
