# Fed-ISIC2019 6-client 실험 결과 (Kaggle, 2026-11-01)

FLamby Fed-ISIC2019(센터 6개, 클래스 8개)에서 **중앙집중(centralized) / FedAvg / FedProx(μ=0.1)** 를 비교한 결과입니다.
모든 수치는 이 폴더의 원본 파일(`*_final_test.json`, `*_metrics.csv`, `stage1_summary.csv`)에서 `analyze_results.py`로 계산했습니다.

## 실행 환경
- Kaggle Notebook (Commit **Version #1**, Save & Run All), GPU **Tesla T4**
- 프로젝트 코드 커밋: `e39f841` (브랜치 `fix/notebook-source-resume-state`)
- FLamby 커밋: `edacf54`
- 노트북: [`kaggle/fed_isic2019_kaggle_run1101.ipynb`](../../kaggle/fed_isic2019_kaggle_run1101.ipynb)
- Python 3.13.15, torch 2.11.0+cu128 (각 실행의 `*_env.json` 참고)

## 실험 설정
- 공식 스케줄: 연합학습(FedAvg/FedProx) **9라운드 × 100스텝**(`num_updates=100`, local_epochs=1, batch 64, lr 5e-4), 중앙집중 **20 에폭**
- 검증 분할: val_fraction=0.1, split_seed=0
- 시드: μ 탐색(Stage 1)은 **41**, 본실험(Stage 2)은 **42 / 100 / 2026**
- **BEST_MU = 0.1**: 시드 41의 Stage 1에서 μ ∈ {0.001, 0.01, 0.1, 1}와 FedAvg(μ=0)를 돌려, **val macro-F1의 마지막 3라운드 평균이 가장 큰 μ**를 선택(동률이면 작은 μ). (`best_mu.json`)
- **test는 `--final_test`로 단 1회만 평가**했으며, 사용한 모델은 **마지막 라운드 모델**(best-val 체크포인트 아님)입니다. 표의 모든 test 지표는 이 결과입니다.
- 통신량 `cumulative_comm_mb`는 마지막 라운드(9)의 누적값이며 중앙집중은 0입니다.
- 지표 `*_std`, `*_gap`은 센터 간(6개) 표준편차/최대-최소 차이이고, 표의 `±`는 시드 3개의 표본 표준편차(ddof=1)입니다.

## 결과 표

### 1. 방법 × 시드별 test (`summary_test_by_seed.csv`)
| method | seed | final_round | balanced_acc | macro_f1 | macro_f1_fixed8 | ece | macro_f1_std | macro_f1_gap | bal_acc_std | bal_acc_gap | cumulative_comm_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| centralized | 42 | 20 | 0.7442 | 0.6775 | 0.6775 | 0.0464 | 0.0519 | 0.1547 | 0.0924 | 0.2406 | 0.0000 |
| centralized | 100 | 20 | 0.7318 | 0.6552 | 0.6552 | 0.0568 | 0.0539 | 0.1355 | 0.0629 | 0.1534 | 0.0000 |
| centralized | 2026 | 20 | 0.7412 | 0.7145 | 0.7145 | 0.0476 | 0.0531 | 0.1592 | 0.0345 | 0.0936 | 0.0000 |
| fedavg | 42 | 9 | 0.6478 | 0.5112 | 0.5112 | 0.0834 | 0.0370 | 0.0867 | 0.1205 | 0.3033 | 1672.6300 |
| fedavg | 100 | 9 | 0.6190 | 0.4881 | 0.4881 | 0.0996 | 0.0463 | 0.1001 | 0.1368 | 0.3409 | 1672.6300 |
| fedavg | 2026 | 9 | 0.6325 | 0.5243 | 0.5243 | 0.0902 | 0.0674 | 0.1898 | 0.1319 | 0.3197 | 1672.6300 |
| fedprox | 42 | 9 | 0.6529 | 0.4944 | 0.4944 | 0.0758 | 0.0540 | 0.1587 | 0.1097 | 0.3217 | 1672.6300 |
| fedprox | 100 | 9 | 0.6769 | 0.5369 | 0.5369 | 0.0900 | 0.0644 | 0.1518 | 0.1478 | 0.3844 | 1672.6300 |
| fedprox | 2026 | 9 | 0.6786 | 0.5598 | 0.5598 | 0.0774 | 0.0724 | 0.2251 | 0.1577 | 0.4571 | 1672.6300 |

### 2. 방법별 시드 평균 ± 표본표준편차 (`summary_test_mean_std.csv`)
| method | balanced_acc | macro_f1 | macro_f1_fixed8 | ece | macro_f1_std | macro_f1_gap | bal_acc_std | bal_acc_gap | cumulative_comm_mb |
|---|---|---|---|---|---|---|---|---|---|
| centralized | 0.7391 ± 0.0064 | 0.6824 ± 0.0299 | 0.6824 ± 0.0299 | 0.0503 ± 0.0057 | 0.0530 ± 0.0010 | 0.1498 ± 0.0126 | 0.0633 ± 0.0290 | 0.1625 ± 0.0739 | 0.00 |
| fedavg | 0.6331 ± 0.0144 | 0.5078 ± 0.0183 | 0.5078 ± 0.0183 | 0.0910 ± 0.0081 | 0.0502 ± 0.0156 | 0.1256 ± 0.0561 | 0.1297 ± 0.0083 | 0.3213 ± 0.0188 | 1672.63 |
| fedprox | 0.6695 ± 0.0143 | 0.5303 ± 0.0332 | 0.5303 ± 0.0332 | 0.0811 ± 0.0078 | 0.0636 ± 0.0092 | 0.1785 ± 0.0405 | 0.1384 ± 0.0254 | 0.3877 ± 0.0678 | 1672.63 |

(CSV 원본은 평균/표준편차를 `*_mean`, `*_std` 컬럼으로 분리해 저장)

### 3. 방법 × 센터별 test, 시드 평균 (`summary_test_per_client.csv`)
| method | center | balanced_acc | macro_f1 | n_test |
|---|---|---|---|---|
| centralized | 0 | 0.7328 | 0.6888 | 2483 |
| centralized | 1 | 0.7079 | 0.6482 | 791 |
| centralized | 2 | 0.7621 | 0.6530 | 672 |
| centralized | 3 | 0.6232 | 0.6203 | 452 |
| centralized | 4 | 0.6469 | 0.6440 | 164 |
| centralized | 5 | 0.7309 | 0.5870 | 88 |
| fedavg | 0 | 0.6566 | 0.5119 | 2483 |
| fedavg | 1 | 0.7601 | 0.5348 | 791 |
| fedavg | 2 | 0.6788 | 0.5583 | 672 |
| fedavg | 3 | 0.5225 | 0.4818 | 452 |
| fedavg | 4 | 0.4719 | 0.4539 | 164 |
| fedavg | 5 | 0.7794 | 0.5290 | 88 |
| fedprox | 0 | 0.7018 | 0.5533 | 2483 |
| fedprox | 1 | 0.7041 | 0.4699 | 791 |
| fedprox | 2 | 0.7381 | 0.5704 | 672 |
| fedprox | 3 | 0.5741 | 0.5202 | 452 |
| fedprox | 4 | 0.4522 | 0.4669 | 164 |
| fedprox | 5 | 0.8400 | 0.6073 | 88 |

### 참고: Stage 1 μ 탐색 (시드 41, val; `stage1_summary.csv`)
| method | mu | rounds | val_macro_f1_last3 | val_bal_acc_last3 | val_f1_gap_last3 |
|---|---|---|---|---|---|
| fedavg | 0.0 | 9 | 0.5018 | 0.6240 | 0.1142 |
| fedprox | 0.001 | 9 | 0.4875 | 0.6203 | 0.1720 |
| fedprox | 0.01 | 9 | 0.4995 | 0.6280 | 0.1387 |
| fedprox | 0.1 | 9 | 0.5299 | 0.6541 | 0.1716 |
| fedprox | 1.0 | 9 | 0.5094 | 0.6308 | 0.2114 |

## 한계
- 센터 4·5는 test가 각각 164·88장뿐이라 **센터별 지표는 잡음이 큽니다**센터별 순위 해석은 주의하세요.
- 시드가 **3개**뿐이라 표준편차 추정이 불안정하고, 방법 간 차이의 유의성은 검증하지 않았습니다.
- μ는 **시드 41 한 번**으로 선정했습니다. 선정 시드와 본실험 시드가 달라 선택이 시드 운에 좌우될 수 있습니다.
- 중앙집중은 20에폭, 연합학습은 9라운드×100스텝으로 **학습량이 다르므로** 단순 비교에 한계가 있습니다.
- **FLamby 논문 수치와 직접 비교하지 마세요.** 검증 분할(val 10%), 학습 스케줄, 지표 정의, 평가 방식이 다를 수 있습니다.

## 파일 구성
- `*_metrics.csv`: 라운드별 val 지표 / `*_final_test.json`: 마지막 라운드(및 best-val 참고용) test 지표
- `*_final_test_preds.npz`: test 예측(y_true, y_pred, y_prob, client_idx) / `*_env.json`: 환경 기록
- `best_mu.json`, `stage1_summary.csv`: μ 탐색 / `summary_test_*.csv`: 위 표 / `analyze_results.py`: 표 생성 스크립트
- 제외: 모델 체크포인트(`*.pt`), 원본 출력 압축본(`*.tar.gz`), ISIC 이미지

## 재현
```bash
python analyze_results.py   # 이 폴더에서 실행하면 summary_test_*.csv 3개를 다시 생성
```
