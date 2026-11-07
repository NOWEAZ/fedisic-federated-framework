# 🏥 Fed-ISIC2019 의료 영상 연합학습 프로젝트 개요 (Project Overview)

## 1. 프로젝트 목적
본 프로젝트는 다기관 피부병변 분류 벤치마크인 **FLamby Fed-ISIC2019 (6-Client, 8-Class)** 공식 환경에서 발생하는 **병원 간 비독립·비동등 분포(Non-IID: Quantity Skew, Feature Shift, Label Imbalance)** 문제를 다루며, 근접 정규화 기반 연합학습 **FedProx의 성능을 높이는 것**을 목표로 합니다. 이 코드(`perf/fedprox-r18-cosine` 브랜치)는 **FedProx만** 학습하며 Centralized와 FedAvg 학습 경로는 제거되었습니다. 비교용 Centralized/FedAvg 수치는 `main`(8af4275)에서 얻은 값을 참고 기준으로만 인용합니다(`EXPERIMENT_REPORT.md` 참조).

단순 분류 정확도에 그치지 않고, 실제 의료 AI 임상 적용에서 필수적인 다음 5가지 축을 종합 평가합니다:
1. **불균형 분류 성능**: `Balanced Accuracy`, `Macro F1-Score` (존재 클래스 기준 주 지표 / 8클래스 고정 보조 지표)
2. **신뢰도 보정(Calibration)**: 15개 구간 기준 `ECE (Expected Calibration Error)`
3. **병원 간 공정성(Client Fairness)**: 6개 클라이언트 간 **`Macro-F1 표준편차 (macro_f1_std)` / `Balanced Accuracy 표준편차 (bal_acc_std)`**, **`최고-최저 격차 (macro_f1_gap, bal_acc_gap)`**, 그리고 **병원별 개별 성능 (`balanced_acc_client0~5`, `macro_f1_client0~5`)**
4. **클라이언트 드리프트(Client Drift)**: 글로벌-로컬 모델 간 `L2 가중치 발산도 (weight_divergence)`
5. **통신 효율성(Communication Efficiency)**: 목표 성능 도달까지의 `누적 통신량 (Communication-to-Accuracy, MB)`

---

## 2. 워크스페이스 파일 및 모듈 구조

| 파일명 | 구분 | 주요 클래스·함수 및 역할 |
| :--- | :---: | :--- |
| **`requirements.txt`** | 환경 설정 | 필수 패키지 명세 (`torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`, `albumentations==1.3.1`) |
| **`data_utils.py`** | 데이터 파이프라인 | • `get_dataloaders`: FLamby 공식 6개 센터(`Center 0 ~ 5`, `NUM_CLIENTS = 6`) 학습 로더, **Validation(Global/클라이언트별)** 로더, **Test(Global/클라이언트별)** 로더 생성<br>• `stratified_train_val_split` / `_build_center_datasets`: 센터별 FLamby train을 클래스 비율을 보존하며 **90% 학습 / 10% 검증** `Subset`으로 분할 (검증셋은 학습용 무작위 증강 없이 공식 test와 동일한 결정론적 변환 적용)<br>• FLamby 공식 클래스·상수·함수(`Baseline`, `BaselineLoss`, `Optimizer`, `BATCH_SIZE=64`, `LR=5e-4`, `NUM_CLIENTS=6`, `NUM_EPOCHS_POOLED=20`, `get_nb_max_rounds`) 임포트 및 제공<br>• `_seed_worker`: 멀티프로세싱 DataLoader 워커 난수 시드 고정 |
| **`federated_core.py`** | 모델 및 연합학습 코어 | • `ClientStepIterator`: FLamby `DataLoaderWithMemory`와 동일하게 라운드 간 반복자(iterator) 상태를 유지하며 정확히 `num_updates` 스텝의 미니배치를 연속 공급<br>• `build_efficientnet_b0`: FLamby 공식 `Baseline` 모델(`EfficientNet-B0` 사전학습, `BatchNorm2d` 유지, 8클래스 분류기) 우선 생성 (`torchvision` 호환 fallback 포함)<br>• `build_baseline_loss` / `_sync_criterion_device`: FLamby 공식 `BaselineLoss` (**Weighted Focal Loss**) 생성 및 내부 텐서 속성(`alpha` 등) 디바이스 재귀 동기화<br>• `train_one_client`: `FedProx`($\mu>0$) 로컬 학습 (`num_updates=100` 고정 스텝 및 `local_epochs` 에폭 모드 지원, FLamby `Optimizer` 상태 유지, 순수 `BaselineLoss` 반환, 배치 크기 1 방어)<br>• `server_aggregate`: 6개 클라이언트 샘플 수 비례 가중 평균 집계<br>• `evaluate_model`: 단일 패스 추론으로 Global 지표 및 6개 병원별 공정성 지표(`macro_f1_std`, `macro_f1_gap`, `bal_acc_std`, `bal_acc_gap`, `client_balanced_acc`, `client_macro_f1`) 동시 산출<br>• `compute_ece` / `compute_weight_divergence` / `compute_round_comm_mb`: 15-bin ECE, L2 가중치 발산도, 라운드당 양방향 통신량(MB) 계산 |
| **`lr_schedule.py`** | 학습률 스케줄 | • `round_lr`: `constant`(기본) 또는 `cosine` 라운드별 학습률 계산. torch에 의존하지 않아 CPU 단위테스트 가능(`tests/test_lr_schedule.py`) |
| **`scripts/run_fedprox_r18_cosine.sh`** | 실행 스크립트 | • FedProx(μ=0.1)를 18라운드 + cosine lr로 시드 42/100/2026에 대해 `--resume --final_test`로 순차 실행 |
| **`main.py`** | 학습 실행기 (CLI) | • `parse_args` / `set_seed`: FLamby 공식 하이퍼파라미터 및 학습 스케줄(`--num_updates 100`, `get_nb_max_rounds`) 적용, 전체 난수 고정, **FedProx 전용**(`--mu > 0` 강제)<br>• FLamby `_Model`과 동일하게 6개 클라이언트 로컬 모델(`client_models`) 및 로컬 옵티마이저(`client_optimizers = [Optimizer(...)]`)를 루프 밖에서 1회 생성하여 라운드 간 `Adam` 모멘텀 상태 유지<br>• `lr_schedule.py`의 `round_lr`로 **라운드별 lr(constant/cosine)** 을 계산해 클라이언트 옵티마이저의 `param_group["lr"]`만 갱신(Adam 모멘텀 상태는 유지)<br>• 매 라운드 **Validation** 지표 실시간 CSV 로깅, 최고 Val 모델(`best_*.pt`) 및 세션 중단 복구용 체크포인트(`last_*.pt`, `--resume`, 클라이언트 옵티마이저 상태 포함) 원자적 저장<br>• `--final_test` (`_run_final_test`): 학습 종료 후 FLamby **Test 셋을 단 1회만** 평가(`*_final_test.json`, 예측 `*_final_test_preds.npz` 저장) |
| **`DESIGN_DECISIONS_AND_CHANGELOG.md`** | 설계 문서 | • 아키텍처, 손실 함수, 공식 학습 스케줄, 클라이언트 옵티마이저 상태 유지, 평가 프로토콜 및 엔지니어링 최적화 내역과 학술적·기술적 근거 정리 |

---

## 3. 데이터셋 구성 (FLamby 공식 6-Client: `NUM_CLIENTS = 6`)

FLamby의 `FedIsic2019`는 실제 의료기관 및 더모스코피 장비 출처에 따라 **6개 병원/센터(Center 0 ~ 5, 전체 Train 18,597장 / Test 4,650장)** 로 자연 분할(Natural Cross-Silo Split)되어 있습니다.

| 클라이언트 | 원본 센터 (`center_id`) | FLamby Train (100%) | 학습 Subset (90%) | 검증 Subset (10%) | FLamby Test |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Client 0** | Center 0 | 9,930 | 약 8,937 | 약 993 | 2,483 |
| **Client 1** | Center 1 | 3,163 | 약 2,847 | 약 316 | 791 |
| **Client 2** | Center 2 | 2,691 | 약 2,422 | 약 269 | 672 |
| **Client 3** | Center 3 | 1,807 | 약 1,626 | 약 181 | 452 |
| **Client 4** | Center 4 | 655 | 약 590 | 약 65 | 164 |
| **Client 5** | Center 5 | 351 | 약 316 | 약 35 | 88 |
| **합계 (Global)** | **Center 0 ~ 5** | **18,597** | **약 16,738** | **약 1,859** | **4,650** |

> **Train / Validation / Test 분리 원칙**:
> - 각 센터의 FLamby 공식 `train` 데이터에서 **10%를 층화(Stratified) 분할하여 Validation**으로 사용하고(`--val_fraction 0.1`, 분할 시드 `--split_seed 0` 고정), 나머지 **90%만 학습**에 사용합니다.
> - 매 라운드 모니터링, 최고 성능 체크포인트(`best_*.pt`) 저장, FedProx의 최적 $\mu^*$ 선정은 모두 **Validation** 기준으로만 수행하며, 공식 **Test 셋은 학습이 완전히 끝난 뒤 `--final_test`로 단 1회만 평가**합니다.

---

## 4. 모델 아키텍처, 손실 함수, 옵티마이저 및 공식 학습 스케줄 (`--method`)

### 4.1 백본 모델(`Baseline`), 손실 함수(`BaselineLoss`) 및 옵티마이저(`Optimizer`)
- **모델 (`flamby.datasets.fed_isic2019.Baseline`)**:
  - FLamby 공식 `Baseline` 모델(ImageNet 사전학습 `EfficientNet-B0`, 내부 `BatchNorm2d` 유지, 8클래스 분류기)을 사용합니다.
- **손실 함수 (`flamby.datasets.fed_isic2019.BaselineLoss`)**:
  - 극심한 클래스 불균형과 어려운 샘플(Hard Examples)에 집중하는 FLamby 공식 **Weighted Focal Loss**를 사용합니다:
    $$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$
- **옵티마이저 (`flamby.datasets.fed_isic2019.Optimizer`)**:
  - FLamby 공식 `Optimizer` (`torch.optim.Adam`, `lr = 5e-4`)를 사용하며, FLamby 전략 래퍼(`_Model`)와 동일하게 **각 클라이언트의 옵티마이저 인스턴스를 루프 밖에서 1회 생성하여 라운드 간 1·2차 모멘텀 상태를 유지**합니다.

### 4.2 학습 방식: FedProx (6개 클라이언트, FLamby 공식 스케줄 기반)
- 각 클라이언트는 매 통신 라운드마다 **`num_updates = 100` 미니배치 스텝**(`ClientStepIterator`)을 수행합니다. 기본 라운드 수는 `get_nb_max_rounds(100, batch_size=64) = 9`이고, **이 브랜치의 실험은 `--rounds 18`로 늘려 돌립니다**(학습 예산 증가이므로 알고리즘 개선과 구분해 보고).
- 매 라운드 시작 시 각 클라이언트 모델(`client_models[k]`)은 `load_state_dict`로 글로벌 가중치를 동기화하되, 클라이언트별 `client_optimizers[k]`(`Adam`)는 초기화하지 않고 유지합니다.
- 로컬 목적함수에 글로벌 모델 $w_{\text{global}}^{(t)}$로부터의 이탈을 억제하는 근접 항을 더해 Non-IID 환경의 **Client Drift**를 완화합니다 (근접 항은 `model.parameters()` 전체에 대해 계산, BN 러닝 통계 버퍼는 제외):
  $$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{Focal}}(w) + \frac{\mu}{2} \left\| w - w_{\text{global}}^{(t)} \right\|_2^2$$
- CSV의 `train_loss`는 근접 항을 제외한 순수 $\mathcal{L}_{\text{Focal}}(w)$만 기록합니다.
- 서버는 6개 클라이언트의 학습 샘플 수($N_k$)에 비례한 가중 평균으로 글로벌 모델을 집계합니다:
  $$w_{\text{global}}^{(t+1)} = \sum_{k=1}^{6} \frac{N_k}{N} w_k^{(t+1)}$$
- **학습률 스케줄 (`--lr_schedule`)**: 기본 `constant`(lr 5e-4 고정, main과 동일). `cosine`이면 라운드 $r$의 lr이 $lr_{min} + 0.5(lr - lr_{min})(1+\cos(\pi (r-1)/R))$로 감쇠합니다(`--lr_min_ratio` 기본 0, 18라운드면 1라운드 5e-4 → 18라운드 약 3.8e-6). cosine이면 결과 파일 태그에 `_cosine`이 붙습니다.

---

## 5. 기록 및 평가 지표 명세

매 라운드 종료 시 `--save_dir` 내 `{exp_tag}_seed{seed}_metrics.csv`에 실시간 기록되는 컬럼입니다. **모든 라운드별 CSV 지표는 Validation 기준**(`eval_split=val`)이며, 최종 1회 Test 결과는 `{exp_tag}_seed{seed}_final_test.json` 및 `{exp_tag}_seed{seed}_final_test_preds.npz`에 저장됩니다.

| 컬럼명 | 의미 및 해석 방법 | 우수 방향 |
| :--- | :--- | :---: |
| `round` | 현재 통신 라운드 번호 (`1 ~ --rounds`) | - |
| `method` / `mu` / `seed` | 실험 식별 메타데이터 (`fedprox`, $\mu$, 난수 시드) | - |
| `eval_split` | 이 행의 지표를 계산한 평가 분할 (CSV는 항상 `val`) | - |
| **`train_loss`** | 학습 데이터의 샘플 수 가중 평균 순수 `BaselineLoss` (Weighted Focal Loss) | $\downarrow$ |
| **`val_loss`** | Global Validation 셋의 `BaselineLoss` (Weighted Focal Loss) | $\downarrow$ |
| **`balanced_acc`** | 8개 클래스별 재현율(Recall)의 산술 평균 (**FLamby 공식 핵심 분류 성능 지표** 및 `best_*.pt` 저장 기준) | $\uparrow$ |
| **`macro_f1`** | 평가셋 정답(`y_true`)에 **실제로 존재하는 클래스들**의 F1-Score 평균 (**주 지표**) | $\uparrow$ |
| `macro_f1_fixed8` | 8개 전체 클래스(`0..7`, 없는 클래스는 F1=0)로 평균한 보조 Macro-F1 | $\uparrow$ |
| **`ece`** | 15개 구간(`n_bins=15`) 기준 **Expected Calibration Error** (**모델 신뢰도 보정 지표**) | $\downarrow$ |
| `max_macro_f1` / `min_macro_f1` | 6개 클라이언트 로컬 평가셋 중 가장 높은 / 가장 낮은 `Macro-F1` (존재 클래스 기준) | $\uparrow$ |
| **`macro_f1_std`** | 6개 클라이언트 간 `Macro-F1` 표본 표준편차 ($\text{Std}_k(\text{F1}_k)$, **병원 간 공정성 핵심 지표**) | $\downarrow$ |
| **`macro_f1_gap`** | 6개 클라이언트 간 최고-최저 `Macro-F1` 격차 ($\max_k \text{F1}_k - \min_k \text{F1}_k$, **최악-최선 병원 격차 지표**) | $\downarrow$ |
| `max_bal_acc` / `min_bal_acc` | 6개 클라이언트 로컬 평가셋 중 가장 높은 / 가장 낮은 `Balanced Accuracy` | $\uparrow$ |
| **`bal_acc_std`** | 6개 클라이언트 간 `Balanced Accuracy` 표본 표준편차 ($\text{Std}_k(\text{BalAcc}_k)$, **병원 간 공정성 핵심 지표**) | $\downarrow$ |
| **`bal_acc_gap`** | 6개 클라이언트 간 최고-최저 `Balanced Accuracy` 격차 ($\max_k \text{BalAcc}_k - \min_k \text{BalAcc}_k$) | $\downarrow$ |
| **`balanced_acc_client0~5`** | 6개 클라이언트(병원 0~5)별 개별 `Balanced Accuracy` | $\uparrow$ |
| **`macro_f1_client0~5`** / `macro_f1_fixed8_client0~5` | 6개 클라이언트(병원 0~5)별 개별 `Macro-F1` (존재 클래스 기준 / 8클래스 고정) | $\uparrow$ |
| **`weight_divergence`** | 글로벌 모델과 6개 로컬 모델 파라미터 간 평균 L2 거리 ($\|w_{\text{local}} - w_{\text{global}}\|_2$, **Client Drift 지표**) | 분석용 |
| **`cumulative_comm_mb`** | 해당 라운드까지의 누적 양방향 통신량(MB). (`r × 2 × 6 × model_size_mb`, ) | $\downarrow$ |
| `lr` | 해당 라운드에 사용한 학습률 (`--lr_schedule cosine`이면 라운드마다 감소) | 분석용 |

---

## 6. CLI 인자 및 실행 방법

### 6.1 `main.py` 지원 인자 목록
| 인자 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--method` | `fedprox` | 학습 방식 (`fedprox`만 지원) |
| `--rounds` | `None` (자동) | 총 라운드 수. 미지정 시 `get_nb_max_rounds(num_updates)` (`9`) 자동 적용. cosine 스케줄은 이 값에 맞춰 감쇠하며, 체크포인트 resume 시 값이 같아야 함 |
| `--num_updates` | `100` | 라운드당 로컬 미니배치 업데이트(step) 횟수 (FLamby 공식 기본값: `100`) |
| `--local_epochs` | `1` | `--num_updates <= 0`일 때의 라운드당 로컬 에폭 수 |
| `--batch_size` | `BATCH_SIZE` (`64`) | 미니배치 크기 (`flamby.datasets.fed_isic2019.BATCH_SIZE`) |
| `--lr` | `LR` (`5e-4`) | `Optimizer` (`torch.optim.Adam`) 학습률 (`flamby.datasets.fed_isic2019.LR`) |
| `--lr_schedule` | `constant` | 라운드별 학습률 스케줄: `constant`(기본) 또는 `cosine` |
| `--lr_min_ratio` | `0.0` | cosine의 하한 비율 (최저 lr = `--lr` × 값) |
| `--mu` | `0.1` | FedProx 근접 항 계수. **0보다 커야 함**(0 이하면 오류) |
| `--seed` | `42` | 재현성 및 다중 시드 실험용 난수 시드 (탐색 시드: `41`, 본실험 3시드: `42, 100, 2026`) |
| `--num_workers` | `4` | DataLoader CPU 워커 프로세스 수 (코랩 기본 런타임 권장: `2`) |
| `--data_path` | `None` | 전처리된 `ISIC_2019_Training_Input_preprocessed` 폴더가 위치한 상위 디렉토리 경로 |
| `--save_dir` | `./results` | 결과 CSV, 체크포인트(`.pt`), 최종 테스트 JSON/NPZ 저장 경로 |
| `--resume` | `False` (플래그) | 동일 실험의 `last_*.pt`가 존재할 경우 마지막 완료 라운드 다음부터 이어서 학습 |
| `--val_fraction` | `0.1` | 센터별 FLamby train에서 층화로 떼어 낼 Validation 비율 |
| `--split_seed` | `0` | train/val 분할 시드 (실험 시드와 무관하게 고정) |
| `--final_test` | `False` (플래그) | 학습 종료 후 마지막 라운드 모델(과 best-val 모델)을 Test 셋으로 **단 1회** 평가해 저장 |

### 6.2 CLI 실행 예시
```bash
# FedProx 1회 실행 (기본: 9라운드, lr 고정, μ=0.1)
python main.py --method fedprox --mu 0.1 --seed 42 --data_path ./isic_data --save_dir ./results_fedprox_r9 --resume --final_test

# 18라운드 + cosine lr (이 브랜치의 실험 설정)
python main.py --method fedprox --mu 0.1 --rounds 18 --lr_schedule cosine --seed 42 --data_path ./isic_data --save_dir ./results_fedprox_r18_cosine --resume --final_test

# 시드 42/100/2026을 차례로 실행
bash scripts/run_fedprox_r18_cosine.sh ./isic_data
```

> 결과 파일 이름에 라운드 수가 들어가지 않으므로 **실험 조건마다 `--save_dir`를 다르게** 쓰세요. `--resume`은 설정(`rounds`, `lr_schedule` 포함)이 같을 때만 이어서 학습합니다.

---

## 7. 실험 프로토콜 (FedProx 전용)

1. **환경 준비**: FLamby 설치(`pip install -e ".[isic2019]"`, `requirements.txt` 패키지 설치)와 ISIC 전처리 데이터 준비를 먼저 합니다. 이전 Colab 노트북(`fed_isic2019_colab.ipynb`)은 FedProx 전용 코드와 맞지 않아 이 브랜치에서 삭제했습니다.
2. **실행**: `bash scripts/run_fedprox_r18_cosine.sh <data_path>`. 시드 `42, 100, 2026`을 순서대로 학습하고 종료 시 `--final_test`로 Test를 단 1회 평가합니다(`*_final_test.json`, `*_final_test_preds.npz`).
3. **비교**: 비교 기준은 `main`의 FedProx(μ=0.1, 9라운드) test balanced acc 0.6561과 참고용 FedAvg 0.6768, Centralized 0.7293(3시드 평균)입니다. 라운드 증가와 cosine을 함께 바꾸므로, 요인을 나누려면 `LR_SCHEDULE=constant`로 18라운드만 추가 실행합니다.
4. **μ**: 이전 재확인 실험에서 μ 0.1과 0.3은 구분되지 않았으므로 기본값은 0.1을 유지합니다.
