# 🏥 Fed-ISIC2019 의료 영상 연합학습 프로젝트 개요 (Project Overview)

## 1. 프로젝트 목적
본 프로젝트는 다기관 피부병변 분류 벤치마크인 **FLamby Fed-ISIC2019 (6-Client, 8-Class)** 공식 환경에서 발생하는 **병원 간 비독립·비동등 분포(Non-IID: Quantity Skew, Feature Shift, Label Imbalance)** 문제를 분석하고, 표준 연합학습인 **FedAvg**와 근접 정규화 기반 **FedProx**, 그리고 상한 기준선인 **Centralized** 학습을 다각도로 비교·검증하기 위해 설계되었습니다.

단순 분류 정확도 비교에 그치지 않고, 실제 의료 AI 임상 적용에서 필수적인 다음 5가지 축을 종합 평가합니다:
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
| **`data_utils.py`** | 데이터 파이프라인 | • `get_dataloaders`: FLamby 공식 6개 센터(`Center 0 ~ 5`, `NUM_CLIENTS = 6`) 학습 로더, Centralized 통합 로더, **Validation(Global/클라이언트별)** 로더, **Test(Global/클라이언트별)** 로더 생성<br>• `stratified_train_val_split` / `_build_center_datasets`: 센터별 FLamby train을 클래스 비율을 보존하며 **90% 학습 / 10% 검증** `Subset`으로 분할 (검증셋은 학습용 무작위 증강 없이 공식 test와 동일한 결정론적 변환 적용)<br>• FLamby 공식 상수 및 함수(`BATCH_SIZE=64`, `LR=5e-4`, `NUM_CLIENTS=6`, `NUM_EPOCHS_POOLED=20`, `BaselineLoss`, `get_nb_max_rounds`) 임포트 및 제공<br>• `_seed_worker`: 멀티프로세싱 DataLoader 워커 난수 시드 고정 |
| **`federated_core.py`** | 모델 및 연합학습 코어 | • `ClientStepIterator`: FLamby `DataLoaderWithMemory`와 동일하게 라운드 간 반복자(iterator) 상태를 유지하며 정확히 `num_updates` 스텝의 미니배치를 연속 공급<br>• `build_efficientnet_b0`: ImageNet 사전학습 `EfficientNet-B0` 로드 (`BatchNorm2d` 유지, 8클래스 분류기 교체)<br>• `build_baseline_loss` / `_sync_criterion_device`: FLamby 공식 `BaselineLoss` (**Weighted Focal Loss**) 생성 및 내부 텐서 속성(`alpha` 등) 디바이스 재귀 동기화<br>• `train_one_client`: `FedAvg`($\mu=0$) 및 `FedProx`($\mu>0$) 로컬 학습 (`num_updates=100` 고정 스텝 및 `local_epochs` 에폭 모드 지원, 순수 `BaselineLoss` 반환, 배치 크기 1 방어)<br>• `server_aggregate`: 6개 클라이언트 샘플 수 비례 가중 평균 집계<br>• `evaluate_model`: 단일 패스 추론으로 Global 지표 및 6개 병원별 공정성 지표(`macro_f1_std`, `macro_f1_gap`, `bal_acc_std`, `bal_acc_gap`, `client_balanced_acc`, `client_macro_f1`) 동시 산출<br>• `compute_ece` / `compute_weight_divergence` / `compute_round_comm_mb`: 15-bin ECE, L2 가중치 발산도, 라운드당 양방향 통신량(MB) 계산 |
| **`main.py`** | 학습 실행기 (CLI) | • `parse_args` / `set_seed`: FLamby 공식 하이퍼파라미터 및 공식 학습 스케줄(`--num_updates 100`, `get_nb_max_rounds`) 적용, 전체 난수 고정<br>• 매 라운드 **Validation** 지표 실시간 CSV 로깅(`macro_f1_std`, `bal_acc_std`, `balanced_acc_client0~5`, `macro_f1_client0~5` 포함), 최고 Val 모델(`best_*.pt`) 및 세션 중단 복구용 체크포인트(`last_*.pt`, `--resume`) 원자적 저장<br>• `--final_test` (`_run_final_test`): 학습 종료 후 FLamby **Test 셋을 단 1회만** 평가(`*_final_test.json`, 예측 `*_final_test_preds.npz` 저장) |
| **`fed_isic2019_colab.ipynb`** | 협업 실행 노트북 | • Phase 0~1: GPU 확인, Drive 마운트, FLamby 설치, ISIC 데이터 전처리(`tar` 캐싱) 및 6-Client Smoke Test<br>• Phase 2~4: **2단계 시드 분리 프로토콜**(Stage 1 `seed=41` 탐색 $\rightarrow$ Stage 2 `seed=42, 100, 2026` 본실험 및 `--final_test` 1회 평가)<br>• Phase 5: **Validation 및 Test `Mean ± Std` 요약표(Final 주 결과 / Best-Val 참고)**, **6개 병원별(Client 0~5) Test 세부 요약표(`summary_test_per_client_mean_std.csv`)**, 목표 정확도 도달 통신량 집계표, 학습 곡선 및 통신 효율성 그래프 자동 생성 |
| **`DESIGN_DECISIONS_AND_CHANGELOG.md`** | 설계 문서 | • 아키텍처, 손실 함수, 공식 학습 스케줄, 평가 프로토콜 및 엔지니어링 최적화 내역과 학술적·기술적 근거 정리 |

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

## 4. 모델 아키텍처, 손실 함수 및 공식 학습 스케줄 (`--method`)

### 4.1 백본 모델 및 FLamby 공식 손실 함수 (`BaselineLoss`)
- **모델**: `torchvision.models.efficientnet_b0(pretrained=True)`
  - ImageNet 사전학습 특징 분포 보존을 위해 내부 `BatchNorm2d`를 유지하고, 최종 분류기(`classifier[1]`)만 `nn.Linear(1280, 8)`로 교체합니다.
- **손실 함수 (`flamby.datasets.fed_isic2019.BaselineLoss`)**:
  - 극심한 클래스 불균형과 어려운 샘플(Hard Examples)에 집중하는 FLamby 공식 **Weighted Focal Loss**를 사용합니다:
    $$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

### 4.2 FLamby 공식 학습 스케줄 및 비교 기법 (3종)
1. **`centralized` (상한 기준선)**:
   - 6개 클라이언트의 학습 데이터를 하나로 모아 단일 글로벌 모델을 학습합니다.
   - FLamby 공식 Pooled 스케줄에 따라 라운드당 `local_epochs = 1` 에폭씩 총 **`NUM_EPOCHS_POOLED = 20` 에폭** 동안 학습하며(`batch_size = 64`, `lr = 5e-4`), 에폭 간 Adam 옵티마이저 모멘텀을 유지합니다.
2. **`fedavg` ($\mu = 0.0$)**:
   - FLamby 공식 연합학습 스케줄에 따라 각 클라이언트가 매 통신 라운드마다 **`num_updates = 100` 미니배치 스텝**(`ClientStepIterator`)을 수행하고, 총 통신 라운드 수는 `get_nb_max_rounds(100, batch_size=64)`에 의해 **`9` 라운드**(클라이언트당 총 900 스텝)로 설정됩니다.
   - 서버는 6개 클라이언트의 학습 샘플 수($N_k$)에 비례한 가중 평균으로 글로벌 모델을 집계합니다:
     $$w_{\text{global}}^{(t+1)} = \sum_{k=1}^{6} \frac{N_k}{N} w_k^{(t+1)}$$
3. **`fedprox` ($\mu > 0.0$)**:
   - `fedavg`와 동일하게 라운드당 **`num_updates = 100` 스텝**(`9` 라운드)을 수행하되, 로컬 학습 목적함수에 글로벌 모델 $w_{\text{global}}^{(t)}$로부터의 이탈을 억제하는 근접 정규화 항(Proximal Term)을 추가하여 Non-IID 환경의 **Client Drift**를 완화합니다:
     $$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{Focal}}(w) + \frac{\mu}{2} \left\| w - w_{\text{global}}^{(t)} \right\|_2^2$$
   - 단, CSV에 기록되는 `train_loss`는 FedAvg와 동일 선상에서 비교할 수 있도록 근접 항을 제외한 순수 $\mathcal{L}_{\text{Focal}}(w)$만 기록합니다.

---

## 5. 기록 및 평가 지표 명세

매 라운드 종료 시 `--save_dir` 내 `{exp_tag}_seed{seed}_metrics.csv`에 실시간 기록되는 컬럼입니다. **모든 라운드별 CSV 지표는 Validation 기준**(`eval_split=val`)이며, 최종 1회 Test 결과는 `{exp_tag}_seed{seed}_final_test.json` 및 `{exp_tag}_seed{seed}_final_test_preds.npz`에 저장됩니다.

| 컬럼명 | 의미 및 해석 방법 | 우수 방향 |
| :--- | :--- | :---: |
| `round` | 현재 통신 라운드 번호 (연합학습 `1 ~ 9`, Centralized `1 ~ 20`) | - |
| `method` / `mu` / `seed` | 실험 식별 메타데이터 (`centralized`/`fedavg`/`fedprox`, $\mu$, 난수 시드) | - |
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
| **`cumulative_comm_mb`** | 해당 라운드까지의 누적 양방향 통신량(MB). (`r × 2 × 6 × model_size_mb`, Centralized는 `0`) | $\downarrow$ |

---

## 6. CLI 인자 및 실행 방법

### 6.1 `main.py` 지원 인자 목록
| 인자 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--method` | `fedavg` | 학습 방식 선택: `centralized`, `fedavg`, `fedprox` |
| `--rounds` | `None` (자동) | 총 라운드 수. 미지정 시 연합학습은 `get_nb_max_rounds(num_updates)` (`9`), Centralized는 `NUM_EPOCHS_POOLED` (`20`) 자동 적용 |
| `--num_updates` | `100` | 연합학습 라운드당 로컬 미니배치 업데이트(step) 횟수 (FLamby 공식 기본값: `100`) |
| `--local_epochs` | `1` | Centralized의 라운드당 에폭 수 (또는 `--num_updates <= 0`일 때의 로컬 에폭 수) |
| `--batch_size` | `BATCH_SIZE` (`64`) | 미니배치 크기 (`flamby.datasets.fed_isic2019.BATCH_SIZE`) |
| `--lr` | `LR` (`5e-4`) | Adam 옵티마이저 학습률 (`flamby.datasets.fed_isic2019.LR`) |
| `--mu` | `0.01` | FedProx 근접 항 계수 (`fedavg`/`centralized`에서는 자동으로 `0.0` 적용) |
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
# 0. Stage 1 하이퍼파라미터 탐색 (탐색 전용 seed=41, 공식 스케줄 9 라운드 × 100 스텝 자동 적용, Validation만 평가)
python main.py --method fedprox --num_updates 100 --mu 0.01 --seed 41 --data_path ./isic_data --save_dir ./results --resume

# 1. Centralized Baseline 본실험 (NUM_EPOCHS_POOLED=20 에폭 자동 적용, 학습 종료 후 1회 --final_test)
python main.py --method centralized --local_epochs 1 --seed 42 --data_path ./isic_data --save_dir ./results --resume --final_test

# 2. FedAvg 본실험 (num_updates=100, 9 라운드 자동 적용, 학습 종료 후 1회 --final_test)
python main.py --method fedavg --num_updates 100 --mu 0.0 --seed 42 --data_path ./isic_data --save_dir ./results --resume --final_test

# 3. FedProx 본실험 (선정된 BEST_MU 적용, num_updates=100, 9 라운드 자동 적용, 학습 종료 후 1회 --final_test)
python main.py --method fedprox --num_updates 100 --mu 0.01 --seed 42 --data_path ./isic_data --save_dir ./results --resume --final_test
```

---

## 7. Google Colab 단계별 실행 프로토콜 (`fed_isic2019_colab.ipynb`)

1. **Phase 0 & Phase 1 (환경, 데이터 준비 및 Smoke Test)**:
   - Google Drive를 마운트하고 FLamby(`pip install -e ".[isic2019]"`)를 설치합니다.
   - 전처리된 이미지(23,247장)는 최초 1회만 생성하여 Drive에 `isic2019_preprocessed.tar`로 보관하고, 이후 세션에서는 로컬 디스크(`/content/isic_data`)로 빠르게 압축 해제하여 고속 I/O를 확보합니다.
   - Smoke Test(셀 1-4)에서 6개 클라이언트의 Train(90%) / Val(10%) / Test 샘플 수와 공식 스케줄(`get_nb_max_rounds(100) = 9`), `BaselineLoss` 정상 작동을 확인합니다.
2. **Phase 2 & Phase 3 (Stage 1 — 탐색 시드 `seed=41`로 최적 $\mu^*$ 선정)**:
   - 탐색 전용 시드 `41`로 `FedAvg`($\mu=0$ 기준선)와 `FedProx`의 $\mu$ 후보군(`0.001, 0.01, 0.1, 1.0`)을 공식 스케줄(`FL_ROUNDS=9`, `NUM_UPDATES=100`)로 1회씩 실행합니다.
   - **Validation 마지막 3라운드(`LAST_K = 3`) 평균 지표**(`Val Macro F1`, `Val Bal Acc`, `Val Client F1 Std`, `Val Client BalAcc Std`, `Weight Divergence`)를 비교하여 Phase 4의 `BEST_MU`를 직접 지정합니다. (이 시드의 결과는 최종 요약표에서 제외됩니다.)
3. **Phase 4 (Stage 2 — 본실험 3시드 `42, 100, 2026` 검증 및 1회성 Test 평가)**:
   - `Centralized`(20 에폭), `FedAvg`(9 라운드 × 100 스텝), `FedProx(BEST_MU)`(9 라운드 × 100 스텝)를 본실험 시드 `42, 100, 2026`으로 실행하고, 학습 종료 시 `--final_test`로 Test 셋을 단 1회 평가합니다.
4. **Phase 5 (최종 요약표 및 그래프 자동 생성)**:
   - `summary_final_mean_std.csv` / `summary_best_mean_std.csv`: Validation 기준 마지막 라운드(Final) 및 최고 라운드(Best) 3시드 `Mean ± Std` 요약표
   - `summary_test_final_mean_std.csv` / `summary_test_best_val_mean_std.csv`: `--final_test`로 단 1회 평가한 **최종 Test 3시드 `Mean ± Std` 요약표** (**마지막 라운드 모델이 보고서 주 결과**, best-val 모델은 참고용)
   - `summary_test_per_client_mean_std.csv`: 6개 병원별(`Client 0 ~ 5`) Test `Balanced Accuracy` 및 `Macro F1` 3시드 `Mean ± Std` 세부 비교표
   - `comm_to_target.csv`: 목표 정확도 도달 비율(`Reached`), 평균 도달 라운드, 평균 누적 통신량(MB) 비교표
   - `comparison_curves_mean_std.png` & `comm_efficiency.png`: 4대 지표(`Bal Acc`, `Client Macro-F1 Std`, `ECE`, `Weight Divergence`) 라운드별 추이 그래프(`±1 Std` 음영 포함) 및 누적 통신량 대비 정확도 효율성 그래프
