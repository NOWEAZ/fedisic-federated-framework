# 🛠️ 설계 의사결정 및 엔지니어링 내역서 (Design Decisions & Engineering Log)

본 문서는 **FLamby Fed-ISIC2019 (6-Client, 8-Class) 의료 영상 연합학습 프로젝트**의 핵심 설계 의사결정(아키텍처, 손실 함수, 학습 스케줄, 평가 프로토콜, 평가지표)과 런타임 안정화·최적화 내역을 기술적·학술적 근거와 함께 정리한 문서입니다. 보고서 및 논문 작성 시 **실험 설계 근거(Experimental Setup & Justification)** 로 바로 활용할 수 있습니다.

---

## 📋 전체 설계 의사결정 요약표

| 구분 | 설계 항목 | 최종 채택 내용 | 핵심 기술적·학술적 근거 요약 |
| :---: | :--- | :--- | :--- |
| **1** | **클라이언트 구성 및 하이퍼파라미터** | **FLamby 공식 6-Client (`Center 0 ~ 5`)** 및 공식 상수(`BATCH_SIZE=64`, `LR=5e-4`, `NUM_EPOCHS_POOLED=20`) | 실제 6개 의료기관·장비의 자연 분할(Natural Cross-Silo Split)과 FLamby 원 논문 벤치마크(NeurIPS 2022) 규격을 그대로 준수하여 재현성 확보 |
| **2** | **모델 및 정규화 계층** | ImageNet 사전학습 **`EfficientNet-B0` (`BatchNorm2d` 유지)** | `GroupNorm` 교체 시 발생하는 사전학습 가중치 분포 붕괴(Representation Collapse)를 방지하고 고채널(320, 1280) 병변 특징 해상도 보존 |
| **3** | **손실 함수 (Loss Function)** | FLamby 공식 **`BaselineLoss` (Weighted Focal Loss)** | 극심한 클래스 불균형과 어려운 샘플(Hard Examples)에 집중하며, 모든 클라이언트가 동일한 목적함수 정의를 공유하도록 통일 |
| **4** | **학습 스케줄 (Training Schedule)** | 연합학습 **`num_updates=100` 스텝 (`get_nb_max_rounds` = `9` 라운드)**, Centralized **`20` 에폭** | 병원 간 최대 28배(9,930장 vs. 351장)에 달하는 샘플 수 격차 환경에서 라운드당 로컬 경사하강 횟수를 100 스텝으로 통일 (`ClientStepIterator` 적용) |
| **5** | **데이터 분할 및 평가 프로토콜** | 센터별 **층화 90/10 Train/Val 분할** 및 학습 종료 후 **`--final_test` 단 1회 평가** | 매 라운드 모니터링·모델 선택·$\mu$ 튜닝을 Validation으로만 수행하여 Test 데이터 누수(Test-set Snooping) 및 선택 편향(Selection Bias) 원천 차단 |
| **6** | **다중 시드 실험 전략** | **2단계 시드 분리** (`seed=41` 탐색 $\rightarrow$ `seed=42, 100, 2026` 본실험) | 하이퍼파라미터($\mu$) 선택 시드와 최종 보고용 3시드를 완전히 분리하여 통계적 엄밀성(`Mean ± Std`) 확보 및 코랩 탐색 시간 절감 |
| **7** | **다각도 평가지표 체계** | **`Balanced Acc`**, **`Macro-F1`**, 병원별 **`balanced_acc_client0~5` / `macro_f1_client0~5`**, 병원 간 **`macro_f1_std` / `bal_acc_std`** 및 **`Gap`**, **`15-bin ECE`**, **`Weight Divergence`**, **`Cumulative Comm (MB)`** | 불균형 분류 성능뿐만 아니라 초소형 병원 노이즈에 강건한 병원 간 공정성(Fairness: Std & Gap), 신뢰도 보정(Calibration), Client Drift, 통신 효율성을 종합 평가 |
| **8** | **엔지니어링 최적화 및 안정화** | 단일 패스 추론, `Centralized` 모멘텀 유지, 순수 손실 로깅, `BN` 크기 1 방어, `--resume` 원자적 체크포인트 | 평가 속도 2배 향상, 상한선 공정 비교, 코랩 세션 중단 시 에폭 중간 위치까지 복원하는 재개 및 GPU/CPU 디바이스 동기화 보장 |

---

## 1. 클라이언트 구성 및 공식 벤치마크 규격 채택 (`data_utils.py`, `main.py`)

### 📌 설계 내용
- `flamby.datasets.fed_isic2019`가 제공하는 **공식 6개 병원/센터(`Center 0 ~ 5`, `NUM_CLIENTS = 6`)** 자연 분할과 공식 하이퍼파라미터 상수(`BATCH_SIZE = 64`, `LR = 5e-4`, `NUM_EPOCHS_POOLED = 20`)를 코드에서 직접 임포트하여 적용함.

### 💡 기술적·학술적 근거
1. **현실적인 의료 Non-IID 환경 반영**:
   - `Fed-ISIC2019`는 실제 4개 의료기관 및 6종 더모스코피 장비(BCN, Vienna 3종, MSK, Rosendahl)에서 수집된 데이터로, 대형 병원인 Center 0(학습 9,930장)부터 초소형 병원인 Center 4(655장), Center 5(351장)까지 **샘플 수 불균형(Quantity Skew)**, **장비별 화질·색감 차이(Feature Shift)**, **병원별 결측 클래스를 동반한 레이블 편향(Label Distribution Skew)** 이 동시에 존재함.
2. **공식 벤치마크 정합성**:
   - FLamby 원 논문(Ogier du Terrail et al., NeurIPS 2022)의 상수 및 데이터 분할을 그대로 사용함으로써 실험 결과의 객관적 비교 가능성(Reproducibility)을 확보함.

---

## 2. 모델 아키텍처 및 정규화 계층: `GroupNorm(2)` 대신 `BatchNorm2d` 유지 (`federated_core.py`)

### 📌 설계 내용
- `torchvision.models.efficientnet_b0(pretrained=True)`를 로드하여 내부 `BatchNorm2d` 계층을 그대로 유지하고, 최종 분류기(`classifier[1]`)만 `nn.Linear(1280, 8)`로 교체함.

### 💡 기술적·학술적 근거
1. **사전학습(Pretrained) 가중치 분포 보존**:
   - ImageNet 사전학습 합성곱(Conv) 가중치들은 기존 `BatchNorm2d`의 채널별 정규화 스케일을 전제로 학습되어 있음.
   - 이를 `GroupNorm`으로 사후 교체하면 중간 활성화(Activation) 분포가 급격히 틀어져 **초기 라운드에서 사전학습된 특징 추출 능력이 붕괴(Representation Collapse)** 되며, 제한된 통신 라운드 내에 수렴하지 못하는 손실이 발생함.
2. **고채널 특징 해상도 유지**:
   - `EfficientNet-B0` 뒷단 채널 수는 `320`, `1280`까지 커지는데, 소수 그룹의 `GroupNorm`으로 묶으면 개별 채널의 미세한 피부 병변 질감·색상 특징 스케일이 뭉개지므로 Pretrained `BatchNorm2d`를 유지하는 것이 타당함.

---

## 3. 손실 함수: FLamby 공식 `BaselineLoss` (Weighted Focal Loss) 적용 (`federated_core.py`)

### 📌 설계 내용
- `flamby.datasets.fed_isic2019.BaselineLoss` (Weighted Focal Loss)를 모든 학습(`Centralized`, `FedAvg`, `FedProx`) 및 평가(`Val`, `Test`)의 기본 분류 손실 함수로 사용함:
  $$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$
- `FedProx`($\mu > 0$) 학습 시에는 역전파 목적함수에만 근접 정규화 항 $\frac{\mu}{2}\|w - w_{\text{global}}^{(t)}\|_2^2$을 더하고, CSV에 기록하는 `train_loss`에는 근접 항을 제외한 순수 $\mathcal{L}_{\text{Focal}}$만 누적하여 `FedAvg`와 동일 스케일에서 비교할 수 있게 함.

### 💡 기술적·학술적 근거
1. **클래스 불균형 및 어려운 샘플(Hard Examples) 집중 학습**:
   - ISIC 2019는 멜라노마·모반 등 다수 클래스와 피부섬유종·혈관병변 등 희소 클래스 간 비율 차이가 극심함. 클래스별 가중치 $\alpha_t$와 초점 계수 $(1 - p_t)^{\gamma}$를 결합한 Weighted Focal Loss는 쉽게 맞추는 다수 클래스 샘플의 그래디언트 비중을 낮추고 오분류하기 쉬운 희소 병변에 학습을 집중시킴.
2. **클라이언트 간 목적함수 정의 통일**:
   - 병원마다 로컬 클래스 빈도로 서로 다른 가중치를 주면 클라이언트별 손실 함수의 최적점 자체가 달라져 인위적인 그래디언트 충돌이 커질 수 있는 반면, 공식 `BaselineLoss`는 모든 클라이언트가 동일한 클래스 가중치 $\alpha$를 공유하여 안정적인 서버 집계를 지원함.

---

## 4. 학습 스케줄: FLamby 공식 `num_updates = 100` 및 `ClientStepIterator` (`federated_core.py`, `main.py`)

### 📌 설계 내용
- **연합학습 (`FedAvg`, `FedProx`)**:
  - 각 클라이언트가 매 통신 라운드마다 에폭 단위가 아닌 **고정 미니배치 업데이트 횟수 `num_updates = 100` 스텝**을 수행함.
  - 총 통신 라운드 수는 FLamby 공식 함수인 `get_nb_max_rounds(num_updates=100, batch_size=64)`를 통해 **`9` 라운드**(클라이언트당 총 900 스텝)로 설정됨.
  - 라운드 간 DataLoader 반복자(iterator) 상태를 유지하는 **`ClientStepIterator`** (FLamby `DataLoaderWithMemory` 동등 구현)를 적용함.
- **중앙집중 학습 (`Centralized`)**:
  - FLamby 공식 Pooled 설정에 따라 라운드당 `1` 에폭씩 **`NUM_EPOCHS_POOLED = 20` 에폭**을 수행함.

### 💡 기술적·학술적 근거
1. **병원 간 로컬 업데이트 스텝 수 균등화**:
   - 90% 학습 분할 및 `batch_size = 64` 기준, Center 0(8,937장)의 1 에폭은 약 **140 스텝**인 반면 Center 5(313장)의 1 에폭은 약 **5 스텝**으로 **28배** 차이가 남.
   - 만약 로컬 에폭 수를 고정하면 한 라운드 동안 대형 병원은 140 스텝을 이동하고 소형 병원은 5 스텝만 이동하여 로컬 업데이트 크기 불균형이 발생함. 라운드당 `num_updates = 100` 스텝으로 통일하면 모든 병원이 동일한 경사하강 횟수를 수행한 뒤 집계되므로 공정한 비교가 가능함.
2. **`ClientStepIterator`를 통한 대형 병원 데이터 편향 방지**:
   - 매 라운드 `iter(dataloader)`를 새로 열면 140 배치를 가진 Center 0은 매번 에폭의 앞쪽 100 배치만 학습하고 뒤쪽 40 배치는 버려지게 됨. `ClientStepIterator`는 이전 라운드가 끝난 배치 위치에서 다음 라운드를 이어서 진행하고, 데이터셋 끝에 도달했을 때만 새 에폭을 시작하여 모든 학습 데이터를 빠짐없이 활용함.

---

## 5. 데이터 분할 및 평가 프로토콜: 층화 90/10 Validation 분할 및 1회성 `--final_test` (`data_utils.py`, `main.py`)

### 📌 설계 내용
- **층화 90/10 Train/Val 분할 (`stratified_train_val_split`, `_build_center_datasets`)**:
  - 각 센터(Center 0 ~ 5)의 FLamby 공식 `train=True` 데이터셋을 클래스 비율을 유지하며 **90% 학습(`train`) / 10% 검증(`val`)** `Subset`으로 분할함.
  - 분할 시드는 실험 시드(`--seed`)와 독립된 `split_seed + center` (`--split_seed 0`)로 고정하여 모든 기법과 시드가 완전히 동일한 검증셋을 공유함.
  - 검증용 `Subset`(`val_view`)은 학습용 무작위 증강 대신 공식 테스트셋과 동일한 결정론적 변환(`CenterCrop + Normalize`)을 적용함.
- **1회성 최종 테스트 평가 (`--final_test`, `_run_final_test`)**:
  - 매 라운드 CSV 기록(`eval_split=val`), 최고 모델 체크포인트(`best_*.pt`) 갱신, FedProx 최적 $\mu^*$ 선정은 오직 **Validation 셋**으로만 수행함.
  - FLamby 공식 **Test 셋(`train=False`)** 은 학습이 완전히 끝난 뒤 `--final_test` 플래그를 통해 **단 1회만 평가**하여 `*_final_test.json`(마지막 라운드 모델 주 결과 + best-val 모델 참고 결과) 및 `*_final_test_preds.npz`(사후 부트스트랩 분석용 예측 배열)로 저장하며, 이미 JSON 파일이 존재하면 재평가를 차단함.

### 💡 기술적·학술적 근거
1. **Test 데이터 누수(Test-set Snooping) 및 선택 편향(Selection Bias) 원천 차단**:
   - 별도 검증셋 없이 매 라운드 Test 셋을 평가하여 최고 성능 라운드(`Best`)나 하이퍼파라미터($\mu$)를 고르면, 테스트셋 노이즈에 맞춘 낙관적 편향(Optimistic Bias)이 발생함.
   - 선택(Selection)과 최종 평가(Reporting)에 쓰이는 데이터를 엄격히 분리하여 학술적 타당성을 확립함.

---

## 6. 다중 시드 실험 전략: 2단계 시드 분리 (`seed=41` 탐색 $\rightarrow$ `42, 100, 2026` 본실험) (`fed_isic2019_colab.ipynb`)

### 📌 설계 내용
- **Stage 1 (Phase 3 — 탐색 전용 `EXPLORE_SEED = 41`)**:
  - `FedAvg`($\mu=0$ 기준선)와 `FedProx` 후보군($\mu \in \{0.001, 0.01, 0.1, 1.0\}$)을 `seed=41`로 1회씩 실행하고, 단일 라운드 노이즈를 줄이기 위해 **Validation 마지막 3라운드(`LAST_K = 3`) 평균 지표**를 비교하여 사용자가 최적 `BEST_MU`를 선정함. 이 시드의 결과는 최종 요약표에서 제외함.
- **Stage 2 (Phase 4 — 본실험 3시드 `FINAL_SEEDS = [42, 100, 2026]`)**:
  - 확정된 3가지 기법(`Centralized`, `FedAvg`, `FedProx(BEST_MU)`)을 독립된 본실험 3개 시드로 실행하고 `--final_test`로 최종 Test `Mean ± Std` 요약표를 생성함.

### 💡 기술적·학술적 근거
1. **하이퍼파라미터 선택 편향 배제 및 코랩 GPU 시간 절약**:
   - 모든 $\mu$ 후보를 3시드로 돌리는 탐색 비용을 줄이면서도, $\mu$ 선택에 쓰인 탐색 시드(`41`)를 최종 보고 시드(`42, 100, 2026`)와 완전히 분리하여 특정 시드에 우연히 유리한 하이퍼파라미터가 최종 성적을 부풀리는 문제를 방지함.

---

## 7. 다각도 평가지표 체계 및 병원 간 공정성(Fairness) 정밀화 (`federated_core.py`, `main.py`, 노트북)

### 📌 설계 내용
1. **Macro-F1 이중 정의 (`macro_f1` 주 지표 / `macro_f1_fixed8` 보조 지표)**:
   - 병원마다 보유하지 않은 결측 클래스(샘플 0개)가 존재하므로, `macro_f1` 및 클라이언트별 `client_macro_f1`은 **해당 평가셋 정답(`y_true`)에 실제로 존재하는 클래스(`np.unique(y_true)`)만** 평균하여 산출함.
   - 보조 지표로 8개 전체 클래스(`0..7`, 없는 클래스는 F1=0) 고정 평균인 `macro_f1_fixed8`도 함께 기록함.
2. **병원별 개별 성능 (`balanced_acc_client0~5`, `macro_f1_client0~5`) 및 공정성 지표 이중화 (`Std` & `Gap`)**:
   - FLamby 공식 주 지표인 `Balanced Accuracy`와 보조 지표인 `Macro-F1` 모두에 대해 6개 병원 각각의 성능(`balanced_acc_client0~5`, `macro_f1_client0~5`)을 기록하고, Phase 5에서 `summary_test_per_client_mean_std.csv`로 병원별 `Mean ± Std`를 자동 집계함.
   - 병원 간 공정성(Client Fairness)을 측정하기 위해 다음 두 지표를 함께 산출함:
     - **클라이언트 간 표준편차 (`macro_f1_std`, `bal_acc_std`)**: 6개 병원 전체의 성능 분산($\text{Std}_k$, `ddof=1`)을 측정하는 핵심 공정성 지표.
     - **최고-최저 격차 (`macro_f1_gap`, `bal_acc_gap`)**: 최선 병원과 최악 병원 간 최대 격차($\max_k - \min_k$)를 측정하는 보조 공정성 지표.
3. **15-Bin Expected Calibration Error (`ece`, `n_bins=15`)**:
   - Guo et al. (ICML 2017) 및 `torchmetrics` 표준인 15개 신뢰도 구간으로 예측 확률과 실제 정확도 간 괴리를 측정함.
4. **Client Drift (`weight_divergence`) 및 통신 효율성 (`cumulative_comm_mb`)**:
   - 매 라운드 글로벌 모델과 6개 로컬 모델 간 평균 L2 거리($\|w_{\text{local}} - w_{\text{global}}\|_2$) 및 양방향 누적 통신량(`2 × 6 × model_size_mb`)을 기록하여 목표 정확도 도달까지의 통신 비용(`Communication-to-Accuracy`)을 비교함.

### 💡 기술적·학술적 근거 (`Std` 및 병원별 `Bal Acc` 도입 배경)
1. **초소형 병원(Client 4, Client 5)의 소수 샘플 노이즈 완화**:
   - 6-클라이언트 환경에서 Client 5는 검증셋이 **38장**(테스트셋 88장), Client 4는 검증셋이 **66장**(테스트셋 164장)으로 작아, 극단값 2개($\max - \min$)만 보는 `Gap` 지표는 특정 소형 병원의 샘플 몇 장이 바뀔 때 크게 흔들릴 수 있음.
   - 반면 6개 병원 전체의 표준편차인 **`macro_f1_std`와 `bal_acc_std`** 는 연합학습 공정성 문헌(Li et al., *Fair Resource Allocation in Federated Learning*, ICLR 2020)의 표준 지표로서 단일 병원 노이즈에 훨씬 강건하게 병원 간 성능 편차를 반영함.
2. **FLamby 공식 주 지표(`Balanced Accuracy`)와의 일관성 확보**:
   - Global 주 지표가 `Balanced Accuracy`이므로, 병원별 세부 성능 역시 `macro_f1_client0~5`뿐만 아니라 `balanced_acc_client0~5`와 `bal_acc_std`를 함께 제시함으로써 Global 성능과 병원별 공정성 분석의 기준을 완벽히 일치시킴.

---

## 8. 엔지니어링 최적화 및 런타임 안정화 내역 (`federated_core.py`, `main.py`, 노트북)

| 번호 | 대상 파일 | 구현 내용 | 엔지니어링 근거 |
| :---: | :--- | :--- | :--- |
| **8.1** | `federated_core.py` | **`evaluate_model` 단일 패스 추론** | 6개 클라이언트별 평가 로더(`client_eval_loaders`)의 추론 결과를 `np.concatenate`하여 Global 지표와 병원별 공정성 지표(`F1` 및 `Bal Acc`의 `Std`/`Gap`)를 한 번의 순회로 동시 산출(평가 시간 50% 단축). |
| **8.2** | `federated_core.py` | **`BaselineLoss` 재귀적 디바이스 동기화 (`_sync_criterion_device`)** | FLamby `BaselineLoss` 내부에 PyTorch 버퍼로 등록되지 않은 일반 텐서 속성(`self.alpha` 등)을 재귀적으로 탐색해 `CPU`/`CUDA` 디바이스로 이동시켜 런타임 디바이스 불일치 에러 원천 차단. |
| **8.3** | `federated_core.py` | **`BatchNorm2d` 단일 샘플 배치 방어** | 미니배치 크기가 1일 때(`images.size(0) == 1`) `BatchNorm2d` 분산 계산 에러(`ValueError`)가 발생하지 않도록 건너뛰되, `num_updates` 스텝 카운트는 차감하지 않아 정확한 업데이트 횟수 보장. |
| **8.4** | `main.py` | **`Centralized`의 Adam 모멘텀 유지 및 `weight_divergence = 0.0` 고정** | 연합학습은 라운드마다 로컬 옵티마이저를 새로 생성함(주의: FLamby 공식 FedAvg는 클라이언트 옵티마이저를 라운드 간 유지하므로 이 점은 공식 구현과 다름. 보고서에 "FLamby와 달리 라운드마다 Adam 상태를 초기화"라고 명시할 것). 상한 기준선인 `Centralized`는 루프 외부의 `central_optimizer`로 에폭 간 모멘텀을 유지함. 또한 `Centralized`의 `weight_divergence`는 `0.0`으로 고정해 Client Drift 그래프 왜곡 방지. |
| **8.5** | `main.py` | **`--resume` 이어서 학습 및 원자적(Atomic) 체크포인트 저장** | 코랩 세션 중단 시 마지막 완료 라운드 다음부터 즉시 재개할 수 있도록 모델·옵티마이저·난수 상태와 클라이언트별 DataLoader 반복자 위치(에폭 내 소비 배치 수와 에폭 시작 시점 generator 상태)를 저장하며, 저장 도중 연결이 끊겨 파일이 깨지지 않도록 `.tmp` 작성 후 `os.replace`로 교체함. 재개 시 반복자 위치를 복원해 에폭 중간에서 끊긴 라운드도 이어가지만, GPU 비결정 연산 때문에 비트 단위 동일성까지는 보장하지 않음. 세션별 라이브러리·GPU·커밋 정보는 `*_env.json`에 누적 기록함. |
| **8.6** | `fed_isic2019_colab.ipynb` | **Google Drive `tar` 캐싱 + 로컬 디스크(`/content`) 고속 I/O** | 23,247장 전처리 이미지를 최초 1회만 생성해 Drive에 단일 `tar`로 보관하고, 매 세션 로컬 디스크로 압축 해제하여 Google Drive FUSE 마운트의 I/O 병목을 해결함. |


---

## 9. 변경 이력 (Changelog)

### 2026-10-31 — 코드 리뷰 반영 (브랜치 `fix/notebook-source-resume-state`)

| 번호 | 대상 파일 | 변경 내용 | 변경 이유 |
| :---: | :--- | :--- | :--- |
| **9.1** | `fed_isic2019_colab.ipynb` (셀 1-1, 1-2, Phase 0) | `REPO_URL`을 이 저장소(`NOWEAZ/fedisic-federated-framework`), `REPO_BRANCH`를 `main`으로 변경. FLamby를 커밋 `edacf54`로 고정(`git checkout`). 결과 저장 폴더를 `CDII_results_6client`로 분리. | 기존 셀이 옛 `Capstone-Design-II`의 3클라이언트 브랜치(`fix/validation-split-and-metrics`)를 clone해 `--num_updates`, `get_nb_max_rounds`가 없어 ImportError 또는 옛 설정으로 학습됨. FLamby HEAD 고정이 없으면 `BaselineLoss`의 alpha와 `get_nb_max_rounds`가 바뀔 때 결과를 재현할 수 없음. 3클라이언트 결과(`CDII_results`)와 섞이지 않게 분리. |
| **9.2** | `federated_core.py` (`ClientStepIterator`) | `state_dict()` / `load_state_dict()` 추가. 현재 에폭 시작 시점의 DataLoader generator 상태와 소비한 배치 수를 저장하고, 재개 시 같은 순열을 다시 만든 뒤 소비한 배치만큼 건너뜀. | 라운드(100스텝)는 대부분 에폭 중간에서 끝나는데, 기존 `--resume`은 반복자 위치를 저장하지 않아 재개하면 다른 순열의 처음부터 시작해 끊기지 않은 실행과 배치 순서·증강 난수가 달라졌음. |
| **9.3** | `main.py` (`last_*.pt`, resume) | 체크포인트에 `client_iterator_states` 저장, 재개 시 복원. 구버전 체크포인트에는 경고를 출력. | 9.2를 실제 재개 경로에 연결. |
| **9.4** | `main.py` (`_run_final_test`) | 기존 `*_final_test.json`이 있을 때 저장된 `config`가 현재 `run_config`와 다르면 오류를 내고 중단. | 결과 파일명은 (method, mu, seed)만 반영하므로, 설정을 바꿔 다시 학습해도 이전 모델의 test 결과가 조용히 재사용될 수 있었음. |
| **9.5** | `main.py` (`_collect_env_info`) | 실행·재개 세션마다 `{exp_tag}_seed{seed}_env.json`에 Python, torch, CUDA, GPU, 주요 라이브러리 버전, 프로젝트와 FLamby 커밋을 누적 기록. | 라이브러리 버전이 고정되어 있지 않아 나중에 결과를 재현·점검할 기록이 필요. 중간에 GPU나 환경이 바뀐 재개 실행도 식별할 수 있음. |
| **9.6** | `tests/test_step_iterator_resume.py` (신규) | 합성 데이터(CPU)로 재개한 배치 순서가 끊기지 않은 실행과 같은지(워커 0개와 2개) 검증. 복원 없이 generator만 되돌리면 달라진다는 점도 함께 확인. | 9.2의 동작을 자동으로 검증. |
| **9.7** | `PROJECT_OVERVIEW.md`, 본 문서 | 센터별 학습/검증 장수를 실제 분할 값으로 정정(예: Client 5 학습 313 / 검증 38). 8.4의 Adam 서술에 "FLamby 공식은 클라이언트 옵티마이저를 라운드 간 유지하므로 이 코드와 다름"을 명시. 8.5 서술을 "에폭 중간 위치까지 복원하는 재개"로 수정하고 비트 단위 동일성은 보장하지 않음을 명시. | 문서의 "약 N장" 수치가 실제 분할(클래스당 최소 1장 규칙)과 달랐고, "무손실 재개"와 "FLamby 공식과 동일" 표현이 구현과 맞지 않았음. |

### 알려진 한계·미반영 항목
- 클라이언트 Adam을 라운드마다 새로 만드는 동작은 유지함(FLamby 공식과 다름). 유지 옵션(`--keep_optimizer`)은 아직 구현하지 않음.
- validation 없이 train 100%로 학습하는 모드(공식 학습량과 동일)는 아직 구현하지 않음.
- `requirements.txt`의 torch, scikit-learn, pandas 등은 미고정. `*_env.json`에 기록된 버전을 확인해 고정할 예정.
- 병원별(특히 Client 4, 5) 지표는 표본이 매우 작아 잡음이 큼. 부트스트랩 신뢰구간이나 큰 병원 기준 값 병기를 권장.
- GPU 비결정 연산(예: AdaptiveAvgPool 역전파) 때문에 같은 시드라도 비트 단위로 같은 결과는 보장되지 않음.
