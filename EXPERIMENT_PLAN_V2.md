# 확장 실험 계획 (2026-11-01, 아직 실행 전)

6센터 Kaggle 결과(`results/fed_isic_6client_kaggle_1101/`) 분석에서 나온 제안을 실험으로 옮긴 계획입니다. **모두 가설이며, 실행해 보기 전에는 효과를 알 수 없습니다.** 기존 9라운드 결과는 기준선으로 그대로 둡니다.

## 공통 규칙
- 시드: 본실험 `42, 100, 2026` + 추가 `7, 1234` (5개). 탐색 `41`, μ 재확인 `40`.
- test는 `--final_test`로 학습이 끝난 뒤 1회만 엽니다. μ 선택에는 test를 쓰지 않습니다.
- 서로 다른 설정은 `--save_dir`을 나누거나 `--tag`/자동 꼬리표(`_keepopt`)로 파일 이름을 구분합니다.
- 비용은 지난 실행(T4)의 실측에서 추정: 연합 1회(9라운드) 약 31분, Centralized 1회 약 30분. 18라운드는 약 62분.

## 실험 목록 (우선순위순)

| # | 실험 | 실행 횟수 | 대략 시간 | 확인하려는 것 |
|---|---|---|---|---|
| E0 | 5시드 완성: fedavg·fedprox(μ=0.1)·centralized를 시드 7, 1234 추가 | 6회 | 약 3.1h | 지금 3시드로는 FedProx 우위를 단정 못 함 |
| E1 | μ 재확인: 시드 41에 μ=0.03, 0.3 추가 / 시드 40에 fedavg와 μ=0.03, 0.1, 0.3 | 6회 | 약 3.1h | μ 선정이 시드 한 번·차이가 작은 상태였음 |
| E2 | `--keep_optimizer` fedavg·fedprox, 5시드 | 10회 | 약 5.2h | FLamby 공식과 같은 Adam 유지의 영향 |
| E3 | FedBN, 5시드 (9라운드) | 5회 | 약 2.6h | 센터별 BN이 센터 4 같은 소규모 센터에 도움이 되는지 |
| E4 | 확장 18라운드 fedavg·fedprox, 3시드 (`--rounds 18 --tag r18`) | 6회 | 약 6.2h | 9라운드에서 미수렴이었던 것이 더 돌리면 개선되는지 |
| | 합계 | 33회 | 약 20.2h | |

E1을 먼저 하면 E2~E4에 쓸 μ를 확정할 수 있습니다. 쿼터가 부족하면 E1 → E3 → E0 → E2 → E4 순으로 줄이는 것을 권장합니다.

## 명령 예시 (`DATA` = 전처리된 데이터 경로)
```bash
# E1: μ 재확인 (Stage 1과 같은 설정, 검증만 봄)
python main.py --method fedprox --mu 0.03 --seed 40 --data_path $DATA --save_dir ./results_mu --resume
python main.py --method fedprox --mu 0.3  --seed 41 --data_path $DATA --save_dir ./results_mu --resume
# E2: Adam 유지
python main.py --method fedprox --mu $BEST_MU --keep_optimizer --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
# E3: FedBN
python main.py --method fedbn --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
# E4: 확장 라운드
python main.py --method fedavg --rounds 18 --tag r18 --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
```
μ 선택 규칙은 기존과 같습니다(마지막 3라운드 val macro-F1 평균 최대, 동률이면 더 작은 μ). 시드 40과 41에서 후보별 값을 평균해 정합니다.

## 분석
```bash
python analysis/analyze_extended.py --results_dir <결과폴더> --pairs fedprox_mu0.1:fedavg fedbn:fedavg fedavg_keepopt:fedavg
```
출력(`analysis_extended/`): `per_class_recall.csv`, `confusion_<방법>.csv`, `bootstrap_ci_overall.csv`, `bootstrap_ci_per_center.csv`, `paired_by_seed.csv`, `paired_summary.csv`, `paired_per_center.csv`.

## 해석할 때 주의
- FedBN은 센터마다 다른 모델(글로벌 가중치 + 센터 BN)로 평가하므로 "서버 모델 하나"인 FedAvg/FedProx와 평가 대상이 다릅니다. 표에서 구분하세요.
- 부트스트랩 구간은 test 표본의 불확실성만 반영합니다. 학습 난수에 따른 변동은 시드별 값(`paired_by_seed.csv`)으로 같이 보세요.
- 확장 18라운드는 학습량이 늘어난 설정이므로 Centralized(20에폭)와의 비교 해석이 달라집니다. 공식 9라운드와 구분해 표기하세요.
- 센터 4·5는 test가 164·88장이라 센터별 순위는 잡음이 큽니다.
