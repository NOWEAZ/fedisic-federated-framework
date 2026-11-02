# 확장 실험 계획 (2026-11-01 작성, 2026-11-02 확장 실험 A·B 실행: 아래 '실행 기록' 참고)

6센터 Kaggle 결과(`results/fed_isic_6client_kaggle_1101/`) 분석에서 나온 제안을 실험으로 옮긴 계획입니다. **모두 가설이며, 실행해 보기 전에는 효과를 알 수 없습니다.** 기존 9라운드 결과는 기준선으로 그대로 둡니다.

## 공통 규칙
- 시드: 본실험 `42, 100, 2026` (3개). 탐색 `41`, μ 재확인 `40`. 처음 계획에는 추가 시드 `7, 1234`(5개)가 있었으나 실제 실행에는 쓰지 않았습니다(실행 기록은 아래 참고).
- test는 `--final_test`로 학습이 끝난 뒤 1회만 엽니다. μ 선택에는 test를 쓰지 않습니다.
- 서로 다른 설정은 `--save_dir`을 나누거나 `--tag`/자동 꼬리표(`_keepopt`)로 파일 이름을 구분합니다.
- 비용은 지난 실행(T4)의 실측에서 추정: 연합 1회(9라운드) 약 31분, Centralized 1회 약 30분. 18라운드는 약 62분.

## 실험 목록 (우선순위순)

| # | 실험 | 실행 횟수 | 대략 시간 | 확인하려는 것 |
|---|---|---|---|---|
| E0 | (미실행) 5시드 완성: fedavg·fedprox(μ=0.1)·centralized를 시드 7, 1234 추가 | 6회 | 약 3.1h | 지금 3시드로는 FedProx 우위를 단정 못 함 |
| E1 | μ 재확인: 시드 41에 μ=0.03, 0.3 추가 / 시드 40에 fedavg와 μ=0.03, 0.1, 0.3 | 6회 | 약 3.1h | μ 선정이 시드 한 번·차이가 작은 상태였음 |
| E2 | `--keep_optimizer` fedavg·fedprox, 3시드 (expB에서 cosine·에폭 기준과 함께 실행) | 6회 | 약 5.2h | FLamby 공식과 같은 Adam 유지의 영향 |
| E3 | FedBN, 3시드 (확장 실험 A에서 9라운드, expB에서 18라운드) | 3회 | 약 2.6h | 센터별 BN이 센터 4 같은 소규모 센터에 도움이 되는지 |
| E4 | 확장 18라운드 fedavg·fedprox, 3시드 (`--rounds 18 --tag r18`) | 6회 | 약 6.2h | 9라운드에서 미수렴이었던 것이 더 돌리면 개선되는지 |
| E5 | 과소적합·과적합 대응 조합 (선택): 18라운드 + `--lr_schedule cosine` (+선택 `--server_momentum 0.5`) fedavg·fedprox, 3시드 / Centralized `--weight_decay 0.01` 3시드 | 12회 | 약 8.5h | 학습량 증가만으로 부족할 때 후반 정체·서버 평균 속도를 개선하는지 / Centralized 약한 과적합 완화 |
| | 합계 (E5 제외, 처음 계획 기준) | 33회 | 약 20.2h | |

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
# E5: 과소적합 대응 (확장 라운드 + 학습률 감소 [+ 서버 모멘텀])
python main.py --method fedprox --mu $BEST_MU --rounds 18 --lr_schedule cosine --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
python main.py --method fedavg --rounds 18 --lr_schedule cosine --server_momentum 0.5 --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
# 과적합 대응 (Centralized): 가중치 감쇠
python main.py --method centralized --weight_decay 0.01 --seed 42 --data_path $DATA --save_dir ./results_v2 --resume --final_test
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

## 실행 기록 (2026-11-02)
- **확장 실험 A**(Kaggle 버전 3, 약 8시간 5분): E4(FedAvg·FedProx μ=0.1, 18라운드)와 E3(FedBN, 9라운드), 시드 42/100/2026. 결과: `results/fed_isic_6client_kaggle_1102_v2/`.
- **확장 실험 B**(Kaggle 버전 4, 약 5시간 58분, T4 x2에 작업 큐로 두 장 동시 실행): E1 μ 재확인 후 E2·E3·E5 일부(cosine)를 한 번에 합쳐 실행. 시드는 3개(42/100/2026)이며 7, 1234는 쓰지 않았습니다. 결과: `results/fed_isic_6client_kaggle_1102_v2b/`.
  - E1: 시드 40·41 평균(마지막 3라운드 val macro-F1)으로 μ=0.3이 선정됨(0.3: 0.5275, 0.1: 0.5252, 0.03: 0.5011; 0.3과 0.1의 차이는 0.0023로 작음).
  - 본 실험: `--keep_optimizer --lr_schedule cosine --num_updates 0 --local_epochs 2 --rounds 18`로 FedAvg·FedProx(μ=0.3)·FedBN. 균형정확도 3시드 평균은 FedAvg 0.7300, FedProx 0.7225, FedBN 0.7674(Centralized 0.7391). FedBN−FedAvg +0.0374, FedBN−FedProx +0.0449(둘 다 3시드 모두 양수), FedProx−FedAvg −0.0076(구간이 0 포함).
  - 한계: keep_optimizer·cosine·에폭 기준·18라운드를 한꺼번에 바꿔서 어느 요인이 얼마나 기여했는지는 분리되지 않음, FedBN은 센터별 BN 모델로 평가(평가 대상이 다름), 시드 3개, 센터 5의 test는 88장.
