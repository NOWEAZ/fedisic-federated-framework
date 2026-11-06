# 성능 향상 계획 (main 기준)

기준 코드: `main` 2c1d851 (2026-11-03). 한 번에 요인 하나만 바꿔 효과를 분리한다.

## 1. 베이스라인 (main이 하는 일)
- FLamby Fed-ISIC2019 공식 6센터, EfficientNet-B0(BN 유지), 가중 Focal Loss(전역 alpha)
- FedAvg/FedProx: 라운드당 클라이언트 100스텝 × 9라운드, Adam lr 5e-4 고정, 배치 64, **클라이언트 Adam 상태 라운드 간 유지**
- 서버: 샘플 수 비례 가중 평균 (Center 0이 약 53%)
- Centralized: 20에폭. 검증 90/10 층화 분할, 테스트는 `--final_test`로 1회
- 시드: 탐색 41, 본실험 42/100/2026

**main의 결과 (8af4275, `results/`, 시드 42/100/2026 test 1회 평가, 9라운드)**

| 방법 | balanced acc | macro-F1 | 병원 간 bal-acc std | ECE |
|---|---|---|---|---|
| Centralized (20에폭) | 0.7293 ± 0.0180 | 0.6719 | 0.0668 | 0.0284 |
| FedAvg | 0.6768 ± 0.0085 | 0.5826 | 0.1274 | 0.1008 |
| FedProx (μ=0.1) | 0.6561 ± 0.0085 | 0.5676 | 0.1172 | 0.0899 |

FedProx는 전체 balanced acc가 FedAvg보다 낮고(−2.1%p), 병원 간 편차·ECE·드리프트만 좋다. μ는 시드 41 val 마지막 3라운드 평균으로 0.1을 골랐다(0.1: 0.663, 1.0: 0.618).

**알려진 한계 / 주의**
- 이전 Kaggle 결과(Centralized 0.739, FedAvg 0.633, FedProx 0.670)는
  클라이언트 Adam을 라운드마다 초기화하던 브랜치(`fix/notebook-source-resume-state`)의 값이라 **main의 베이스라인이 아니다**.
- 에폭 중간 `ClientStepIterator` 위치는 체크포인트에 저장되지 않아 `--resume`이 비트 단위로 동일하지 않다.
- 센터 4·5의 val/test 표본이 매우 작아(val 65/35장, test 164/88장) 병원별 지표는 잡음이 크다.
- 학습 길이가 짧다(9라운드). 이전 브랜치 실험에서는 9→18라운드에서 FedAvg·FedProx 모두 올랐다(예산 효과이며 알고리즘 개선은 아님).

## 2. 개선 후보 (우선순위)
| 순위 | 후보 | 바꾸는 요인 | 비고 |
|---|---|---|---|
| 1 | 라운드별 학습률 감쇠(cosine) | lr 스케줄만 | 비용·통신 추가 없음, 전역 모델 평가라 main과 직접 비교 가능. **이 브랜치에서 구현** |
| 2 | FedBN (BN 통계·감마/베타 로컬 유지) | 집계 대상 | 이전 실험에서 가장 큰 향상이었으나 센터별 모델 평가라 비교 대상이 달라짐 |
| 3 | 서버 가중치 완화(균등/√N) 또는 서버 모멘텀(FedAvgM) | 집계 규칙 | 소형 센터 반영↑, 대형 센터 성능↓ 가능 |
| 4 | 증강 강화(색·밝기 등) | 로컬 학습 데이터 | 장비별 Feature shift 완화, 학습 비용↑ |
| 5 | 학습 예산 증가(18라운드) | `--rounds` | 가장 확실하나 알고리즘 개선이 아니고 Centralized 예산과 어긋남 |

## 3. 1순위 구현: `--lr_schedule cosine`
- `--lr_schedule {constant,cosine}` (기본 constant = main과 동일), `--lr_min_ratio` (기본 0)
- 라운드 r의 lr = lr_min + 0.5·(lr − lr_min)·(1 + cos(π·(r−1)/R)). 1라운드 = 5e-4, 마지막 라운드 ≈ 3%.
- Adam 모멘텀은 그대로 두고 `param_group["lr"]`만 매 라운드 갱신. Centralized는 에폭 단위로 동일 적용.
- cosine이면 결과 파일 태그에 `_cosine`이 붙어 베이스라인과 섞이지 않음. CSV에 `lr` 열 추가.
- resume: 구 체크포인트(키 없음)는 constant로 간주. 스케줄이 다르면 오류.

## 4. 검증 실험 (GPU 필요, 아직 미실행)
같은 시드(42, 100, 2026)·9라운드로 두 번만 실행하고 test balanced acc를 비교한다.
```bash
python main.py --method fedavg --seed 42 --data_path ./isic_data --save_dir ./results/base   --resume --final_test
python main.py --method fedavg --seed 42 --lr_schedule cosine --data_path ./isic_data --save_dir ./results/cosine --resume --final_test
```
FedProx는 베이스라인 FedAvg 결과로 효과가 확인된 뒤에 같은 방식으로 추가한다.
