# Fed-ISIC2019 6-client — Kaggle 확장 실험 A 결과 (2026-11-02)

Kaggle 노트북 버전 3("expA 18 rounds + FedBN", Save & Run All, GPU T4 x2, 29,136초 ≈ 8시간 5분)의 결과입니다.
코드: `main.py` 등은 브랜치 `feature/v2-extended-experiments`, 노트북은 브랜치 `feature/v2-expB-notebooks`의 `kaggle/fed_isic2019_kaggle_v2_expA.ipynb`.
이전 9라운드 결과(Centralized, FedAvg, FedProx μ=0.1; 시드 42/100/2026)는 브랜치 `results/fed-isic-6client-1101`에서 읽었습니다.

## 무엇을 돌렸나

| 실험 | 방법 | 라운드 | 시드 | 비고 |
| :--- | :--- | :---: | :--- | :--- |
| E4 (학습량 늘리기) | FedAvg | 18 (공식: 9) | 42, 100, 2026 | `--tag r18`, `--final_test` |
| E4 (학습량 늘리기) | FedProx μ=0.1 | 18 (공식: 9) | 42, 100, 2026 | `--tag r18`, `--final_test` |
| E3 | FedBN (BatchNorm을 센터별로 유지) | 9 | 42, 100, 2026 | `--final_test` |

나머지 설정은 기준선과 같습니다(lr 5e-4 고정, Adam, 배치 64, 라운드당 로컬 100 업데이트, val_fraction 0.1, split_seed 0).
test 셋은 실행마다 한 번만 열었습니다(`--final_test`, 마지막 라운드 모델).

## 최종 test 결과 (시드별)

Bal Acc = 전체 test 표본의 balanced accuracy. Gap / Std = 6개 센터 사이 balanced accuracy의 최댓값−최솟값, 표준편차.

| 방법 | 시드 | Bal Acc | Macro F1 | ECE | Center Gap | Center Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FedAvg, 18라운드 | 42 | 0.6752 | 0.5386 | 0.0731 | 0.2953 | 0.1109 |
| FedAvg, 18라운드 | 100 | 0.6803 | 0.5766 | 0.0796 | 0.4245 | 0.1490 |
| FedAvg, 18라운드 | 2026 | 0.6777 | 0.5915 | 0.0570 | 0.3790 | 0.1518 |
| FedProx μ=0.1, 18라운드 | 42 | 0.7030 | 0.5988 | 0.0762 | 0.3190 | 0.1282 |
| FedProx μ=0.1, 18라운드 | 100 | 0.7122 | 0.5875 | 0.0599 | 0.3958 | 0.1492 |
| FedProx μ=0.1, 18라운드 | 2026 | 0.6929 | 0.5804 | 0.0609 | 0.3548 | 0.1290 |
| FedBN, 9라운드 | 42 | 0.6742 | 0.5560 | 0.0885 | 0.2502 | 0.1039 |
| FedBN, 9라운드 | 100 | 0.6455 | 0.5256 | 0.0628 | 0.1687 | 0.0753 |
| FedBN, 9라운드 | 2026 | 0.6105 | 0.4772 | 0.0515 | 0.3297 | 0.1214 |

## 3시드 평균과 95% 부트스트랩 구간 (`analysis_extended/bootstrap_ci_overall.csv`)

| 방법 | Bal Acc (95% 구간) | Macro F1 (95% 구간) |
| :--- | :---: | :---: |
| Centralized (9라운드 시기 기준선) | 0.7391 (0.7205–0.7579) | 0.6824 (0.6641–0.6986) |
| FedAvg, 9라운드 | 0.6331 (0.6165–0.6482) | 0.5078 (0.4886–0.5259) |
| FedAvg, 18라운드 | 0.6778 (0.6616–0.6931) | 0.5689 (0.5496–0.5866) |
| FedProx μ=0.1, 9라운드 | 0.6695 (0.6522–0.6867) | 0.5303 (0.5115–0.5491) |
| FedProx μ=0.1, 18라운드 | 0.7027 (0.6853–0.7195) | 0.5889 (0.5690–0.6074) |
| FedBN, 9라운드 | 0.6434 (0.6267–0.6609) | 0.5196 (0.5014–0.5374) |

구간은 test 표본의 불확실성만 반영합니다(센터 안에서 복원추출). 시드 사이 변동은 `analysis_extended/paired_by_seed.csv`에 따로 있습니다.

## 같은 시드끼리 짝지은 차이 (`analysis_extended/paired_summary.csv`, Bal Acc)

| 비교 | 평균 차이 | 차이가 양수인 시드 | 부트스트랩 구간이 0을 제외 |
| :--- | :---: | :---: | :---: |
| FedAvg 18r − FedAvg 9r | +0.0447 | 3/3 | 예 |
| FedProx 18r − FedProx 9r | +0.0332 | 3/3 | 예 |
| FedProx 18r − FedAvg 18r | +0.0249 | 3/3 | 예 |
| FedBN − FedAvg (9r) | +0.0103 | 2/3 | 아니오 |
| FedBN − FedProx μ=0.1 (9r) | −0.0261 | 1/3 | 예 (0보다 낮음) |

## 한계

- FedBN은 센터마다 자기 BatchNorm 상태("전역 가중치 + 센터 BN")로 평가하고, FedAvg/FedProx는 서버 모델 하나로 평가합니다. 같은 종류의 모델이 아니므로 그 점을 염두에 두고 비교해야 합니다.
- 18라운드 실행은 공식 9라운드의 두 배 학습량이라, 공식 학습량 기준의 Centralized(20에폭)와 바로 비교할 수 없습니다.
- 방법마다 시드가 3개뿐입니다. 라운드를 늘린 효과는 3시드 모두 양수로 일관되지만, FedBN 비교는 덜 분명합니다(FedBN 시드 2026이 나머지 두 시드보다 많이 낮음).
- 이 폴더에 없는 것: 모델 체크포인트(`*.pt`), 실행별 환경 JSON, 원본 tar(`results_v2.tar.gz`, 285.6 MB, Kaggle 버전 3 Output에 있음).

## 파일

- `<방법>_seed<N>_final_test.json` — 실행별 최종 test 지표와 설정
- `<방법>_seed<N>_final_test_preds.npz` — `analysis/analyze_extended.py`에서 쓰는 test 예측
- `<방법>_seed<N>_metrics.csv` — 라운드별 학습/검증 지표
- `analysis_extended/*.csv` — `analysis/analyze_extended.py`의 출력(부트스트랩 구간, 짝 비교, 혼동행렬, 클래스별 재현율)
