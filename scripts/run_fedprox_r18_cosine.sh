#!/usr/bin/env bash
# FedProx만 18라운드 + cosine lr로 3시드 실행 (main 베이스라인: FedProx μ=0.1, 9라운드, constant lr).
# 사용법: bash scripts/run_fedprox_r18_cosine.sh <data_path> [save_dir]
#   환경변수: MU(기본 0.1) ROUNDS(기본 18) LR_SCHEDULE(기본 cosine) SEEDS(기본 "42 100 2026")
# 결과 파일 이름에 라운드 수가 들어가지 않으므로 실험 조건마다 save_dir를 다르게 쓸 것.
# 끊겨도 같은 명령을 다시 실행하면 --resume으로 이어서 진행한다.
set -euo pipefail
DATA_PATH="${1:?data_path 필요}"
MU="${MU:-0.1}"
ROUNDS="${ROUNDS:-18}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
SEEDS="${SEEDS:-42 100 2026}"
SAVE_DIR="${2:-./results_fedprox_r${ROUNDS}_${LR_SCHEDULE}}"

for s in $SEEDS; do
  echo "=== FedProx mu=${MU} rounds=${ROUNDS} lr_schedule=${LR_SCHEDULE} seed=${s} ==="
  python main.py --method fedprox --mu "$MU" --rounds "$ROUNDS" --num_updates 100 \
    --lr_schedule "$LR_SCHEDULE" --seed "$s" \
    --data_path "$DATA_PATH" --save_dir "$SAVE_DIR" --resume --final_test
done
