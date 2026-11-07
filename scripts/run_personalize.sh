#!/usr/bin/env bash
# 학습이 끝난 FedProx 글로벌 모델(last_*.pt)을 병원별로 미세조정해 평가한다 (연합학습은 다시 돌리지 않음).
#
# 사용법: bash scripts/run_personalize.sh <data_path> <ckpt_dir>
#   <ckpt_dir>: scripts/run_fedprox_r18_cosine.sh 의 save_dir (last_*.pt가 있는 폴더)
# 환경변수: TAG(기본 fedprox_mu0.1_cosine), SEEDS(기본 "42 100 2026"), STEPS(기본 50,100,200)
set -euo pipefail

DATA_PATH="${1:?data_path 필요}"
CKPT_DIR="${2:?ckpt_dir 필요}"
TAG="${TAG:-fedprox_mu0.1_cosine}"
SEEDS="${SEEDS:-42 100 2026}"
STEPS="${STEPS:-50,100,200}"

for s in $SEEDS; do
  python personalize.py \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --tag "$TAG" \
    --seed "$s" \
    --steps "$STEPS"
done
