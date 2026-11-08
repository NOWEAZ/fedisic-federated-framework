#!/usr/bin/env bash
# 학습이 끝난 FedProx 글로벌 모델(last_*.pt)에 대해 개인화의 근접 항(mu)과 learning rate(lr)를 훑는다 (재학습 없음).
# 설정은 병원 공통으로 val에서 하나를 고르고, test는 raw / mask_only / personalized 1회씩만 연다.
# 결과는 <tag>_seed<S>_personalize_sweep.json / _preds.npz 로 따로 저장되어 기존 개인화 결과를 덮어쓰지 않는다.
#
# 사용법: bash scripts/run_personalize_sweep.sh <data_path> <ckpt_dir> [save_dir]
#   <ckpt_dir>: last_*.pt가 있는 폴더. save_dir를 생략하면 ckpt_dir에 저장한다.
# 환경변수: TAG(기본 fedprox_mu0.1_cosine) SEEDS(기본 "42 100 2026") STEPS(50,100,200) MUS(0.1,1.0) LRS(1e-4,1e-3)
#           SCOPES(head,bn_head,all) LOSSES(local)
set -euo pipefail

DATA_PATH="${1:?data_path 필요}"
CKPT_DIR="${2:?ckpt_dir 필요}"
SAVE_DIR="${3:-$CKPT_DIR}"
TAG="${TAG:-fedprox_mu0.1_cosine}"
SEEDS="${SEEDS:-42 100 2026}"
STEPS="${STEPS:-50,100,200}"
MUS="${MUS:-0.1,1.0}"
LRS="${LRS:-1e-4,1e-3}"
SCOPES="${SCOPES:-head,bn_head,all}"
LOSSES="${LOSSES:-local}"

for s in $SEEDS; do
  python personalize.py \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --save_dir "$SAVE_DIR" \
    --tag "$TAG" \
    --seed "$s" \
    --steps "$STEPS" \
    --scopes "$SCOPES" \
    --losses "$LOSSES" \
    --mus "$MUS" \
    --lrs "$LRS"
done
