"""
병원별 개인화(로컬 미세조정) 실험에서 쓰는 순수 numpy 보조 함수.

torch 없이 단위 테스트할 수 있도록 분리했다. 병원이 자기 *학습* 라벨 분포만 사용한다
(다른 병원의 데이터나 test 라벨은 쓰지 않는다).
"""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 8
# adabn: 파라미터 학습 없이 병원 학습 이미지로 BatchNorm 통계만 다시 계산 (AdaBN)
SCOPES = ("adabn", "head", "bn_head", "all")
LOSS_KINDS = ("global", "local")


def class_counts(labels, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """클래스별 샘플 수 (길이 num_classes)."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.bincount(labels, minlength=num_classes).astype(np.int64)


def present_mask(counts) -> np.ndarray:
    """병원 학습 라벨에 한 장이라도 있는 클래스 True."""
    return np.asarray(counts) > 0


def local_class_alpha(counts, max_alpha: float = 20.0) -> np.ndarray:
    """
    병원 로컬 분포 기준 역빈도 가중치. alpha_c = N / (K * n_c).
    Σ_c (n_c / N) * alpha_c = 1 이므로 평균 손실 크기가 가중하지 않은 손실과 같다.
    없는 클래스는 0, 너무 큰 값은 max_alpha로 자른다.
    """
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    k = int(present.sum())
    if k == 0:
        raise ValueError("학습 라벨이 비어 있습니다.")
    total = counts.sum()
    alpha = np.zeros_like(counts)
    alpha[present] = total / (k * counts[present])
    return np.minimum(alpha, max_alpha)


def mask_probs(probs, present) -> np.ndarray:
    """없는 클래스의 확률을 0으로 만들고 행을 다시 정규화한다."""
    probs = np.asarray(probs, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    masked = probs * present[None, :]
    denom = masked.sum(axis=1, keepdims=True)
    # 전부 0이 되는 극단적 경우(없는 클래스에만 확률이 몰림)는 균등 분포로 대체
    uniform = present.astype(np.float64) / present.sum()
    masked = np.where(denom > 0, masked / np.maximum(denom, 1e-12), uniform[None, :])
    return masked


def balanced_accuracy(y_true, y_pred) -> float:
    """정답에 존재하는 클래스 recall의 평균 (sklearn balanced_accuracy_score와 같은 정의)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    recalls = []
    for c in np.unique(y_true):
        sel = y_true == c
        recalls.append(float(np.mean(y_pred[sel] == c)))
    return float(np.mean(recalls)) if recalls else float("nan")


def select_config(val_scores: dict):
    """
    val_scores: {(scope, loss, steps): [병원별 val balanced acc 6개]}.
    병원 평균 val balanced acc가 가장 높은 설정을 고른다. 동률이면 스텝이 적고 학습 범위가 작은 쪽
    (head < bn_head < all, global < local)을 택한다. 병원별 val이 너무 작아(병원 4는 65장) 병원마다
    따로 고르지 않고 모든 병원에 공통 설정 하나를 쓴다.
    """
    if not val_scores:
        raise ValueError("val_scores가 비어 있습니다.")

    def key(item):
        (scope, loss, steps), scores = item
        mean = float(np.mean(scores))
        return (-round(mean, 6), steps, SCOPES.index(scope), LOSS_KINDS.index(loss))

    best_key, best_scores = sorted(val_scores.items(), key=key)[0]
    return best_key, float(np.mean(best_scores))
