"""라운드(에폭) 단위 학습률 스케줄. torch/FLamby에 의존하지 않아 CPU 단위 테스트가 가능합니다."""
import math

LR_SCHEDULES = ("constant", "cosine")


def round_lr(
    base_lr: float,
    round_idx: int,
    total_rounds: int,
    schedule: str = "constant",
    min_ratio: float = 0.0,
) -> float:
    """
    round_idx(1부터 시작)번째 라운드에 쓸 학습률을 반환합니다.

    - constant: 항상 base_lr (FLamby 공식 설정, main의 기존 동작과 동일)
    - cosine:   1라운드는 base_lr, 이후 코사인으로 감소해 base_lr * min_ratio에 가까워집니다.
                lr_r = lr_min + 0.5 * (base_lr - lr_min) * (1 + cos(pi * (r - 1) / R)),  lr_min = base_lr * min_ratio
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"알 수 없는 lr_schedule: {schedule} (가능: {LR_SCHEDULES})")
    if schedule == "constant":
        return float(base_lr)
    if total_rounds < 1 or not 1 <= round_idx <= total_rounds:
        raise ValueError(f"round_idx={round_idx}가 1..{total_rounds} 범위를 벗어났습니다.")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_ratio는 0과 1 사이여야 합니다: {min_ratio}")
    lr_min = base_lr * min_ratio
    progress = (round_idx - 1) / total_rounds
    return float(lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress)))
