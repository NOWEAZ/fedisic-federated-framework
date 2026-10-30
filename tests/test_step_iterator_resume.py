"""ClientStepIterator의 state_dict/load_state_dict가 에폭 중간 재개를 정확히 복원하는지 확인한다.

CPU와 합성 데이터만 쓴다 (FLamby 데이터·GPU 불필요). 실행: python -m pytest tests -q  또는  python tests/test_step_iterator_resume.py
"""
import os
import random
import sys
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# federated_core가 import하는 외부 패키지 중 없는 것은 가짜 모듈로 대체 (이 테스트는 모델·손실·지표를 쓰지 않는다)
for _name in ("torchvision", "sklearn", "sklearn.metrics", "tqdm"):
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        _m.models = None
        _m.balanced_accuracy_score = _m.f1_score = _m.tqdm = None
        sys.modules[_name] = _m
if "flamby" not in sys.modules:
    fake = types.ModuleType("flamby.datasets.fed_isic2019")
    fake.BaselineLoss = object
    sys.modules["flamby"] = types.ModuleType("flamby")
    sys.modules["flamby.datasets"] = types.ModuleType("flamby.datasets")
    sys.modules["flamby.datasets.fed_isic2019"] = fake

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_core import ClientStepIterator  # noqa: E402


class NoisyDataset(Dataset):
    """증강처럼 워커의 random/numpy 난수를 쓰는 데이터셋."""

    def __len__(self):
        return 300

    def __getitem__(self, i):
        return torch.tensor([float(i), random.random() + np.random.rand()])


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def _make_loader(num_workers):
    gen = torch.Generator()
    gen.manual_seed(42)
    return DataLoader(
        NoisyDataset(), batch_size=64, shuffle=True, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=gen,
    )


def _run(it, rounds, steps, with_noise):
    """with_noise=False면 샘플 인덱스만 비교한다 (num_workers=0에서는 증강 난수가 메인 프로세스의 전역 난수라
    main.py의 _restore_rng_states가 따로 복원하는 영역이므로 이 테스트 범위 밖)."""
    out = []
    for _ in range(rounds):
        for _ in range(steps):
            b = it.get_next_batch().clone().numpy().round(6)
            out.append((b if with_noise else b[:, 0]).tolist())
    return out


def _check(num_workers):
    wn = num_workers > 0
    steps = 3  # 300/64 = 5배치/에폭이라 라운드 경계가 에폭 중간에 걸린다
    full = _run(ClientStepIterator(_make_loader(num_workers)), 4, steps, wn)

    loader = _make_loader(num_workers)
    it = ClientStepIterator(loader)
    first = _run(it, 2, steps, wn)
    saved_iter, saved_gen = it.state_dict(), loader.generator.get_state()

    loader2 = _make_loader(num_workers)
    loader2.generator.set_state(saved_gen)  # main.py의 _restore_rng_states와 같은 순서
    it2 = ClientStepIterator(loader2)
    it2.load_state_dict(saved_iter)
    assert torch.equal(loader2.generator.get_state(), saved_gen), "복원 후 generator 상태가 저장 시점과 달라짐"
    second = _run(it2, 2, steps, wn)

    assert first + second == full, "재개한 배치 순서/증강 난수가 끊기지 않은 실행과 다름"


def test_resume_matches_uninterrupted_single_process():
    _check(0)


def test_resume_matches_uninterrupted_with_workers():
    _check(2)


def test_naive_resume_would_differ():
    """복원 로직 없이 generator만 되돌리면 달라진다는 것(이 수정이 필요한 이유)을 확인한다."""
    steps = 3
    full = _run(ClientStepIterator(_make_loader(0)), 4, steps, False)
    loader = _make_loader(0)
    it = ClientStepIterator(loader)
    first = _run(it, 2, steps, False)
    gen_state = loader.generator.get_state()
    loader2 = _make_loader(0)
    loader2.generator.set_state(gen_state)
    second = _run(ClientStepIterator(loader2), 2, steps, False)
    assert first + second != full


if __name__ == "__main__":
    test_resume_matches_uninterrupted_single_process()
    test_resume_matches_uninterrupted_with_workers()
    test_naive_resume_would_differ()
    print("ok")
