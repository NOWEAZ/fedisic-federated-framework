"""main.py의 --keep_optimizer / fedbn / prox_loss 기록 / resume을 CPU·합성 데이터로 끝까지 돌려 확인한다.

FLamby 데이터·torchvision·GPU 없이 동작하도록 데이터 로더와 모델을 작은 가짜로 바꿔 끼운다.
실행: python -m pytest tests -q
"""
import csv
import importlib.machinery
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

for _name in ("torchvision", "tqdm"):
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, None)  # torch가 find_spec으로 조회해도 깨지지 않게
        _m.models = None
        _m.tqdm = lambda it=None, **kw: it
        sys.modules[_name] = _m
if "flamby" not in sys.modules:
    fake = types.ModuleType("flamby.datasets.fed_isic2019")
    fake.BaselineLoss = object
    sys.modules["flamby"] = types.ModuleType("flamby")
    sys.modules["flamby.datasets"] = types.ModuleType("flamby.datasets")
    sys.modules["flamby.datasets.fed_isic2019"] = fake

# data_utils는 FLamby 전체를 import하므로 이 테스트에서는 필요한 상수·함수만 가진 가짜로 대체한다
_du = types.ModuleType("data_utils")
_du.BATCH_SIZE, _du.LR, _du.NUM_CLIENTS, _du.NUM_EPOCHS_POOLED = 8, 5e-4, 6, 2
_du.get_nb_max_rounds = lambda num_updates, batch_size=8: 2
_du.get_dataloaders = None
sys.modules["data_utils"] = _du

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as main_mod  # noqa: E402

SIZES = [40, 24, 32, 16, 20, 28]


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(x.mean(dim=(2, 3)))


def _ds(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = torch.randint(0, 8, (n,), generator=g)
    return TensorDataset(x, y)


def _fake_get_dataloaders(batch_size=8, num_workers=0, data_path=None, seed=None, val_fraction=0.1, split_seed=0):
    def loader(ds, shuffle=True, gen_seed=0):
        gen = torch.Generator().manual_seed(gen_seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, drop_last=shuffle)

    clients = [loader(_ds(n, 10 + i), gen_seed=(seed or 0) + i) for i, n in enumerate(SIZES)]
    central = loader(_ds(sum(SIZES), 99), gen_seed=(seed or 0) + 100)
    val = DataLoader(_ds(30, 7), batch_size=batch_size)
    test = DataLoader(_ds(30, 8), batch_size=batch_size)
    val.client_eval_loaders = [DataLoader(_ds(10 + i, 20 + i), batch_size=batch_size) for i in range(6)]
    test.client_eval_loaders = [DataLoader(_ds(10 + i, 40 + i), batch_size=batch_size) for i in range(6)]
    test.client_test_loaders = test.client_eval_loaders
    return clients, central, val, test


def run_main(save_dir, *argv):
    main_mod.get_dataloaders = _fake_get_dataloaders
    main_mod.build_efficientnet_b0 = lambda num_classes=8, pretrained=True: TinyNet()
    main_mod.build_baseline_loss = lambda device="cpu": nn.CrossEntropyLoss().to(device)
    sys.argv = ["main.py", "--save_dir", str(save_dir), "--num_updates", "3", "--batch_size", "8",
                "--num_workers", "0", *argv]
    main_mod.main()


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_default_unchanged_and_prox_zero(tmp_path):
    run_main(tmp_path, "--method", "fedavg", "--rounds", "2", "--seed", "1")
    ck = _load(tmp_path / "last_fedavg_seed1.pt")
    assert ck["client_optimizer_states"] is None and ck["client_bn_states"] is None
    rows = _rows(tmp_path / "fedavg_seed1_metrics.csv")
    assert len(rows) == 2 and all(float(r["prox_loss"]) == 0.0 for r in rows)


def test_keep_optimizer_carries_adam_state_and_resume(tmp_path):
    run_main(tmp_path, "--method", "fedavg", "--keep_optimizer", "--rounds", "1", "--seed", "2")
    ck = _load(tmp_path / "last_fedavg_keepopt_seed2.pt")
    steps1 = [float(next(iter(s["state"].values()))["step"]) for s in ck["client_optimizer_states"]]
    assert steps1 == [3.0] * 6  # 라운드당 3스텝
    # 이어서 2라운드까지 → 같은 상태 위에 스텝이 누적되어야 한다 (매 라운드 초기화되면 3에 머문다)
    run_main(tmp_path, "--method", "fedavg", "--keep_optimizer", "--rounds", "2", "--seed", "2", "--resume")
    ck2 = _load(tmp_path / "last_fedavg_keepopt_seed2.pt")
    steps2 = [float(next(iter(s["state"].values()))["step"]) for s in ck2["client_optimizer_states"]]
    assert steps2 == [6.0] * 6
    assert len(_rows(tmp_path / "fedavg_keepopt_seed2_metrics.csv")) == 2


def test_keep_optimizer_uses_separate_filenames(tmp_path):
    run_main(tmp_path, "--method", "fedavg", "--rounds", "1", "--seed", "3")
    # 옵션만 바꿔도 파일 이름(_keepopt)이 달라 서로의 결과를 덮어쓰지 않는다
    run_main(tmp_path, "--method", "fedavg", "--keep_optimizer", "--rounds", "1", "--seed", "3")
    assert (tmp_path / "last_fedavg_seed3.pt").exists() and (tmp_path / "last_fedavg_keepopt_seed3.pt").exists()


def test_fedprox_logs_prox_loss(tmp_path):
    run_main(tmp_path, "--method", "fedprox", "--mu", "1.0", "--rounds", "2", "--seed", "4")
    rows = _rows(tmp_path / "fedprox_mu1_seed4_metrics.csv")
    assert float(rows[0]["prox_loss"]) >= 0.0 and float(rows[1]["prox_loss"]) > 0.0
    assert float(rows[1]["prox_dist"]) > 0.0


def test_fedbn_keeps_bn_local_and_evaluates_per_center(tmp_path):
    torch.manual_seed(0)
    run_main(tmp_path, "--method", "fedbn", "--rounds", "2", "--seed", "5", "--final_test")
    ck = _load(tmp_path / "last_fedbn_seed5.pt")
    states = ck["client_bn_states"]
    assert len(states) == 6 and all(s is not None for s in states)
    # 서버(글로벌) 모델의 BN은 집계되지 않아 초기값 그대로: running_mean=0, running_var=1
    sd = ck["model_state_dict"]
    assert torch.allclose(sd["bn.running_mean"], torch.zeros(4))
    assert torch.allclose(sd["bn.running_var"], torch.ones(4))
    # 센터별 BN 통계는 서로 달라야 한다
    means = torch.stack([s["bn.running_mean"] for s in states])
    assert (means.std(dim=0) > 0).any()
    # 합성곱 가중치는 집계되어 초기값에서 움직였어야 한다
    assert not torch.allclose(sd["fc.weight"], torch.zeros_like(sd["fc.weight"]))
    res = json.load(open(tmp_path / "fedbn_seed5_final_test.json", encoding="utf-8"))
    assert res["eval_split"] == "test" and len(res["final"]["client_balanced_acc"]) == 6
    assert res["best_val"] is not None
    npz = np.load(tmp_path / "fedbn_seed5_final_test_preds.npz")
    assert set(npz.files) == {"y_true", "y_pred", "y_prob", "client_idx"}


def test_fedbn_resume_restores_bn_states(tmp_path):
    run_main(tmp_path, "--method", "fedbn", "--rounds", "1", "--seed", "6")
    run_main(tmp_path, "--method", "fedbn", "--rounds", "2", "--seed", "6", "--resume")
    assert len(_rows(tmp_path / "fedbn_seed6_metrics.csv")) == 2


def test_tag_in_filenames(tmp_path):
    run_main(tmp_path, "--method", "fedavg", "--rounds", "1", "--seed", "7", "--tag", "r18")
    assert (tmp_path / "fedavg_r18_seed7_metrics.csv").exists()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
