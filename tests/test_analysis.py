"""analysis/analyze_extended.py의 지표 계산이 sklearn과 같은지, 전체 파이프라인이 도는지 확인한다 (CPU, 합성 데이터)."""
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import analyze_extended as ae  # noqa: E402


def _make(tmp_path, runs=("fedavg", "fedprox_mu0.1"), seeds=(1, 2, 3), n=600):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 8, n)
    client_idx = np.sort(rng.integers(0, 6, n))
    for r_i, run in enumerate(runs):
        for s in seeds:
            noise = rng.random(n) < (0.5 - 0.1 * r_i)
            y_pred = np.where(noise, rng.integers(0, 8, n), y_true)
            np.savez_compressed(tmp_path / f"{run}_seed{s}_final_test_preds.npz",
                                y_true=y_true, y_pred=y_pred, y_prob=np.zeros((n, 8)), client_idx=client_idx)
    return y_true, client_idx


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.integers(20, 200)
        y = rng.integers(0, 8, n)
        # 일부 클래스가 정답에 없는 경우를 포함
        y[y == 5] = 1
        p = np.where(rng.random(n) < 0.4, rng.integers(0, 8, n), y)
        bal, f1 = ae.metrics_from_cm(ae.confusion(y, p))
        assert abs(bal - balanced_accuracy_score(y, p)) < 1e-12
        assert abs(f1 - f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)) < 1e-12


def test_pipeline_runs_and_outputs(tmp_path):
    _make(tmp_path)
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "analyze_extended.py"),
                    "--results_dir", str(tmp_path), "--out_dir", str(out), "--n_boot", "30",
                    "--pairs", "fedprox_mu0.1:fedavg"], check=True)
    names = {p.name for p in out.iterdir()}
    for need in ("per_class_recall.csv", "bootstrap_ci_overall.csv", "bootstrap_ci_per_center.csv",
                 "paired_by_seed.csv", "paired_summary.csv", "paired_per_center.csv",
                 "confusion_fedavg.csv", "confusion_fedprox_mu0.1.csv"):
        assert need in names, need
    import pandas as pd

    ci = pd.read_csv(out / "bootstrap_ci_overall.csv")
    assert ((ci.ci_low <= ci.mean_over_seeds + 0.05) & (ci.ci_high >= ci.mean_over_seeds - 0.05)).all()
    pair = pd.read_csv(out / "paired_by_seed.csv")
    assert len(pair) == 3 and set(pair.seed) == {1, 2, 3}
