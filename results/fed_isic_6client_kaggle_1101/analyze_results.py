"""Fed-ISIC2019 6-client 결과 요약 스크립트.

사용법:
    python analyze_results.py [--results_dir 결과폴더] [--out_dir 출력폴더]
기본값은 이 스크립트가 있는 폴더입니다.

입력 : <method>_seed<seed>_final_test.json (--final_test 로 마지막 라운드 모델을 test 1회 평가한 결과),
       <method>_seed<seed>_metrics.csv     (라운드별 val 지표; cumulative_comm_mb 사용),
       *_final_test_preds.npz              (client_idx 로 센터별 test 표본 수 계산)
출력 : summary_test_by_seed.csv, summary_test_mean_std.csv, summary_test_per_client.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [("centralized", "centralized"), ("fedavg", "fedavg"), ("fedprox", "fedprox_mu0.1")]
SEEDS = [42, 100, 2026]
METRICS = ["balanced_acc", "macro_f1", "macro_f1_fixed8", "ece",
           "macro_f1_std", "macro_f1_gap", "bal_acc_std", "bal_acc_gap"]
N_CLIENTS = 6
NDIGITS = 4


def run_name(prefix, seed):
    return f"{prefix}_seed{seed}"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=here)
    ap.add_argument("--out_dir", type=Path, default=here)
    args = ap.parse_args()
    rd, od = args.results_dir, args.out_dir
    od.mkdir(parents=True, exist_ok=True)

    by_seed, per_client = [], []
    for method, prefix in METHODS:
        for seed in SEEDS:
            name = run_name(prefix, seed)
            res = json.loads((rd / f"{name}_final_test.json").read_text(encoding="utf-8"))
            assert res["eval_split"] == "test", name
            final = res["final"]
            metrics = pd.read_csv(rd / f"{name}_metrics.csv")
            last = metrics.loc[metrics["round"].idxmax()]
            assert int(last["round"]) == res["final_round"], name

            row = {"method": method, "seed": seed, "final_round": res["final_round"]}
            row.update({k: final[k] for k in METRICS})
            row["cumulative_comm_mb"] = float(last["cumulative_comm_mb"])
            by_seed.append(row)

            cidx = np.load(rd / f"{name}_final_test_preds.npz")["client_idx"]
            n_test = np.bincount(cidx, minlength=N_CLIENTS)
            for c in range(N_CLIENTS):
                per_client.append({
                    "method": method, "seed": seed, "center": c,
                    "balanced_acc": final["client_balanced_acc"][c],
                    "macro_f1": final["client_macro_f1"][c],
                    "n_test": int(n_test[c]),
                })

    df = pd.DataFrame(by_seed)
    df.round(NDIGITS).to_csv(od / "summary_test_by_seed.csv", index=False, float_format=f"%.{NDIGITS}f")

    cols = METRICS + ["cumulative_comm_mb"]
    g = df.groupby("method", sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    ms = pd.DataFrame({"method": mean.index, "n_seeds": g.size().values})
    for k in cols:
        ms[f"{k}_mean"] = mean[k].values
        ms[f"{k}_std"] = std[k].values
    ms.round(NDIGITS).to_csv(od / "summary_test_mean_std.csv", index=False, float_format=f"%.{NDIGITS}f")

    pc = pd.DataFrame(per_client)
    out = (pc.groupby(["method", "center"], sort=False)
             .agg(balanced_acc=("balanced_acc", "mean"), macro_f1=("macro_f1", "mean"),
                  n_test=("n_test", "first"), n_test_unique=("n_test", "nunique"))
             .reset_index())
    assert (out["n_test_unique"] == 1).all(), "센터별 test 표본 수가 시드마다 다릅니다"
    out = out.drop(columns="n_test_unique")
    out.round(NDIGITS).to_csv(od / "summary_test_per_client.csv", index=False, float_format=f"%.{NDIGITS}f")
    print("wrote 3 CSVs to", od)


if __name__ == "__main__":
    main()
