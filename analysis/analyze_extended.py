"""확장 분석: 클래스별 recall, 혼동행렬, 부트스트랩 신뢰구간, 시드별 짝지은 비교.

입력 : <run>_seed<seed>_final_test_preds.npz  (--final_test가 저장한 마지막 라운드 모델의 test 예측:
       y_true, y_pred, y_prob, client_idx).  학습을 다시 돌리지 않고 이 파일만으로 계산한다.
실행 : python analysis/analyze_extended.py --results_dir <결과폴더> [--out_dir <출력폴더>]
       [--runs fedavg fedprox_mu0.1 ...] [--pairs fedprox_mu0.1:fedavg ...] [--n_boot 2000]

방법 이름(run)은 파일 이름 접두어 그대로다 (예: centralized, fedavg, fedprox_mu0.1, fedavg_keepopt, fedbn).
--runs를 생략하면 폴더에서 찾은 전부를 쓰고, --pairs를 생략하면 fedavg가 있을 때 "다른 연합 방법 : fedavg"를 짝짓는다.

부트스트랩 방식
- test 표본을 "센터 안에서" 복원추출한다 (센터별 표본 수를 고정하는 설계에 맞춤).
- 한 번 뽑은 표본 인덱스를 모든 방법·모든 시드에 똑같이 적용한다 (test 순서가 모든 실행에서 같음을 확인함).
  그래서 방법 간 차이의 신뢰구간이 짝지은(paired) 비교가 된다.
- 한 방법의 점추정은 시드 평균이고, 신뢰구간은 "같은 재추출에서 시드 평균을 다시 계산"한 분포의 2.5~97.5% 구간이다.
  이 구간은 test 표본의 불확실성만 반영한다. 시드 사이 변동(학습 난수)은 paired_by_seed.csv의 시드별 값으로 따로 본다.
- 정의: balanced acc = 정답에 존재하는 클래스 recall 평균, macro-F1 = 정답에 존재하는 클래스 F1 평균
  (evaluate_model의 macro_f1과 같은 정의).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = len(CLASS_NAMES)
N_CENTERS = 6
METRICS = ["balanced_acc", "macro_f1"]
NDIGITS = 4


def confusion(y_true, y_pred, n_classes=N_CLASSES):
    """행=정답, 열=예측인 혼동행렬 (정수 개수)."""
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_cm(cm):
    """혼동행렬에서 (balanced_acc, macro_f1). 정답에 존재하는 클래스만 평균한다."""
    support = cm.sum(axis=1)
    present = support > 0
    tp = np.diag(cm).astype(np.float64)
    recall = tp[present] / support[present]
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (support - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(recall.mean()), float(f1[present].mean())


def load_runs(results_dir: Path, wanted):
    """{run: {seed: dict(y_true, y_pred, client_idx)}} 와 기준 y_true / client_idx 를 돌려준다."""
    pat = re.compile(r"^(?P<run>.+)_seed(?P<seed>\d+)_final_test_preds\.npz$")
    runs: dict[str, dict[int, dict]] = {}
    for f in sorted(results_dir.glob("*_final_test_preds.npz")):
        m = pat.match(f.name)
        if not m:
            continue
        run, seed = m.group("run"), int(m.group("seed"))
        if wanted and run not in wanted:
            continue
        z = np.load(f)
        runs.setdefault(run, {})[seed] = {k: z[k] for k in ("y_true", "y_pred", "client_idx")}
    if not runs:
        raise SystemExit(f"{results_dir}에서 *_final_test_preds.npz를 찾지 못했습니다.")
    ref = next(iter(next(iter(runs.values())).values()))
    for run, by_seed in runs.items():
        for seed, d in by_seed.items():
            if not (np.array_equal(d["y_true"], ref["y_true"]) and np.array_equal(d["client_idx"], ref["client_idx"])):
                raise SystemExit(f"{run} seed{seed}: test 표본 순서가 다른 실행과 다릅니다. 짝지은 부트스트랩을 쓸 수 없습니다.")
    return runs, ref["y_true"], ref["client_idx"]


def per_center_cms(y_true, y_pred, client_idx, idx=None):
    """센터별 혼동행렬 배열 (N_CENTERS, C, C). idx가 주어지면 그 인덱스(재추출)로 계산."""
    if idx is not None:
        y_true, y_pred, client_idx = y_true[idx], y_pred[idx], client_idx[idx]
    out = np.zeros((N_CENTERS, N_CLASSES, N_CLASSES), dtype=np.int64)
    for c in range(N_CENTERS):
        m = client_idx == c
        out[c] = confusion(y_true[m], y_pred[m])
    return out


def point_table(runs, y_true, client_idx):
    """{run: {seed: {'overall': (bal, f1), 'centers': [(bal, f1)...]}}} 점추정."""
    table = {}
    for run, by_seed in runs.items():
        table[run] = {}
        for seed, d in by_seed.items():
            cms = per_center_cms(y_true, d["y_pred"], client_idx)
            table[run][seed] = {
                "overall": metrics_from_cm(confusion(y_true, d["y_pred"])),
                "centers": [metrics_from_cm(cms[c]) for c in range(N_CENTERS)],
            }
    return table


def bootstrap(runs, y_true, client_idx, n_boot, rng):
    """
    반복마다 센터 안에서 복원추출한 인덱스 하나를 모든 (run, seed)에 적용해
    seed 평균 지표를 모은다. 반환: {run: array (n_boot, 1+N_CENTERS, 2)}  (0번째 = 전체, 이후 센터별; 마지막 축 = bal, f1)
    """
    center_rows = [np.flatnonzero(client_idx == c) for c in range(N_CENTERS)]
    out = {run: np.zeros((n_boot, 1 + N_CENTERS, 2)) for run in runs}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in center_rows])
        for run, by_seed in runs.items():
            acc = np.zeros((1 + N_CENTERS, 2))
            for d in by_seed.values():
                cms = per_center_cms(y_true, d["y_pred"], client_idx, idx)
                acc[0] = np.asarray(acc[0]) + metrics_from_cm(cms.sum(axis=0))
                for c in range(N_CENTERS):
                    acc[1 + c] += metrics_from_cm(cms[c])
            out[run][b] = acc / len(by_seed)
    return out


def ci(values, alpha=0.05):
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None, help="기본값: <results_dir>/analysis_extended")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--pairs", nargs="*", default=None, help="'A:B' 형식. 차이는 A - B")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_dir / "analysis_extended")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, y_true, client_idx = load_runs(args.results_dir, set(args.runs) if args.runs else None)
    names = list(runs)
    n_test = np.bincount(client_idx, minlength=N_CENTERS)
    print(f"runs: { {r: sorted(s) for r, s in runs.items()} } | test 표본 {len(y_true)} (센터별 {n_test.tolist()})")

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    elif "fedavg" in runs:
        pairs = [(r, "fedavg") for r in names if r not in ("fedavg", "centralized")]
    else:
        pairs = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            raise SystemExit(f"--pairs의 {a}:{b} 중 결과가 없는 방법이 있습니다. 사용 가능: {names}")

    pts = point_table(runs, y_true, client_idx)

    # 1) 클래스별 recall / precision / 예측 비율, 2) 혼동행렬 (시드 평균)
    support = np.bincount(y_true, minlength=N_CLASSES)
    rows = []
    for run, by_seed in runs.items():
        recs, precs, shares, cm_sum = [], [], [], np.zeros((N_CLASSES, N_CLASSES))
        for d in by_seed.values():
            cm = confusion(y_true, d["y_pred"])
            cm_sum += cm
            recs.append(np.divide(np.diag(cm), support, out=np.full(N_CLASSES, np.nan), where=support > 0))
            col = cm.sum(axis=0)
            precs.append(np.divide(np.diag(cm), col, out=np.full(N_CLASSES, np.nan), where=col > 0))
            shares.append(col / cm.sum())
        recs, precs, shares = np.array(recs), np.array(precs), np.array(shares)
        for k, name in enumerate(CLASS_NAMES):
            rows.append({
                "method": run, "class": name, "n_test": int(support[k]),
                "recall_mean": np.nanmean(recs[:, k]), "recall_std": np.nanstd(recs[:, k], ddof=1) if len(recs) > 1 else np.nan,
                "precision_mean": np.nanmean(precs[:, k]) if not np.all(np.isnan(precs[:, k])) else np.nan,
                "pred_share_mean": shares[:, k].mean(), "true_share": support[k] / support.sum(),
            })
        mean_cm = cm_sum / len(by_seed)
        norm = np.divide(mean_cm, support[:, None], out=np.zeros_like(mean_cm), where=support[:, None] > 0)
        df = pd.DataFrame(norm, index=[f"true_{n}" for n in CLASS_NAMES], columns=[f"pred_{n}" for n in CLASS_NAMES])
        df.insert(0, "n_test", support)
        df.round(NDIGITS).to_csv(out_dir / f"confusion_{run}.csv", float_format=f"%.{NDIGITS}f")
    pd.DataFrame(rows).round(NDIGITS).to_csv(out_dir / "per_class_recall.csv", index=False, float_format=f"%.{NDIGITS}f")

    # 3) 부트스트랩
    rng = np.random.default_rng(args.boot_seed)
    boot = bootstrap(runs, y_true, client_idx, args.n_boot, rng)

    overall, per_center = [], []
    for run in names:
        seeds = sorted(runs[run])
        mean_pt = np.mean([[pts[run][s]["overall"][m] for m in range(2)] for s in seeds], axis=0)
        for m, metric in enumerate(METRICS):
            lo, hi = ci(boot[run][:, 0, m])
            overall.append({"method": run, "metric": metric, "n_seeds": len(seeds),
                            "mean_over_seeds": mean_pt[m], "ci_low": lo, "ci_high": hi})
        for c in range(N_CENTERS):
            c_pt = np.mean([[pts[run][s]["centers"][c][m] for m in range(2)] for s in seeds], axis=0)
            for m, metric in enumerate(METRICS):
                lo, hi = ci(boot[run][:, 1 + c, m])
                per_center.append({"method": run, "center": c, "n_test": int(n_test[c]), "metric": metric,
                                   "mean_over_seeds": c_pt[m], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(overall).round(NDIGITS).to_csv(out_dir / "bootstrap_ci_overall.csv", index=False, float_format=f"%.{NDIGITS}f")
    pd.DataFrame(per_center).round(NDIGITS).to_csv(out_dir / "bootstrap_ci_per_center.csv", index=False, float_format=f"%.{NDIGITS}f")

    # 4) 시드별 짝지은 비교
    by_seed_rows, summary_rows, center_rows = [], [], []
    for a, b in pairs:
        common = sorted(set(runs[a]) & set(runs[b]))
        if not common:
            print(f"[경고] {a}:{b}는 공통 시드가 없어 건너뜁니다.")
            continue
        for s in common:
            row = {"pair": f"{a} - {b}", "seed": s}
            for m, metric in enumerate(METRICS):
                d = pts[a][s]["overall"][m] - pts[b][s]["overall"][m]
                row[f"{metric}_diff"] = d
                row[f"{metric}_sign"] = "+" if d > 0 else ("-" if d < 0 else "0")
            by_seed_rows.append(row)
        # mean_diff는 공통 시드끼리의 차이 평균. 부트스트랩 구간은 각 방법의 시드 평균 차이 (두 방법의 시드 집합이 같으면 동일한 양)
        for m, metric in enumerate(METRICS):
            diffs = np.array([pts[a][s]["overall"][m] - pts[b][s]["overall"][m] for s in common])
            bd = boot[a][:, 0, m] - boot[b][:, 0, m]
            lo, hi = ci(bd)
            summary_rows.append({
                "pair": f"{a} - {b}", "metric": metric, "n_common_seeds": len(common),
                "mean_diff": diffs.mean(), "n_seeds_positive": int((diffs > 0).sum()),
                "min_diff": diffs.min(), "max_diff": diffs.max(),
                "boot_ci_low": lo, "boot_ci_high": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
            for c in range(N_CENTERS):
                cd = np.mean([pts[a][s]["centers"][c][m] - pts[b][s]["centers"][c][m] for s in common])
                lo, hi = ci(boot[a][:, 1 + c, m] - boot[b][:, 1 + c, m])
                center_rows.append({"pair": f"{a} - {b}", "center": c, "n_test": int(n_test[c]), "metric": metric,
                                    "mean_diff": cd, "boot_ci_low": lo, "boot_ci_high": hi})
    if by_seed_rows:
        pd.DataFrame(by_seed_rows).round(NDIGITS).to_csv(out_dir / "paired_by_seed.csv", index=False, float_format=f"%.{NDIGITS}f")
        pd.DataFrame(summary_rows).round(NDIGITS).to_csv(out_dir / "paired_summary.csv", index=False, float_format=f"%.{NDIGITS}f")
        pd.DataFrame(center_rows).round(NDIGITS).to_csv(out_dir / "paired_per_center.csv", index=False, float_format=f"%.{NDIGITS}f")
    print("wrote:", ", ".join(sorted(p.name for p in out_dir.iterdir())))


if __name__ == "__main__":
    main()
