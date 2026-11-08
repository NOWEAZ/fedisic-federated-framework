"""
병원별 개인화(로컬 미세조정) 평가.

학습이 끝난 FedProx 글로벌 모델(`last_*.pt`)에서 시작해, 각 병원이 *자기 학습 데이터*로 몇 십~몇 백 스텝만
추가로 학습한 뒤 그 병원의 test에서 평가한다. 연합학습은 다시 돌리지 않으며 병원 밖으로 데이터가 나가지 않는다.

설정 공간 (병원 6곳에 공통으로 하나만 고른다)
- scope: adabn(학습 없이 BN 통계만 병원 이미지로 재계산) / head(마지막 층만) / bn_head(BN 파라미터 + 마지막 층) / all(전체, FedProx 근접 항 유지)
- loss: global(FLamby 전역 가중 Focal Loss 그대로) / local(병원 로컬 학습 라벨 분포 기준 가중 Focal Loss)
- steps: 추가 학습 스텝 수
선택 기준은 병원별 *검증셋* balanced acc의 병원 평균이다. test는 설정을 고른 뒤 아래 세 가지에만 쓴다:
  raw(글로벌 모델 그대로), mask_only(병원 학습 라벨에 없는 클래스 제거), personalized(선택된 설정).
추론 시에는 모든 경우에 병원 *학습* 라벨에 없는 클래스를 제거한다(test 라벨은 쓰지 않는다).

이 평가는 "글로벌 모델 하나"가 아니라 "병원별 모델"을 평가하므로 기존 결과표와 별도 행으로 보고해야 한다.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data_utils import (
    BATCH_SIZE,
    NUM_CLIENTS,
    _extract_labels_from_dataset,
    get_dataloaders,
)
from federated_core import (
    ClientStepIterator,
    _collect_predictions,
    build_baseline_loss,
    build_efficientnet_b0,
    compute_ece,
)
from personalize_utils import (
    LOSS_KINDS,
    SCOPES,
    balanced_accuracy,
    class_counts,
    local_class_alpha,
    mask_probs,
    present_mask,
    select_config,
)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="FedProx 글로벌 모델의 병원별 개인화(로컬 미세조정) 평가")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, required=True, help="last_*.pt 가 있는 폴더 (학습 결과 폴더)")
    p.add_argument("--save_dir", type=str, default=None, help="결과 저장 폴더 (기본: ckpt_dir)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="fedprox_mu0.1_cosine", help="학습 때의 실험 태그")
    p.add_argument("--steps", type=str, default="50,100,200", help="비교할 추가 학습 스텝 수(쉼표 구분)")
    p.add_argument("--scopes", type=str, default=",".join(SCOPES))
    p.add_argument("--losses", type=str, default=",".join(LOSS_KINDS))
    p.add_argument("--lr", type=float, default=1e-4, help="미세조정 Adam 학습률")
    p.add_argument("--mu", type=float, default=0.1, help="글로벌 모델로부터 멀어지지 않게 하는 근접 항 계수(0이면 끔)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- 모델 설정
HEAD_PREFIXES = ("classifier", "base_model._fc", "_fc", "fc")  # torchvision: classifier, FLamby Baseline: base_model._fc


def _is_head_param(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def configure_trainable(model: nn.Module, scope: str):
    """scope에 맞춰 requires_grad를 설정하고 학습 대상 파라미터 리스트를 반환."""
    for prm in model.parameters():
        prm.requires_grad = scope == "all"
    if scope in ("head", "bn_head"):
        n_head = 0
        for name, prm in model.named_parameters():
            if _is_head_param(name):
                prm.requires_grad = True
                n_head += 1
        if n_head == 0:
            names = [n for n, _ in model.named_parameters()][-4:]
            raise RuntimeError(f"마지막 분류 층 파라미터를 찾지 못했습니다(HEAD_PREFIXES={HEAD_PREFIXES}). 마지막 파라미터 이름: {names}")
    if scope == "bn_head":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for prm in m.parameters(recurse=False):
                    prm.requires_grad = True
    return [prm for prm in model.parameters() if prm.requires_grad]


def set_train_mode(model: nn.Module, scope: str):
    """head 범위에서는 BN 통계와 드롭아웃이 바뀌지 않도록 모델 전체를 eval로 두고 마지막 층의 가중치만 학습한다."""
    if scope == "head":
        model.eval()
    else:
        model.train()


def local_focal_loss(logits, targets, alpha, present, gamma: float = 2.0):
    """병원 로컬 가중 Focal Loss. 학습 라벨에 없는 클래스의 로짓은 제외한다."""
    logits = logits.masked_fill(~present[None, :], -1e4)
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, targets[:, None]).squeeze(1)
    pt = logp_t.exp()
    return (-(alpha[targets] * (1.0 - pt) ** gamma * logp_t)).mean()


@torch.no_grad()
def predict(model, loader, device, criterion):
    model.eval()
    _, y_prob, _, y_true = _collect_predictions(model, loader, device, criterion, desc="Eval")
    return y_prob.astype(np.float64), y_true


def masked_metrics(y_true, probs, present):
    """병원 학습 라벨에 없는 클래스를 제거한 뒤의 예측과 balanced acc."""
    mp = mask_probs(probs, present)
    y_pred = mp.argmax(axis=1)
    return mp, y_pred, balanced_accuracy(y_true, y_pred)


# ---------------------------------------------------------------- 미세조정
def _adabn_client(global_model, loader, marks, ctx, k, val_loader=None):
    """
    AdaBN: 파라미터는 그대로 두고, 병원 학습 이미지를 forward만 해서 BatchNorm 러닝 통계를 그 병원 기준으로 다시 계산한다.
    라벨과 역전파가 필요 없다. marks는 통계 계산에 쓰는 배치 수(누적 평균).
    """
    device = ctx["device"]
    model = copy.deepcopy(global_model).to(device)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        raise RuntimeError("BatchNorm2d 층을 찾지 못했습니다.")
    orig_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # 누적 이동 평균 (배치 통계의 단순 평균)
    iterator = ClientStepIterator(loader)
    marks = sorted(set(int(m) for m in marks))
    scores = {}
    model.train()
    step = 0
    with torch.no_grad():
        while step < marks[-1]:
            images, _targets = iterator.get_next_batch()
            if images.size(0) == 1:
                continue
            model(images.to(device, non_blocking=True))
            step += 1
            if val_loader is not None and step in marks:
                probs, y_true = predict(model, val_loader, device, ctx["criterion"])
                scores[step] = masked_metrics(y_true, probs, ctx["present"][k])[2]
                model.train()
    for m, mom in zip(bns, orig_momentum):
        m.momentum = mom
    model.eval()
    return model, scores


def finetune_client(
    global_model, loader, scope, loss_kind, marks, ctx, k, val_loader=None
):
    """
    병원 k의 학습 데이터로 글로벌 모델을 미세조정한다.
    marks: 평가/종료할 스텝 번호들. val_loader가 있으면 각 mark에서 병원 val balanced acc(제거 적용)를 기록한다.
    반환: (최종 모델, {step: val balanced acc})
    """
    device = ctx["device"]
    if scope == "adabn":
        return _adabn_client(global_model, loader, marks, ctx, k, val_loader)
    model = copy.deepcopy(global_model).to(device)
    params = configure_trainable(model, scope)
    optimizer = torch.optim.Adam(params, lr=ctx["lr"])
    ref = [prm.detach().clone() for prm in params]
    iterator = ClientStepIterator(loader)
    present_t = torch.tensor(ctx["present"][k], dtype=torch.bool, device=device)
    alpha_t = torch.tensor(ctx["alpha"][k], dtype=torch.float32, device=device)
    marks = sorted(set(int(m) for m in marks))
    scores = {}

    set_train_mode(model, scope)
    step = 0
    while step < marks[-1]:
        images, targets = iterator.get_next_batch()
        if images.size(0) == 1:  # BatchNorm은 배치 크기 1에서 계산할 수 없음
            continue
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long().view(-1)

        optimizer.zero_grad()
        logits = model(images)
        if loss_kind == "local":
            loss = local_focal_loss(logits, targets, alpha_t, present_t)
        else:
            loss = ctx["criterion"](logits, targets)
        if ctx["mu"] > 0:
            prox = sum(((prm - r) ** 2).sum() for prm, r in zip(params, ref))
            loss = loss + (ctx["mu"] / 2.0) * prox
        loss.backward()
        optimizer.step()
        step += 1

        if val_loader is not None and step in marks:
            probs, y_true = predict(model, val_loader, device, ctx["criterion"])
            scores[step] = masked_metrics(y_true, probs, ctx["present"][k])[2]
            set_train_mode(model, scope)
    return model, scores


def summarize(per_client_true, per_client_prob, present):
    """병원별/전체 지표를 계산 (제거 적용된 확률 입력)."""
    bal, f1 = [], []
    all_true, all_prob, all_pred, all_idx = [], [], [], []
    for k, (yt, pr) in enumerate(zip(per_client_true, per_client_prob)):
        mp, yp, b = masked_metrics(yt, pr, present[k])
        bal.append(b)
        f1.append(float(f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0)))
        all_true.append(yt)
        all_prob.append(mp)
        all_pred.append(yp)
        all_idx.append(np.full(len(yt), k, dtype=np.int64))
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    y_pred = np.concatenate(all_pred)
    out = {
        "client_balanced_acc": bal,
        "client_macro_f1": f1,
        "min_bal_acc": float(np.min(bal)),
        "mean_client_bal_acc": float(np.mean(bal)),
        "bal_acc_std": float(np.std(bal, ddof=1)),
        "balanced_acc": balanced_accuracy(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "ece": compute_ece(y_prob, y_true),
    }
    preds = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob.astype(np.float32),
        "client_idx": np.concatenate(all_idx),
    }
    return out, preds


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = args.save_dir or args.ckpt_dir
    os.makedirs(save_dir, exist_ok=True)
    out_json = os.path.join(save_dir, f"{args.tag}_seed{args.seed}_personalize.json")
    out_npz = os.path.join(save_dir, f"{args.tag}_seed{args.seed}_personalize_preds.npz")
    if os.path.exists(out_json):
        print(f"[Personalize] 이미 결과가 있어 test를 다시 평가하지 않습니다: {out_json}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [int(s) for s in args.steps.split(",") if s]
    scopes = [s for s in args.scopes.split(",") if s]
    losses = [s for s in args.losses.split(",") if s]
    for s in scopes:
        if s not in SCOPES:
            raise ValueError(f"알 수 없는 scope: {s}")
    for s in losses:
        if s not in LOSS_KINDS:
            raise ValueError(f"알 수 없는 loss: {s}")

    client_loaders, _unused, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_path=args.data_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    val_loaders = val_loader.client_eval_loaders
    test_loaders = test_loader.client_test_loaders

    # 병원이 아는 정보: 자기 학습 라벨 분포뿐
    counts = [class_counts(_extract_labels_from_dataset(l.dataset)) for l in client_loaders]
    present = [present_mask(c) for c in counts]
    alpha = [local_class_alpha(c) for c in counts]
    for k in range(NUM_CLIENTS):
        print(f"[Info] 병원 {k} 학습 라벨 수: {counts[k].tolist()}")

    ckpt_path = os.path.join(args.ckpt_dir, f"last_{args.tag}_seed{args.seed}.pt")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    global_model = build_efficientnet_b0(num_classes=8, pretrained=True)
    global_model.load_state_dict(ckpt["model_state_dict"])
    global_model.to(device)
    print(f"[Info] 체크포인트 로드: {ckpt_path} (round {ckpt.get('round')})")

    criterion = build_baseline_loss(device=device)
    ctx = {
        "device": device,
        "criterion": criterion,
        "lr": args.lr,
        "mu": args.mu,
        "present": present,
        "alpha": alpha,
    }

    # 1) 검증셋에서 설정 탐색 (test는 열지 않는다)
    val_scores = {}
    base_val = []
    for k in range(NUM_CLIENTS):
        probs, y_true = predict(global_model, val_loaders[k], device, criterion)
        base_val.append(masked_metrics(y_true, probs, present[k])[2])
    print(f"[Val] 글로벌 모델(제거 적용) 병원별 balanced acc: {np.round(base_val, 4).tolist()} "
          f"평균 {np.mean(base_val):.4f}")

    for scope in scopes:
        for loss_kind in losses:
            if scope == "adabn" and loss_kind != losses[0]:
                continue  # AdaBN은 손실을 쓰지 않으므로 한 번만 실행
            per_client = []
            for k in range(NUM_CLIENTS):
                _, sc = finetune_client(
                    global_model, client_loaders[k], scope, loss_kind, steps, ctx, k, val_loader=val_loaders[k]
                )
                per_client.append(sc)
            for st in steps:
                val_scores[(scope, loss_kind, st)] = [per_client[k][st] for k in range(NUM_CLIENTS)]
                print(f"[Val] {scope:8s} {loss_kind:6s} steps={st:4d} → 병원별 "
                      f"{np.round(val_scores[(scope, loss_kind, st)], 4).tolist()} "
                      f"평균 {np.mean(val_scores[(scope, loss_kind, st)]):.4f}")

    (best_scope, best_loss, best_steps), best_val_mean = select_config(val_scores)
    print(f"[Select] val 평균 최고 설정: scope={best_scope}, loss={best_loss}, steps={best_steps} "
          f"(val 평균 {best_val_mean:.4f}; 글로벌 모델 {np.mean(base_val):.4f})")

    # 2) test 평가: raw / mask_only / personalized 각 1회
    raw_true, raw_prob = [], []
    for k in range(NUM_CLIENTS):
        probs, y_true = predict(global_model, test_loaders[k], device, criterion)
        raw_true.append(y_true)
        raw_prob.append(probs)

    # raw: 제거 없이 (기존 final_test.json과 같아야 한다)
    all_present = [np.ones(8, dtype=bool) for _ in range(NUM_CLIENTS)]
    raw_metrics, _ = summarize(raw_true, raw_prob, all_present)
    mask_metrics, _ = summarize(raw_true, raw_prob, present)

    pers_true, pers_prob = [], []
    for k in range(NUM_CLIENTS):
        model_k, _ = finetune_client(
            global_model, client_loaders[k], best_scope, best_loss, [best_steps], ctx, k
        )
        probs, y_true = predict(model_k, test_loaders[k], device, criterion)
        pers_true.append(y_true)
        pers_prob.append(probs)
        del model_k
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pers_metrics, pers_preds = summarize(pers_true, pers_prob, present)

    def show(name, m):
        print(f"[Test] {name:13s} 병원별 {np.round(m['client_balanced_acc'], 4).tolist()} | "
              f"최저 {m['min_bal_acc']:.4f} 평균 {m['mean_client_bal_acc']:.4f} 표준편차 {m['bal_acc_std']:.4f} | "
              f"전체 bal acc {m['balanced_acc']:.4f} macro-F1 {m['macro_f1']:.4f} ECE {m['ece']:.4f}")

    show("raw", raw_metrics)
    show("mask_only", mask_metrics)
    show("personalized", pers_metrics)

    result = {
        "seed": args.seed,
        "tag": args.tag,
        "checkpoint_round": ckpt.get("round"),
        "finetune": {"lr": args.lr, "mu": args.mu, "steps_grid": steps, "scopes": scopes, "losses": losses},
        "selected": {"scope": best_scope, "loss": best_loss, "steps": best_steps, "val_mean": best_val_mean},
        "val_global_model_masked": base_val,
        "val_grid": {f"{s}|{l}|{st}": v for (s, l, st), v in val_scores.items()},
        "train_label_counts": [c.tolist() for c in counts],
        "test": {"raw": raw_metrics, "mask_only": mask_metrics, "personalized": pers_metrics},
        "note": "병원별 모델 평가. 글로벌 모델 결과와 별도 설정으로 보고할 것.",
    }
    np.savez_compressed(out_npz, **pers_preds)
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_json)
    print(f"[Done] {out_json}")


if __name__ == "__main__":
    main()
