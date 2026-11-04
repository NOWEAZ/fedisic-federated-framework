import argparse
import copy
import csv
import json
import os
import random
import numpy as np
import torch

from data_utils import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    Optimizer,
    get_dataloaders,
    get_nb_max_rounds,
)
from lr_schedule import LR_SCHEDULES, round_lr
from federated_core import (
    ClientStepIterator,
    build_baseline_loss,
    build_efficientnet_b0,
    compute_model_size_mb,
    compute_round_comm_mb,
    compute_weight_divergence,
    evaluate_model,
    server_aggregate,
    train_one_client,
)


# 매 라운드 CSV에는 검증(validation) 지표만 기록한다. test는 학습 종료 후 1회만 평가한다 (--final_test).
CSV_FIELDS = [
    "round",
    "method",
    "mu",
    "seed",
    "eval_split",
    "train_loss",
    "val_loss",
    "balanced_acc",
    "macro_f1",
    "macro_f1_fixed8",
    "ece",
    "max_macro_f1",
    "min_macro_f1",
    "macro_f1_gap",
    "macro_f1_std",
    "max_bal_acc",
    "min_bal_acc",
    "bal_acc_gap",
    "bal_acc_std",
    *[f"balanced_acc_client{k}" for k in range(NUM_CLIENTS)],
    *[f"macro_f1_client{k}" for k in range(NUM_CLIENTS)],
    *[f"macro_f1_fixed8_client{k}" for k in range(NUM_CLIENTS)],
    "weight_divergence",
    "cumulative_comm_mb",
    "lr",
]

# resume 시 기존 체크포인트와 반드시 일치해야 하는 실험 설정 키
RESUME_CONFIG_KEYS = [
    "method",
    "mu",
    "seed",
    "lr",
    "num_updates",
    "local_epochs",
    "batch_size",
    "val_fraction",
    "split_seed",
    "lr_schedule",
    "lr_min_ratio",
]
# 이 옵션이 추가되기 전에 만든 체크포인트에는 키가 없으므로, 없으면 기본값(기존 동작)으로 간주한다.
RESUME_CONFIG_DEFAULTS = {"lr_schedule": "constant", "lr_min_ratio": 0.0}


def set_seed(seed: int):
    """실험 재현성을 위해 Python, NumPy, PyTorch(CPU/GPU) 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fed-ISIC2019 6-Client Federated Learning (Centralized / FedAvg / FedProx)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        choices=["centralized", "fedavg", "fedprox"],
        help="학습 방식 선택: centralized, fedavg, fedprox",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "총 통신 라운드 수. 미지정(None) 시 FLamby 공식 스케줄 자동 적용: "
            f"연합학습은 get_nb_max_rounds(num_updates), centralized는 NUM_EPOCHS_POOLED({NUM_EPOCHS_POOLED})"
        ),
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=100,
        help=(
            "연합학습(fedavg/fedprox)에서 라운드당 클라이언트 로컬 미니배치 업데이트(step) 횟수 "
            "(FLamby 공식 벤치마크 기본값: 100). 0 이하로 주면 --local_epochs 기반으로 동작"
        ),
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="centralized의 라운드당 에폭 수 (기본값: 1) 또는 --num_updates <= 0일 때의 로컬 에폭 수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"미니배치 크기 (FLamby 공식 기본값: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"학습률 Learning Rate (FLamby 공식 기본값: {LR})",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=list(LR_SCHEDULES),
        help="라운드(Centralized는 에폭) 단위 학습률 스케줄. constant=FLamby 공식(기본), "
             "cosine=1라운드 --lr에서 시작해 코사인으로 감소",
    )
    parser.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.0,
        help="cosine 스케줄의 하한 비율 (최저 lr = --lr * 이 값). constant에서는 무시",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx 근접 항 계수 (method='fedprox'일 때 적용, fedavg/centralized는 0.0으로 강제)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="로그 CSV 및 최고 성능 모델 가중치(.pt) 저장 디렉토리",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="전처리된 Fed-ISIC2019 데이터 경로 (ISIC_2019_Training_Input_preprocessed 폴더의 상위 경로). "
             "None이면 FLamby의 dataset_location.yaml 설정을 사용",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers 수",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성 및 다중 시드 실험을 위한 난수 시드 (기본값: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="save_dir에 동일 실험의 last 체크포인트가 있으면 마지막 완료 라운드 다음부터 이어서 학습",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="센터별 train에서 층화로 떼어 낼 검증(validation) 비율. 매 라운드 평가와 모델/mu 선택에 사용",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="train/val 분할 시드. 실험 시드(--seed)와 무관하게 고정해 모든 실행이 같은 검증셋을 쓰도록 함",
    )
    parser.add_argument(
        "--final_test",
        action="store_true",
        help="학습 종료 후 마지막 라운드 모델(과 best-val 모델)을 FLamby test로 딱 1회 평가해 "
             "*_final_test.json / *_final_test_preds.npz로 저장 (이미 결과 파일이 있으면 다시 열지 않음)",
    )
    return parser.parse_args()


def _atomic_torch_save(obj, path: str):
    """저장 도중 세션이 끊겨도 기존 파일이 손상되지 않도록 임시 파일에 저장 후 교체합니다."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _capture_rng_states(client_loaders, centralized_loader) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "client_loader_generators": [
            l.generator.get_state() if l.generator is not None else None for l in client_loaders
        ],
        "central_loader_generator": (
            centralized_loader.generator.get_state()
            if centralized_loader.generator is not None
            else None
        ),
    }


def _restore_rng_states(states: dict, client_loaders, centralized_loader):
    try:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"].cpu())
        if torch.cuda.is_available() and states.get("cuda") is not None:
            cuda_states = [s.cpu() for s in states["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        for loader, g_state in zip(client_loaders, states.get("client_loader_generators", [])):
            if loader.generator is not None and g_state is not None:
                loader.generator.set_state(g_state.cpu())
        c_state = states.get("central_loader_generator")
        if centralized_loader.generator is not None and c_state is not None:
            centralized_loader.generator.set_state(c_state.cpu())
    except Exception as e:
        print(f"[Resume Warning] 난수 상태 일부 복원 생략 (GPU/환경 변경 가능성): {e}")


def _truncate_csv(csv_path: str, last_round: int):
    """resume 시 체크포인트 이후에 기록된(중복될) CSV 행을 제거합니다."""
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if int(row["round"]) <= last_round]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _atomic_write_text(path: str, text: str):
    tmp_path = path + ".tmp"
    with open(tmp_path, mode="w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def _run_final_test(
    args,
    run_config: dict,
    exp_tag: str,
    global_model,
    best_model_path: str,
    test_loader,
    criterion,
    device,
):
    """
    학습이 모두 끝난 뒤 FLamby test를 딱 1회 평가합니다.

    - 마지막 라운드 모델(주 결과)과 best-val 모델(참고)을 평가합니다. 둘 다 test가 아니라 val로 정해진 모델입니다.
    - 결과 파일이 이미 있으면 test를 다시 열지 않고 건너뜁니다 (여러 번 보고 고르는 것을 방지).
    - 마지막 모델의 예측(y_true, y_pred, y_prob, client_idx)을 .npz로 저장해 bootstrap 신뢰구간 등을
      학습 재실행 없이 계산할 수 있게 합니다.
    """
    json_path = os.path.join(args.save_dir, f"{exp_tag}_seed{args.seed}_final_test.json")
    npz_path = os.path.join(args.save_dir, f"{exp_tag}_seed{args.seed}_final_test_preds.npz")
    if os.path.exists(json_path):
        print(f"[Final Test] 이미 결과가 있어 test를 다시 평가하지 않습니다: {json_path}")
        return

    def _strip(metrics: dict) -> dict:
        return {k: v for k, v in metrics.items() if k != "predictions"}

    print("[Final Test] 마지막 라운드 모델을 test 셋으로 평가합니다 (1회).")
    final_metrics = evaluate_model(
        model=global_model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        return_predictions=True,
    )
    preds = final_metrics["predictions"]
    np.savez_compressed(
        npz_path,
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        y_prob=preds["y_prob"],
        client_idx=preds["client_idx"],
    )

    result = {
        "eval_split": "test",
        "config": run_config,
        "final_round": args.rounds,
        "final": _strip(final_metrics),
        "best_val": None,
    }

    if os.path.exists(best_model_path):
        try:
            best_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
        except TypeError:
            best_ckpt = torch.load(best_model_path, map_location="cpu")
        best_model = copy.deepcopy(global_model)
        best_model.load_state_dict(best_ckpt["model_state_dict"])
        best_metrics = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            criterion=criterion,
        )
        result["best_val"] = {
            "round": best_ckpt.get("round"),
            "val_balanced_acc": best_ckpt.get("balanced_acc"),
            **_strip(best_metrics),
        }

    _atomic_write_text(json_path, json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"[Final Test] Bal Acc: {final_metrics['balanced_acc']:.4f} "
        f"(Client Std: {final_metrics['bal_acc_std']:.4f}, Gap: {final_metrics['bal_acc_gap']:.4f}) | "
        f"Macro F1: {final_metrics['macro_f1']:.4f} "
        f"(Client Std: {final_metrics['macro_f1_std']:.4f}, Gap: {final_metrics['macro_f1_gap']:.4f}) | "
        f"ECE: {final_metrics['ece']:.4f} → {json_path}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # FLamby 공식 벤치마크 학습 스케줄 결정
    if args.method == "centralized":
        effective_num_updates = None
        if args.rounds is None:
            args.rounds = NUM_EPOCHS_POOLED
    else:
        effective_num_updates = args.num_updates if args.num_updates > 0 else None
        if args.rounds is None:
            if effective_num_updates is not None:
                args.rounds = get_nb_max_rounds(effective_num_updates, batch_size=args.batch_size)
            else:
                args.rounds = NUM_EPOCHS_POOLED

    effective_mu = args.mu if args.method == "fedprox" else 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device} | Seed: {args.seed}")
    schedule_desc = (
        f"Num Updates/Round: {effective_num_updates}"
        if effective_num_updates is not None
        else f"Local Epochs/Round: {args.local_epochs}"
    )
    print(
        f"[Info] Method: {args.method} | Clients: {NUM_CLIENTS} | Rounds: {args.rounds} | "
        f"{schedule_desc} | Batch Size: {args.batch_size} | LR: {args.lr} | Mu: {effective_mu}"
    )

    # 1. DataLoader 준비 (FLamby 공식 6개 센터: Center 0 ~ 5)
    client_loaders, centralized_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_path=args.data_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )

    num_clients = len(client_loaders)
    if num_clients != NUM_CLIENTS:
        raise ValueError(
            f"클라이언트 수({num_clients})가 CSV 열 정의(NUM_CLIENTS={NUM_CLIENTS})와 다릅니다."
        )
    client_sample_counts = [len(loader.dataset) for loader in client_loaders]
    # FLamby DataLoaderWithMemory와 동일하게 라운드 간 반복자 상태를 유지하는 래퍼
    client_step_iterators = [ClientStepIterator(loader) for loader in client_loaders]

    print(f"[Info] Number of clients: {num_clients} | Client samples: {client_sample_counts}")
    print(
        f"[Info] Centralized train samples: {len(centralized_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)} "
        f"(val_fraction={args.val_fraction}, split_seed={args.split_seed})"
    )

    # 2. FLamby 공식 손실 함수 (BaselineLoss: Weighted Focal Loss) 초기화
    criterion = build_baseline_loss(device=device)

    # 3. 글로벌 모델(FLamby 공식 Baseline: EfficientNet-B0) 초기화 및 1라운드당 통신량(MB) 계산
    global_model = build_efficientnet_b0(num_classes=8, pretrained=True).to(device)
    model_size_mb = compute_model_size_mb(global_model)
    round_comm_mb = (
        0.0
        if args.method == "centralized"
        else compute_round_comm_mb(global_model, num_clients)
    )
    print(
        f"[Info] Model Size: {model_size_mb:.2f} MB | "
        f"Comm Volume per Round (Down+Up): {round_comm_mb:.2f} MB"
    )

    # FLamby 공식 구현(_Model 및 Optimizer = torch.optim.Adam)과 동일하게 옵티마이저를 루프 밖에서 1회 생성해 상태 유지:
    # - Centralized: 단일 글로벌 모델에 대한 central_optimizer 유지
    # - 연합학습(FedAvg/FedProx): 6개 클라이언트별 로컬 모델(client_models)과 로컬 옵티마이저(client_optimizers)를
    #   루프 밖에서 1회 생성하고, 매 라운드 글로벌 가중치만 load_state_dict로 동기화하여 Adam 모멘텀 상태를 유지
    if args.method == "centralized":
        central_optimizer = Optimizer(global_model.parameters(), lr=args.lr)
        client_models = None
        client_optimizers = None
    else:
        central_optimizer = None
        client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
        client_optimizers = [
            Optimizer(m.parameters(), lr=args.lr) for m in client_models
        ]

    # 4. 결과 기록용 CSV 및 모델 저장 경로 설정 (시드 및 FedProx mu 값 구분)
    if args.method == "fedprox":
        exp_tag = f"{args.method}_mu{effective_mu:g}"
    else:
        exp_tag = f"{args.method}"
    if args.lr_schedule != "constant":
        # 기본(constant) 실험과 결과 파일이 섞이지 않도록 태그에 스케줄을 반영
        exp_tag += f"_{args.lr_schedule}"
        if args.lr_min_ratio > 0:
            exp_tag += f"min{args.lr_min_ratio:g}"

    csv_path = os.path.join(args.save_dir, f"{exp_tag}_seed{args.seed}_metrics.csv")
    best_model_path = os.path.join(args.save_dir, f"best_{exp_tag}_seed{args.seed}.pt")
    last_ckpt_path = os.path.join(args.save_dir, f"last_{exp_tag}_seed{args.seed}.pt")

    run_config = {
        "method": args.method,
        "mu": effective_mu,
        "seed": args.seed,
        "lr": args.lr,
        "rounds": args.rounds,
        "num_updates": effective_num_updates,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "lr_schedule": args.lr_schedule,
        "lr_min_ratio": args.lr_min_ratio,
    }

    start_round = 1
    best_balanced_acc = -1.0

    # 5. Resume: 동일 실험의 last 체크포인트가 있으면 이어서 학습
    if args.resume and os.path.exists(last_ckpt_path):
        try:
            ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(last_ckpt_path, map_location="cpu")
        saved_config = ckpt.get("config", {})
        mismatched = [
            k
            for k in RESUME_CONFIG_KEYS
            if saved_config.get(k, RESUME_CONFIG_DEFAULTS.get(k)) != run_config[k]
        ]
        if mismatched:
            raise ValueError(
                f"[Resume] 체크포인트 설정과 현재 인자가 다릅니다: "
                + ", ".join(f"{k} (ckpt={saved_config.get(k, RESUME_CONFIG_DEFAULTS.get(k))}, now={run_config[k]})" for k in mismatched)
            )

        global_model.load_state_dict(ckpt["model_state_dict"])
        global_model.to(device)
        if central_optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            central_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if client_models is not None:
            for c_model in client_models:
                c_model.load_state_dict(global_model.state_dict())
                c_model.to(device)
            saved_client_opts = ckpt.get("client_optimizer_state_dicts")
            if client_optimizers is not None and saved_client_opts is not None:
                for c_opt, opt_state in zip(client_optimizers, saved_client_opts):
                    c_opt.load_state_dict(opt_state)
        best_balanced_acc = ckpt["best_balanced_acc"]
        start_round = ckpt["round"] + 1
        _restore_rng_states(ckpt["rng_states"], client_loaders, centralized_loader)
        _truncate_csv(csv_path, ckpt["round"])
        print(
            f"[Resume] {last_ckpt_path} 로드 완료 → Round {start_round}부터 재개 "
            f"(현재 Best Val Balanced Acc: {best_balanced_acc:.4f})"
        )
        if start_round > args.rounds:
            # 학습은 이미 끝났으므로 루프는 건너뛰고, --final_test가 지정됐다면 아래에서 test 평가만 수행
            print(f"[Resume] 이미 {ckpt['round']} 라운드까지 완료된 실험입니다. 학습을 건너뜁니다.")
    else:
        if args.resume:
            print("[Resume] 기존 체크포인트가 없어 처음부터 학습합니다.")
        elif os.path.exists(last_ckpt_path):
            print(f"[Warning] 기존 체크포인트({last_ckpt_path})를 무시하고 처음부터 학습합니다. 이어서 학습하려면 --resume을 사용하세요.")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    # 6. 라운드별 학습 및 평가 루프
    for r in range(start_round, args.rounds + 1):
        # 라운드별 학습률 (constant면 항상 args.lr). Adam의 모멘텀 상태는 그대로 두고 lr만 갱신한다.
        round_learning_rate = round_lr(
            args.lr, r, args.rounds, args.lr_schedule, args.lr_min_ratio
        )
        for opt in ([central_optimizer] if central_optimizer is not None else client_optimizers):
            for group in opt.param_groups:
                group["lr"] = round_learning_rate

        if args.method == "centralized":
            global_model, train_loss = train_one_client(
                model=global_model,
                dataloader=centralized_loader,
                criterion=criterion,
                lr=args.lr,
                local_epochs=args.local_epochs,
                device=device,
                mu=0.0,
                global_model=None,
                optimizer=central_optimizer,
                num_updates=None,
            )
            # Centralized는 단일 글로벌 모델만 존재하므로 클라이언트 간 가중치 발산(Client Drift)은 0.0
            avg_divergence = 0.0
        else:
            local_models = []
            local_losses = []
            divergences = []

            for c_model, c_opt, step_iter in zip(
                client_models, client_optimizers, client_step_iterators
            ):
                # FLamby _Model과 동일하게 로컬 모델 파라미터에 현재 글로벌 가중치를 인플레이스 동기화하되,
                # 파라미터 텐서 인스턴스와 로컬 옵티마이저(Adam 모멘텀 상태)는 라운드 간 그대로 유지
                c_model.load_state_dict(global_model.state_dict())
                trained_local_model, c_loss = train_one_client(
                    model=c_model,
                    dataloader=step_iter,
                    criterion=criterion,
                    lr=args.lr,
                    local_epochs=args.local_epochs,
                    device=device,
                    mu=effective_mu,
                    global_model=global_model,
                    optimizer=c_opt,
                    num_updates=effective_num_updates,
                )

                div = compute_weight_divergence(global_model, trained_local_model)
                local_models.append(trained_local_model)
                local_losses.append(c_loss)
                divergences.append(div)

            # 샘플 수 비례 가중 평균으로 글로벌 모델 업데이트 및 평균 손실(BaselineLoss 기준) 계산
            total_samples = sum(client_sample_counts)
            train_loss = sum(
                l * n for l, n in zip(local_losses, client_sample_counts)
            ) / max(total_samples, 1)
            avg_divergence = sum(divergences) / max(len(divergences), 1)

            global_model = server_aggregate(
                global_model=global_model,
                client_models=local_models,
                client_sample_counts=client_sample_counts,
            )

        # 글로벌 검증셋 및 클라이언트별 검증셋 평가 (test는 학습 종료 후 --final_test로 1회만 평가)
        eval_metrics = evaluate_model(
            model=global_model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
        )
        val_loss = eval_metrics["loss"]
        balanced_acc = eval_metrics["balanced_acc"]
        macro_f1 = eval_metrics["macro_f1"]
        macro_f1_fixed8 = eval_metrics["macro_f1_fixed8"]
        ece = eval_metrics["ece"]
        max_macro_f1 = eval_metrics["max_macro_f1"]
        min_macro_f1 = eval_metrics["min_macro_f1"]
        macro_f1_gap = eval_metrics["macro_f1_gap"]
        macro_f1_std = eval_metrics["macro_f1_std"]
        max_bal_acc = eval_metrics["max_bal_acc"]
        min_bal_acc = eval_metrics["min_bal_acc"]
        bal_acc_gap = eval_metrics["bal_acc_gap"]
        bal_acc_std = eval_metrics["bal_acc_std"]
        client_bal_acc = eval_metrics["client_balanced_acc"]
        client_f1 = eval_metrics["client_macro_f1"]
        client_f1_fixed8 = eval_metrics["client_macro_f1_fixed8"]
        cumulative_comm_mb = r * round_comm_mb

        print(
            f"[Round {r:03d}/{args.rounds:03d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Bal Acc: {balanced_acc:.4f} (Std: {bal_acc_std:.4f}, Gap: {bal_acc_gap:.4f}) | "
            f"Macro F1: {macro_f1:.4f} (Std: {macro_f1_std:.4f}, Gap: {macro_f1_gap:.4f}) | "
            f"ECE: {ece:.4f} | Weight Div: {avg_divergence:.4f} | LR: {round_learning_rate:.2e} | "
            f"Comm: {cumulative_comm_mb:.1f} MB"
        )

        # 매 라운드 결과를 실시간으로 CSV 파일에 기록 (flush 보장)
        row_data = {
            "round": r,
            "method": args.method,
            "mu": effective_mu,
            "seed": args.seed,
            "eval_split": "val",
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "balanced_acc": round(balanced_acc, 6),
            "macro_f1": round(macro_f1, 6),
            "macro_f1_fixed8": round(macro_f1_fixed8, 6),
            "ece": round(ece, 6),
            "max_macro_f1": round(max_macro_f1, 6),
            "min_macro_f1": round(min_macro_f1, 6),
            "macro_f1_gap": round(macro_f1_gap, 6),
            "macro_f1_std": round(macro_f1_std, 6),
            "max_bal_acc": round(max_bal_acc, 6),
            "min_bal_acc": round(min_bal_acc, 6),
            "bal_acc_gap": round(bal_acc_gap, 6),
            "bal_acc_std": round(bal_acc_std, 6),
            "weight_divergence": round(avg_divergence, 6),
            "cumulative_comm_mb": round(cumulative_comm_mb, 2),
            "lr": round_learning_rate,
        }
        for k in range(NUM_CLIENTS):
            row_data[f"balanced_acc_client{k}"] = round(client_bal_acc[k], 6)
            row_data[f"macro_f1_client{k}"] = round(client_f1[k], 6)
            row_data[f"macro_f1_fixed8_client{k}"] = round(client_f1_fixed8[k], 6)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader() if os.path.getsize(csv_path) == 0 else None
            writer.writerow(row_data)
            f.flush()

        # 최고 balanced_acc 달성 시 .pt 가중치 저장
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            _atomic_torch_save(
                {
                    "round": r,
                    "config": run_config,
                    "balanced_acc": balanced_acc,
                    "macro_f1": macro_f1,
                    "ece": ece,
                    "max_macro_f1": max_macro_f1,
                    "min_macro_f1": min_macro_f1,
                    "macro_f1_gap": macro_f1_gap,
                    "macro_f1_std": macro_f1_std,
                    "max_bal_acc": max_bal_acc,
                    "min_bal_acc": min_bal_acc,
                    "bal_acc_gap": bal_acc_gap,
                    "bal_acc_std": bal_acc_std,
                    "cumulative_comm_mb": cumulative_comm_mb,
                    "model_state_dict": global_model.state_dict(),
                },
                best_model_path,
            )
            print(
                f"  -> [*] Best Val Balanced Acc 갱신 ({best_balanced_acc:.4f})! 가중치 저장 완료: {best_model_path}"
            )

        # 매 라운드 종료 시 resume용 last 체크포인트 저장 (모델 + 옵티마이저 + 난수 상태)
        _atomic_torch_save(
            {
                "round": r,
                "config": run_config,
                "best_balanced_acc": best_balanced_acc,
                "model_state_dict": global_model.state_dict(),
                "optimizer_state_dict": (
                    central_optimizer.state_dict() if central_optimizer is not None else None
                ),
                "client_optimizer_state_dicts": (
                    [c_opt.state_dict() for c_opt in client_optimizers]
                    if client_optimizers is not None
                    else None
                ),
                "rng_states": _capture_rng_states(client_loaders, centralized_loader),
            },
            last_ckpt_path,
        )

    print(
        f"[Done] 학습 완료. 최고 Val Balanced Acc: {best_balanced_acc:.4f} | 결과 파일: {csv_path}"
    )

    if args.final_test:
        _run_final_test(
            args=args,
            run_config=run_config,
            exp_tag=exp_tag,
            global_model=global_model,
            best_model_path=best_model_path,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )


if __name__ == "__main__":
    main()
