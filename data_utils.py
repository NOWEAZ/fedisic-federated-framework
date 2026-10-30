import inspect
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    BaselineLoss,
    FedIsic2019,
    get_nb_max_rounds,
)


def _seed_worker(worker_id: int):
    """DataLoader 멀티프로세싱 워커의 난수 시드를 고정합니다."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _extract_labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """Dataset(또는 ConcatDataset, Subset)에서 이미지 디코딩 없이 빠르게 레이블 배열을 추출합니다."""
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([_extract_labels_from_dataset(d) for d in dataset.datasets])

    if isinstance(dataset, Subset):
        base_labels = _extract_labels_from_dataset(dataset.dataset)
        return base_labels[np.asarray(dataset.indices, dtype=np.int64)]

    # FLamby FedIsic2019 내부 속성 확인 (버전별 호환)
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            vals = getattr(dataset, attr)
            return np.asarray(vals, dtype=np.int64).reshape(-1)

    if hasattr(dataset, "dic") and isinstance(dataset.dic, dict) and "target" in dataset.dic:
        return np.asarray(dataset.dic["target"], dtype=np.int64).reshape(-1)

    # Fallback: 데이터셋 직접 순회
    labels = []
    for _, y in dataset:
        if isinstance(y, torch.Tensor):
            labels.append(int(y.item()))
        else:
            labels.append(int(y))
    return np.asarray(labels, dtype=np.int64)


def stratified_train_val_split(
    labels,
    val_fraction: float = 0.1,
    split_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    클래스 비율을 유지하며(층화) 인덱스를 train/val로 나눕니다.

    - 클래스마다 round(n_c * val_fraction)장을 val로 보냅니다 (샘플이 2장 이상인 클래스는 최소 1장).
    - 샘플이 1장뿐인 클래스는 train에만 남기고, 모든 클래스는 train에 최소 1장을 유지합니다.
    - 같은 (labels, val_fraction, split_seed)이면 항상 같은 결과를 냅니다 (실험 시드와 무관하게 고정 가능).

    Returns:
        tuple: (train_indices, val_indices), 각각 오름차순 정렬된 int64 배열
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction은 0과 1 사이여야 합니다: {val_fraction}")

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.RandomState(split_seed)

    train_parts, val_parts = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        if len(idx) >= 2:
            n_val = max(n_val, 1)
        n_val = min(n_val, len(idx) - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return train_idx, val_idx


def _build_center_datasets(
    center: int,
    flamby_kwargs: dict,
    val_fraction: float,
    split_seed: int,
):
    """
    한 센터의 (학습 Subset, 검증 Subset, 테스트 Dataset)을 만듭니다.

    FLamby의 train 데이터를 층화로 train/val로 나눕니다. 검증용 뷰는 학습용 증강(무작위 변환)을
    쓰지 않도록 test와 같은 결정론적 변환(CenterCrop + Normalize)으로 교체합니다.
    분할 시드는 센터마다 다르게(split_seed + center) 주되, 실험 시드와는 무관하게 고정됩니다.
    """
    train_full = FedIsic2019(center=center, train=True, pooled=False, **flamby_kwargs)
    test_ds = FedIsic2019(center=center, train=False, pooled=False, **flamby_kwargs)
    val_view = FedIsic2019(center=center, train=True, pooled=False, **flamby_kwargs)
    val_view.augmentations = test_ds.augmentations

    labels = _extract_labels_from_dataset(train_full)
    train_idx, val_idx = stratified_train_val_split(labels, val_fraction, split_seed + center)
    return Subset(train_full, train_idx), Subset(val_view, val_idx), test_ds


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    data_path: str | None = None,
    debug: bool = False,
    seed: int | None = None,
    val_fraction: float = 0.1,
    split_seed: int = 0,
):
    """
    FLamby의 FedIsic2019 공식 6개 센터(Center 0 ~ 5) 데이터셋을 사용해
    6개 클라이언트별 DataLoader, Centralized 통합 DataLoader,
    Global Validation/Test DataLoader를 생성합니다.

    각 센터의 FLamby train 데이터를 층화로 (1 - val_fraction) : val_fraction 으로 나눠
    학습/검증으로 쓰고, FLamby 공식 test는 최종 평가에만 사용합니다.

    Args:
        batch_size: 미니배치 크기 (FLamby 공식 기본값: BATCH_SIZE = 64)
        num_workers: DataLoader 워커 프로세스 수
        data_path: FLamby 데이터 경로 (None이면 FLamby 기본 설정 사용)
        debug: FLamby debug 모드 여부
        seed: 재현성을 위한 난수 시드 (None이면 고정하지 않음). 학습 셔플/증강에만 영향
        val_fraction: 센터별 train에서 떼어 낼 검증 비율 (0 < val_fraction < 1)
        split_seed: train/val 분할 시드. 실험 시드와 독립적으로 고정되어 모든 실행이 같은 val을 사용

    Returns:
        tuple: (client_loaders, centralized_loader, val_loader, test_loader)
            - client_loaders ( list[DataLoader] ): 6개 클라이언트(센터 0~5)의 학습용 DataLoader 리스트
            - centralized_loader ( DataLoader ): 전체 6개 센터 학습 데이터를 하나로 묶은 통합 DataLoader
            - val_loader ( DataLoader ): 글로벌 검증용 통합 DataLoader (매 라운드 평가·모델/mu 선택용)
            - test_loader ( DataLoader ): 글로벌 최종 평가용 통합 Test DataLoader (마지막에 1회만 사용)
            val_loader와 test_loader에는 6개 클라이언트별 평가 로더 리스트가 `client_eval_loaders` 속성으로
            연결되어 있습니다.
    """
    flamby_kwargs = {}
    if "debug" in inspect.signature(FedIsic2019.__init__).parameters:
        flamby_kwargs["debug"] = debug
    if data_path is not None:
        flamby_kwargs["data_path"] = data_path

    client_datasets = []
    client_val_datasets = []
    client_test_datasets = []
    for center in range(NUM_CLIENTS):
        train_ds, val_ds, test_ds = _build_center_datasets(
            center, flamby_kwargs, val_fraction, split_seed
        )
        client_datasets.append(train_ds)
        client_val_datasets.append(val_ds)
        client_test_datasets.append(test_ds)

    pin_memory = torch.cuda.is_available()
    worker_init_fn = _seed_worker if seed is not None else None

    # 1. 6개 클라이언트(센터 0~5) 학습 DataLoader
    client_loaders = []
    for idx, ds in enumerate(client_datasets):
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed + idx)
        client_loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                worker_init_fn=worker_init_fn,
                generator=gen,
            )
        )

    # 2. Centralized 통합 학습 DataLoader
    centralized_dataset = ConcatDataset(client_datasets)
    central_gen = None
    if seed is not None:
        central_gen = torch.Generator()
        central_gen.manual_seed(seed + 100)
    centralized_loader = DataLoader(
        centralized_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=central_gen,
    )

    def _make_eval_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # 3. Global Validation/Test DataLoader (전체 6개 센터 통합) 및 클라이언트별 평가 DataLoader
    # 클라이언트 간 최고-최저 Macro-F1(공정성 지표) 계산을 위해 클라이언트별 로더를 속성으로 연결
    val_loader = _make_eval_loader(ConcatDataset(client_val_datasets))
    val_loader.client_eval_loaders = [_make_eval_loader(ds) for ds in client_val_datasets]

    test_loader = _make_eval_loader(ConcatDataset(client_test_datasets))
    test_loader.client_eval_loaders = [_make_eval_loader(ds) for ds in client_test_datasets]
    test_loader.client_test_loaders = test_loader.client_eval_loaders

    return client_loaders, centralized_loader, val_loader, test_loader
