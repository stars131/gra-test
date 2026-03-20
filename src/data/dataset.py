"""
PyTorch Dataset classes for network attack detection.
Supports single-source and multi-source data loading.
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union


class NetworkAttackDataset(Dataset):
    """
    单源网络攻击检测数据集

    用于加载和提供单一数据源的网络流量数据。
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        初始化数据集

        Args:
            features: 特征数组 (N, D)
            labels: 标签数组 (N,)
            transform: 可选的数据转换函数
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))


class MultiSourceDataset(Dataset):
    """
    多源网络攻击检测数据集

    用于加载和提供多个数据源（如流量特征 + 时序特征）的数据。
    支持注意力融合模型的训练。
    """

    def __init__(
        self,
        source1_features: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
        source2_or_labels: np.ndarray,
        labels: Optional[np.ndarray] = None,
        source_names: Optional[List[str]] = None,
        source1_name: str = "traffic",
        source2_name: str = "temporal",
        transform1: Optional[callable] = None,
        transform2: Optional[callable] = None,
        transforms: Optional[List[Optional[callable]]] = None
    ):
        """
        初始化多源数据集

        Args:
            source1_features: 第一数据源特征，或全部数据源特征列表
            source2_or_labels: 第二数据源特征 (旧接口) 或标签 (新接口)
            labels: 标签数组 (旧接口)
            source_names: 数据源名称列表
            source1_name/source2_name: 兼容旧接口
            transform1/transform2: 兼容旧接口
            transforms: 各数据源的转换函数列表
        """
        if isinstance(source1_features, (list, tuple)):
            if not source1_features:
                raise ValueError("至少需要一个数据源特征数组")
            arrays = [torch.FloatTensor(features) for features in source1_features]
            label_array = source2_or_labels
        else:
            if labels is None:
                raise ValueError("旧接口至少需要 source1_features 和 source2_features")
            arrays = [
                torch.FloatTensor(source1_features),
                torch.FloatTensor(source2_or_labels),
            ]
            label_array = labels

        self.sources = arrays
        self.labels = torch.LongTensor(label_array)
        self.source_names = source_names or [source1_name, source2_name]
        if len(self.source_names) != len(self.sources):
            self.source_names = [f"source{i}" for i in range(1, len(self.sources) + 1)]

        self.transforms = transforms or [transform1, transform2]
        if len(self.transforms) != len(self.sources):
            self.transforms = [
                self.transforms[i] if i < len(self.transforms) else None
                for i in range(len(self.sources))
            ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        获取数据项

        Returns:
            (*source_features, label)
        """
        features = []
        for source, transform in zip(self.sources, self.transforms):
            tensor = source[idx]
            if transform:
                tensor = transform(tensor)
            features.append(tensor)
        y = self.labels[idx]

        return (*features, y)

    @property
    def source1_dim(self) -> int:
        return self.sources[0].shape[1]

    @property
    def source2_dim(self) -> int:
        return self.sources[1].shape[1] if len(self.sources) > 1 else 0

    @property
    def source_dims(self) -> List[int]:
        return [source.shape[1] for source in self.sources]

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))


class DataTransforms:
    """
    数据转换工具类

    提供各种数据增强和转换方法。
    """

    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    @staticmethod
    def random_dropout(tensor: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """随机置零（特征dropout）"""
        mask = torch.rand_like(tensor) > p
        return tensor * mask

    @staticmethod
    def feature_scaling(tensor: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """随机特征缩放"""
        scale = torch.empty(tensor.shape).uniform_(*scale_range)
        return tensor * scale


class TrainingTransform:
    """
    训练时的数据增强组合

    可以组合多种增强方法。
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        dropout_p: float = 0.1,
        use_noise: bool = True,
        use_dropout: bool = False
    ):
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.use_noise = use_noise
        self.use_dropout = use_dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_noise and self.noise_std > 0:
            x = DataTransforms.add_gaussian_noise(x, self.noise_std)
        if self.use_dropout and self.dropout_p > 0:
            x = DataTransforms.random_dropout(x, self.dropout_p)
        return x


def create_data_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    创建单源数据的DataLoader

    Args:
        data_dict: 包含训练/验证/测试数据的字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_weighted_sampler: 是否使用加权采样（处理类不平衡）
        augment_train: 是否对训练数据进行增强

    Returns:
        包含train/val/test DataLoader的字典
    """
    # 训练数据增强
    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None

    # 创建数据集
    train_dataset = NetworkAttackDataset(
        data_dict['X_train'],
        data_dict['y_train'],
        transform=train_transform
    )
    val_dataset = NetworkAttackDataset(
        data_dict['X_val'],
        data_dict['y_val']
    )
    test_dataset = NetworkAttackDataset(
        data_dict['X_test'],
        data_dict['y_test']
    )

    # 加权采样器（处理类不平衡）
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict['y_train'])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict['y_train']]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False

    # 创建DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    print(f"\nDataLoader 创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(loaders['train'])} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(loaders['val'])} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(loaders['test'])} 批次")

    return loaders


def create_multi_source_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    augment_train: bool = False
) -> Dict[str, DataLoader]:
    """
    创建多源数据的DataLoader

    Args:
        data_dict: 包含多源训练/验证/测试数据的字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_weighted_sampler: 是否使用加权采样
        augment_train: 是否对训练数据进行增强

    Returns:
        包含train/val/test DataLoader的字典
    """
    source_indices = sorted({
        int(match.group(1))
        for key in data_dict.keys()
        for match in [re.match(r'^X(\d+)_train$', key)]
        if match is not None
    })

    if source_indices and len(source_indices) < 2:
        raise ValueError("至少需要两个数据源（X1_* 和 X2_*）")

    if source_indices:
        for idx in source_indices:
            for split in ('train', 'val', 'test'):
                key = f'X{idx}_{split}'
                if key not in data_dict:
                    raise KeyError(f"缺少数据键: {key}")

    # 训练数据增强
    train_transform = TrainingTransform(noise_std=0.01) if augment_train else None

    source_indices = source_indices or [1, 2]
    train_sources = [data_dict[f'X{idx}_train'] for idx in source_indices]
    val_sources = [data_dict[f'X{idx}_val'] for idx in source_indices]
    test_sources = [data_dict[f'X{idx}_test'] for idx in source_indices]
    source_names = data_dict.get('source_names') or [f"source{idx}" for idx in source_indices]

    # 创建数据集
    train_dataset = MultiSourceDataset(
        train_sources,
        data_dict['y_train'],
        source_names=source_names,
        transforms=[train_transform] * len(train_sources),
    )
    val_dataset = MultiSourceDataset(
        val_sources,
        data_dict['y_val']
    )
    test_dataset = MultiSourceDataset(
        test_sources,
        data_dict['y_test']
    )

    # 加权采样器
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        class_counts = np.bincount(data_dict['y_train'])
        weights = 1.0 / class_counts
        sample_weights = weights[data_dict['y_train']]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False

    # 创建DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    print(f"\n多源 DataLoader 创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(loaders['train'])} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(loaders['val'])} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(loaders['test'])} 批次")
    print(f"  数据源数量: {train_dataset.num_sources}")
    for idx, dim in enumerate(train_dataset.source_dims, start=1):
        print(f"  Source {idx} 维度: {dim}")

    return loaders


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    计算类别权重（用于损失函数）

    Args:
        labels: 标签数组

    Returns:
        类别权重张量
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """
    计算样本权重（用于加权采样）

    Args:
        labels: 标签数组

    Returns:
        样本权重数组
    """
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    return weights[labels]
