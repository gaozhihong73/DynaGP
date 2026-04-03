from pathlib import Path
from typing import Dict, Tuple
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import apply_global_norm, apply_residual_global_norm, apply_timepoint_norm, denormalize_phenotype


class TFPPDataset:
    """SNP和表型数据集类"""

    def __init__(self,
                 snp_file: str,
                 pheno_dir: str,
                 phenotype_names: List[str]):
        """
        初始化数据集
        Args:
            snp_file: SNP数据文件路径(.npy格式)
            phenotype_names: 选择的表型名称列表,例如['RGBVI', 'ExB', 'WI', 'CIVE']
        """
        if not Path(snp_file).exists():
            raise FileNotFoundError(f"SNP文件不存在: {snp_file}")

        # 加载SNP数据
        self.snp = np.load(snp_file) + 1  # [400, n_feature]

        self.pheno_dir = pheno_dir
        self.phenotype_names = phenotype_names

        # 加载表型数据(未归一化)
        self.phenotype = self.load_phenotypes()  # [400, 表型数量, 14]

    def load_phenotypes(self):
        """
        加载并合并所有表型数据(不进行归一化)
        phenotype_names 表型名称

        Returns:
            phenotype_data: numpy数组,形状为[400, 表型数量, 14]
        """
        phenos = []

        # 遍历每个表型文件
        for phenotype_name in self.phenotype_names:
            pheno = pd.read_csv(Path(f"{self.pheno_dir}/{phenotype_name}.csv"))
            pheno = pheno.values[:, np.newaxis, :].astype(np.float32)
            phenos.append(pheno)

        # 合并所有表型数据: [400, 表型数量, 14]
        pheno_data = np.concatenate(phenos, axis=1)
        return pheno_data

    def __len__(self):
        """返回数据集中的样本数量"""
        return self.snp.shape[0]

    def __getitem__(self, idx):
        """获取单个样本"""
        return self.snp[idx], self.phenotype[idx]


def get_dataloader(dataset,
                   batch_size: int,
                   norm_method: str,
                   scaler_type: str,
                   test_ratio: float = 0.1,
                   val_ratio: float = 0.1,
                   seed: int = 73,
                   device: str = 'cpu',
                   num_workers: int = 4
                   ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Args:
        norm_method: 归一化策略选择
            - 'global': 整个表型统一归一化
            - 'timepoint': 每个时间点独立归一化
            - 'residual_global': 减均值 -> 整个表型归一化
            - 'residual_timepoint': 减均值 -> 每个时间点独立归一化
        scaler_type: 缩放器类型 ('zscore', 'minmax', 'robust'). 默认 'zscore'.
    """
    # ===================== 第一步: 划分数据集 =====================
    snp_data = dataset.snp  # [400, n_feature]
    # snp_data = np.random.randint(-1, 3, snp_data.shape)  # 随机数验证
    phenotype_data = dataset.phenotype  # [400, 表型数量, 14]

    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed)
    val_adj = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_adj, random_state=seed)

    train_snp, train_phenotype = snp_data[train_idx], phenotype_data[train_idx]
    val_snp, val_phenotype = snp_data[val_idx], phenotype_data[val_idx]
    test_snp, test_phenotype = snp_data[test_idx], phenotype_data[test_idx]

    # 控制样本数量实验
    # train_snp = train_snp[:200]
    # train_phenotype = train_phenotype[:200]

    # 核心归一化策略函数库
    def _get_scaler_cls(scaler_type: str):
        """根据字符串返回Scaler类"""
        if scaler_type == 'zscore':
            return StandardScaler
        elif scaler_type == 'minmax':
            return MinMaxScaler
        elif scaler_type == 'robust':
            return RobustScaler
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    # 归一化函数
    ScalerClass = _get_scaler_cls(scaler_type)

    # 归一化策略
    norm_funcs = {
        'global':          apply_global_norm,
        'timepoint':       apply_timepoint_norm,
        'residual_global': apply_residual_global_norm,
        }

    if norm_method not in norm_funcs:
        raise ValueError(f"Unknown norm_method: {norm_method}. Choose from {list(norm_funcs.keys())}")

    # 调用对应的处理函数
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = norm_funcs[norm_method](
        train_phenotype, val_phenotype, test_phenotype, dataset.phenotype_names, ScalerClass
        )

    # 封装 DataLoaders
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    train_dataset = TensorDataset(to_tensor(train_snp), to_tensor(train_phe_norm))
    val_dataset = TensorDataset(to_tensor(val_snp), to_tensor(val_phe_norm))
    test_dataset = TensorDataset(to_tensor(test_snp), to_tensor(test_phe_norm))

    pin_memory = (device != "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # 5. 构建 Scalers 信息包
    scalers = {
        'phenotype_names': dataset.phenotype_names,
        'norm_method':     norm_method,  # 关键：记录当前用了什么方法
        'pheno_meta':      pheno_meta  # 包含 scalers 字典和 mean_curve
        }

    return train_loader, val_loader, test_loader, scalers


def get_dataloader_with_kfold(train_indices: List,
                              val_indices: List,
                              test_indices: List,
                              dataset: TFPPDataset,
                              batch_size: int,
                              device: str,
                              norm_method: str,
                              scaler_type: str,
                              num_workers: int = 4
                              ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    使用sklearn创建训练集、验证集和测试集的DataLoader
    关键改进: 每个表型独立归一化,避免混淆不同表型的统计特性

    Args:
        dataset: TFPPDataset实例
        batch_size: 批次大小
        device: 训练设备('cpu'或'cuda')
        num_workers: 数据加载的工作进程数
    """
    # ===================== 第一步: 划分数据集 =====================
    snp_data = dataset.snp
    phenotype_data = dataset.phenotype

    # 提取各数据集的SNP和表型数据
    train_snp, train_phenotype = snp_data[train_indices], phenotype_data[train_indices]
    val_snp, val_phenotype = snp_data[val_indices], phenotype_data[val_indices]
    test_snp, test_phenotype = snp_data[test_indices], phenotype_data[test_indices]

    # 控制样本数量实验
    # train_snp = train_snp[:200]
    # train_phenotype = train_phenotype[:200]

    # 核心归一化策略函数库
    def _get_scaler_cls(scaler_type: str):
        """根据字符串返回Scaler类"""
        if scaler_type == 'zscore':
            return StandardScaler
        elif scaler_type == 'minmax':
            return MinMaxScaler
        elif scaler_type == 'robust':
            return RobustScaler
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    # 归一化函数
    ScalerClass = _get_scaler_cls(scaler_type)
    # 归一化策略
    norm_funcs = {
        'global':          apply_global_norm,
        'timepoint':       apply_timepoint_norm,
        'residual_global': apply_residual_global_norm,
        }

    if norm_method not in norm_funcs:
        raise ValueError(f"Unknown norm_method: {norm_method}. Choose from {list(norm_funcs.keys())}")

    # 调用对应的处理函数
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = norm_funcs[norm_method](
        train_phenotype, val_phenotype, test_phenotype, dataset.phenotype_names, ScalerClass
        )

    # 封装 DataLoaders
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    train_dataset = TensorDataset(to_tensor(train_snp), to_tensor(train_phe_norm))
    val_dataset = TensorDataset(to_tensor(val_snp), to_tensor(val_phe_norm))
    test_dataset = TensorDataset(to_tensor(test_snp), to_tensor(test_phe_norm))

    pin_memory = (device != "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
        )

    scalers = {
        'phenotype_names': dataset.phenotype_names,
        'norm_method':     norm_method,  # 关键：记录当前用了什么方法
        'pheno_meta':      pheno_meta  # 包含 scalers 字典和 mean_curve
        }
    return train_loader, val_loader, test_loader, scalers
