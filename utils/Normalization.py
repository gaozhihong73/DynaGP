import os
import os
import pickle
from typing import Dict, Union

import numpy as np
import torch


def save_scalers(scalers, save_path):
    """
    保存sklearn的scaler对象到文件

    Args:
        scalers: scaler字典
        save_path: 保存路径(.pkl格式)
    """
    dir = os.path.dirname(save_path)
    if dir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scalers, f)


def load_scalers(load_path):
    """
    从文件加载sklearn的scaler对象

    Args:
        load_path: scaler文件路径(.pkl格式)

    Returns:
        scalers: scaler字典
    """
    with open(load_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def apply_global_norm(train, val, test, phenotype_names, ScalerClass):
    """
    策略1: 按照整个表型归一化 (Global per Phenotype)
    逻辑: 将该表型下所有时间点的数据视为同一个分布，计算一个全局的 mean/std。
    """
    scalers_dict = {}

    # 复制数据以防修改原始引用
    train_norm = train.copy()
    val_norm = val.copy()
    test_norm = test.copy()

    for i, name in enumerate(phenotype_names):
        # 1. 准备数据: 展平所有时间点 [N, T] -> [N*T, 1]
        # 这样 Scaler 就会认为只有一个特征，计算出一个全局均值和方差
        train_flat = train[:, i, :].reshape(-1, 1)

        # 2. 拟合
        scaler = ScalerClass()
        scaler.fit(train_flat)
        scalers_dict[name] = scaler

        # 3. 变换辅助函数
        def transform_block(data):
            B, P, T = data.shape
            # 提取该表型的数据并展平
            flat = data[:, i, :].reshape(-1, 1)
            norm = scaler.transform(flat)
            # 恢复形状 [B, T]
            return norm.reshape(B, T)

        # 4. 应用变换
        train_norm[:, i, :] = transform_block(train)
        val_norm[:, i, :] = transform_block(val)
        test_norm[:, i, :] = transform_block(test)

    return train_norm, val_norm, test_norm, {'scalers': scalers_dict}


def apply_timepoint_norm(train, val, test, phenotype_names, ScalerClass):
    """
    策略2 (优化版): 按表型中的每个时间点归一化 (Per Timepoint)
    逻辑: 利用 sklearn 的向量化特性，直接对 [N, T] 矩阵拟合，
          自动为每一列(时间点)计算独立的 mean/std。
    """
    scalers_dict = {}

    train_norm = train.copy()
    val_norm = val.copy()
    test_norm = test.copy()

    for i, name in enumerate(phenotype_names):
        # 1. 提取该表型的矩阵 [N, T]
        # 注意：这里直接使用矩阵，每一列代表一个时间点
        train_matrix = train[:, i, :]
        val_matrix = val[:, i, :]
        test_matrix = test[:, i, :]

        # 2. 拟合
        scaler = ScalerClass()
        # sklearn 会按列计算统计量，因此 fit 后 scaler 内部存有 T 组 mean/std
        scaler.fit(train_matrix)
        scalers_dict[name] = scaler

        # 3. 变换 (一次性处理整个矩阵)
        train_norm[:, i, :] = scaler.transform(train_matrix)
        val_norm[:, i, :] = scaler.transform(val_matrix)
        test_norm[:, i, :] = scaler.transform(test_matrix)

    return train_norm, val_norm, test_norm, {'scalers': scalers_dict}


def apply_residual_global_norm(train, val, test, phenotype_names, ScalerClass):
    """
    策略3: 先减去均值再按照整个表型归一化 (Residual + Global)
    """
    # 1. 计算均值曲线 (只用训练集) [1, P, T]
    mean_curve = np.mean(train, axis=0, keepdims=True)

    # 2. 计算残差
    # 利用广播机制: [N, P, T] - [1, P, T]
    train_res = train - mean_curve
    val_res = val - mean_curve
    test_res = test - mean_curve

    # 3. 对残差应用 Global 归一化
    # 复用 apply_global_norm
    t_n, v_n, te_n, meta = apply_global_norm(train_res, val_res, test_res, phenotype_names, ScalerClass)

    # 4. 补充均值曲线到元数据中
    meta['mean_curve'] = mean_curve
    return t_n, v_n, te_n, meta


# ==============================================================================
# 4. 统一的反归一化接口 (更新版)
# ==============================================================================
def denormalize_phenotype(normalized_data: Union[np.ndarray, torch.Tensor],
                          scalers_package: Dict) -> torch.Tensor:
    """
    自动根据 scalers_package 中的 'norm_method' 选择正确的反归一化路径。
    支持输入形状: [B, P, T] (批量) 或 [P, T] (单样本)。
    """
    # 1. 类型转换与形状检查
    if isinstance(normalized_data, torch.Tensor):
        normalized_data = normalized_data.detach().cpu().numpy()

    # 处理单样本情况: [P, T] -> [1, P, T]
    original_ndim = normalized_data.ndim
    if original_ndim == 2:
        normalized_data = normalized_data[np.newaxis, ...]

    method = scalers_package['norm_method']
    meta = scalers_package['pheno_meta']
    names = scalers_package['phenotype_names']

    B, P, T = normalized_data.shape

    # 这里的 restored 将存储反归一化后的“数值”（可能是原始值，也可能是残差）
    restored = np.zeros_like(normalized_data)

    # === 反归一化逻辑 ===

    def _inv_global(data, pheno_scalers):
        """处理 Global Norm 的反变换"""
        out = np.zeros_like(data)
        for i, name in enumerate(names):
            scaler = pheno_scalers[name]
            # [B, T] -> flatten [B*T, 1] -> inverse -> reshape [B, T]
            flat = data[:, i, :].reshape(-1, 1)
            out[:, i, :] = scaler.inverse_transform(flat).reshape(B, T)
        return out

    def _inv_timepoint(data, pheno_scalers):
        """处理 Timepoint Norm 的反变换 (适配优化后的向量化Scaler)"""
        out = np.zeros_like(data)
        for i, name in enumerate(names):
            scaler = pheno_scalers[name]
            # [B, T] -> inverse_transform -> [B, T]
            # 这里的 scaler 已经包含了 T 个特征的统计量，直接传入矩阵即可
            out[:, i, :] = scaler.inverse_transform(data[:, i, :])
        return out

    # 1. 核心数值恢复 (Inverse Scaling)
    if 'timepoint' in method:  # 覆盖 'timepoint' 和 'residual_timepoint'
        # 这里的 scaler 结构现在是 {name: scaler_obj}，不再是嵌套字典
        temp_data = _inv_timepoint(normalized_data, meta['scalers'])
    else:  # 'global' 和 'residual_global'
        temp_data = _inv_global(normalized_data, meta['scalers'])

    # 2. 均值恢复 (Add Mean Curve)
    if 'residual' in method:
        mean_curve = meta['mean_curve']  # [1, P, T]
        # 广播加法: [B, P, T] + [1, P, T]
        temp_data = temp_data + mean_curve

    restored = temp_data

    # 如果输入是单样本，输出也降维回去
    if original_ndim == 2:
        restored = restored.squeeze(0)

    return torch.tensor(restored, dtype=torch.float32)
