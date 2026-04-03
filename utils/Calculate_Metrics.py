from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr


def compute_metrics_vectorized(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    修正版 V3：计算指标。
    R2 采用 "Time-Adjusted Global R2" (时间校正全局R2)。
    既避免了单点计算的数值爆炸，又避免了全局计算的虚高问题。
    """
    # 1. 基础误差指标 (MSE/MAE/RMSE 保持原样，按列平均)
    residuals = target - pred
    mse_per_col = np.mean(np.square(residuals), axis=0)
    mae_per_col = np.mean(np.abs(residuals), axis=0)
    rmse_per_col = np.sqrt(mse_per_col)

    # 2. PCC (保持按列平均，因为它对量纲不敏感且能反映排序能力)
    pred_mean = np.mean(pred, axis=0)
    target_mean_val = np.mean(target, axis=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean_val
    numerator = np.sum(pred_centered * target_centered, axis=0)
    denominator = np.sqrt(np.sum(np.square(pred_centered), axis=0) * np.sum(np.square(target_centered), axis=0))

    denom_zero_mask = denominator < 1e-10
    safe_denom = np.where(denom_zero_mask, 1.0, denominator)
    pcc_per_col = numerator / safe_denom
    pcc_per_col = np.where(denom_zero_mask, 0.0, pcc_per_col)

    # ---------------- [核心修正] ----------------
    # 3. R2 Score (Time-Adjusted Global R2)
    # 衡量模型解释"基因型差异"的能力，剔除"时间生长趋势"的干扰。

    # (A) 计算总残差平方和 (SS_res) - 针对所有点的预测误差求和
    # flatten() 展平所有数据
    ss_res_sum = np.sum(np.square(target - pred))

    # (B) 计算总离差平方和 (SS_tot) - 关键点！
    # 我们计算每个样本相对于"该时间点均值"的偏离，而不是相对于"全局均值"的偏离。
    # target shape: [N, T]
    # mean shape: [T] -> 利用广播机制 [N, T] - [T]
    target_mean_per_time = np.mean(target, axis=0)
    ss_tot_sum = np.sum(np.square(target - target_mean_per_time))

    if ss_tot_sum < 1e-10:
        # 如果连样本间的差异都没有（所有种子长得一模一样），则R2无意义，设为0
        r2_final = 0.0
    else:
        r2_final = 1 - (ss_res_sum / ss_tot_sum)
    # -------------------------------------------

    return {
        'mse':  float(np.nanmean(mse_per_col)),
        'rmse': float(np.nanmean(rmse_per_col)),
        'mae':  float(np.nanmean(mae_per_col)),
        'pcc':  float(np.nanmean(pcc_per_col)),
        'r2':   float(r2_final)  # 这是真正反映育种预测能力的 R2
        }


def calculate_phenotype_metrics_by_time(pred: torch.Tensor, target: torch.Tensor,
                                        selected_phenotypes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    计算每个表型按时间点计算后的平均指标。
    【优化后】：利用 compute_metrics_vectorized 直接处理矩阵，移除内层循环。
    """
    assert pred.shape == target.shape

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    phenotype_metrics = {}

    for idx, pheno_name in enumerate(selected_phenotypes):
        # 数据形状 [Batch, Time]
        pred_pheno = pred_np[:, idx, :]
        target_pheno = target_np[:, idx, :]

        # 直接传入 2D 矩阵，函数内部会沿着 axis=0 计算并取 axis=1 的均值
        metrics = compute_metrics_vectorized(pred_pheno, target_pheno)

        # 如果需要 Spearman (按时间点平均比较慢，这里可选添加)
        # 通常 Spearman 我们看 Global 的就够了，这里略过以保证速度

        phenotype_metrics[pheno_name] = metrics

    return phenotype_metrics


def calculate_phenotype_metrics(pred: torch.Tensor, target: torch.Tensor,
                                selected_phenotypes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    计算每个表型的全局指标 (Global Metrics)。
    将所有样本和时间点展平后计算。
    """
    assert pred.shape == target.shape

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    phenotype_metrics = {}

    for idx, pheno_name in enumerate(selected_phenotypes):
        # 展平: [Batch * Time]
        pred_flat = pred_np[:, idx, :].flatten()
        target_flat = target_np[:, idx, :].flatten()

        # 计算基础指标
        metrics = compute_metrics_vectorized(pred_flat, target_flat)

        # 【新增】针对 Global 数据计算 Spearman
        # 因为 flatten 后只计算一次，开销可以接受
        try:
            spearman_val, _ = spearmanr(pred_flat, target_flat)
            metrics['spearman'] = float(spearman_val)
        except:
            metrics['spearman'] = 0.0

        phenotype_metrics[pheno_name] = metrics

    return phenotype_metrics


def calculate_metrics_by_average(phenotype_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    汇总所有表型的指标。
    """
    if not phenotype_metrics:
        return {}

    # 获取第一个表型的 key 作为参考 (例如 mse, pcc, r2, spearman)
    first_key = next(iter(phenotype_metrics))
    metric_keys = phenotype_metrics[first_key].keys()

    overall_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in phenotype_metrics.values()]
        overall_metrics[key] = float(np.mean(values))

    return overall_metrics
