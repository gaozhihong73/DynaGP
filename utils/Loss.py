from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(nn.Module):
    def __init__(self, diversity_weight: float = 0.1, target_std: float = 1.0, class_weights: torch.Tensor = None):
        """
        Args:
            diversity_weight: 多样性损失的权重
            target_std: 期望的最小标准差阈值 (通常设为 1.0)
            class_weights: 交叉熵损失的类别权重
        """
        super().__init__()
        self.diversity_weight = diversity_weight
        self.target_std = target_std
        self.recon_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, output, target, encoded):
        # 1. 重建损失 (主任务)
        recon_loss = self.recon_criterion(output, target)

        # 2. 改进的多样性损失 (Hinge Loss)
        # 计算当前 batch 的标准差
        encoded_std = encoded.std(dim=0).mean()

        # 逻辑：
        # 如果 std < 1.0 (例如 0.1)，则 loss = 1.0 - 0.1 = 0.9 (惩罚，迫使 std 变大)
        # 如果 std >= 1.0 (例如 1.5)，则 loss = 0 (不惩罚，也不鼓励继续变大)
        # diversity_loss = F.relu(self.target_std - encoded_std)

        # 总损失
        # total_loss = recon_loss + self.diversity_weight * diversity_loss
        total_loss = recon_loss

        return total_loss, recon_loss.item(), encoded_std.item()


class AdaptiveMultiTaskLoss(nn.Module):
    """
    自适应任务权重损失函数
    Task 1: 重建损失 (CrossEntropy)
    Task 2: 表型损失 (MSE)
    """

    def __init__(self,
                 phenotype_names: List,
                 log_var_recon: nn.Parameter,
                 log_var_pheno: nn.Parameter,
                 ):
        super().__init__()
        self.phenotype_names = phenotype_names
        self.num_phenotypes = len(phenotype_names)
        self.recon_criterion = nn.MSELoss()
        # 损失函数本身
        self.pheno_loss_fn = nn.SmoothL1Loss()
        # ------------------------------------------------------------------
        # 传入在主模型 (ConvAE) 中定义的可学习参数
        # ------------------------------------------------------------------
        self.log_var_recon = log_var_recon
        self.log_var_pheno = log_var_pheno

    def forward(self, recon_loss, phenotype_pred, phenotype_true):
        # 2. 表型损失
        pheno_loss_sum = 0.0
        for i in range(self.num_phenotypes):
            pheno_loss_sum += self.pheno_loss_fn(phenotype_pred[:, i, :], phenotype_true[:, i, :])
        avg_pheno_loss = pheno_loss_sum / self.num_phenotypes

        # 3. 自适应加权
        precision_recon = torch.exp(-self.log_var_recon)
        precision_pheno = torch.exp(-self.log_var_pheno)

        # 注意：这里去掉
        weighted_recon_loss = precision_recon * recon_loss + self.log_var_recon
        weighted_pheno_loss = precision_pheno * avg_pheno_loss + self.log_var_pheno

        total_loss = weighted_recon_loss + weighted_pheno_loss

        return total_loss, avg_pheno_loss.item()

class VariancePenaltyLoss(nn.Module):
    """
    改进版损失函数，用于解决预测方差坍缩问题。

    核心思想：
    1.  使用 MSE 损失保证预测值的基本准确性。
    2.  引入一个新的惩罚项 (deviation_penalty)，直接惩罚预测值与其均值的偏离程度。
        这会鼓励模型为不同样本输出不同的预测值，从而有效防止模型预测常数。
    3.  通过权重 `lambda_deviation` 来平衡 MSE 损失和方差惩罚的重要性。
    """

    def __init__(
            self,
            lambda_deviation: float = 0.8,  # 方差惩罚项的权重，建议从 0.8 到 0.95 开始
            reduction: str = 'mean'
            ):
        super(VariancePenaltyLoss, self).__init__()
        if not (0 <= lambda_deviation <= 1):
            raise ValueError("lambda_deviation 必须在 [0, 1] 范围内")

        self.lambda_deviation = lambda_deviation
        self.lambda_mse = 1.0 - self.lambda_deviation

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
        self.eps = 1e-8  # 用于数值稳定性

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值，形状 [B, P, T]（B:批次，P:表型数，T:时间点）
            target: 真实值，形状 [B, P, T]
        """
        # 1. 基础 MSE 损失
        loss_mse = self.mse_loss(pred, target)

        # 2. 计算“预测值偏离其均值”的惩罚项 (deviation penalty)
        # 2.1 在批次维度 (B) 上计算每个 [P, T] 位置的预测均值
        # shape: [P, T]
        pred_mean = pred.mean(dim=0, keepdim=True)  # keepdim=True 保持维度以便广播

        # 2.2 计算每个预测值与其均值的差
        # shape: [B, P, T]
        deviations = pred - pred_mean

        # 2.3 计算惩罚项。我们惩罚的是“偏离”本身，所以使用平方差或绝对差。
        # 使用平方差与 MSE 保持一致，梯度也更平滑。
        # shape: [B, P, T]
        squared_deviations = torch.square(deviations)

        # 2.4 根据 reduction 方式计算最终的惩罚项
        if self.reduction == 'mean':
            # 计算所有元素的平均值
            deviation_penalty = squared_deviations.mean()
        elif self.reduction == 'sum':
            # 计算所有元素的总和
            deviation_penalty = squared_deviations.sum()
        else:
            raise ValueError(f"不支持的 reduction 方式: {self.reduction}")

        # 3. 组合损失
        # 注意：deviation_penalty 已经是一个标量 (scalar)
        total_loss = self.lambda_mse * loss_mse + self.lambda_deviation * deviation_penalty

        return total_loss

class ManualMultiTaskLoss(nn.Module):
    """
    手动权重多任务损失函数 (替代 AdaptiveMultiTaskLoss)

    Total Loss = recon_weight * ReconLoss + pheno_weight * PhenoLoss
    """

    def __init__(self,
                 phenotype_names: List[str],
                 recon_weight: float = 1.0,
                 pheno_weight: float = 10.0):
        """
        Args:
            phenotype_names: 表型名称列表
            recon_weight: 重建损失的权重
            pheno_weight: 表型损失的权重
        """
        super().__init__()
        self.phenotype_names = phenotype_names
        self.num_phenotypes = len(phenotype_names)

        # 1. 定义具体的损失计算方式
        # Trainer 中会调用 criterion.recon_criterion 来计算每个块的损失
        self.recon_criterion = nn.MSELoss()
        self.pheno_loss_fn = nn.MSELoss()  # 也可以改回 MSELoss，视情况而定

        # 2. 保存手动权重
        self.recon_weight = recon_weight
        self.pheno_weight = pheno_weight

    def forward(self, recon_loss, phenotype_pred, phenotype_true):
        """
        Args:
            recon_loss: 外部计算好的平均重建损失 (标量)
            phenotype_pred: 模型预测的表型 [B, P, T]
            phenotype_true: 真实表型 [B, P, T]
        """
        # 1. 计算表型损失
        pheno_loss_sum = 0.0
        for i in range(self.num_phenotypes):
            # 对每个表型分别计算损失并累加
            pheno_loss_sum += self.pheno_loss_fn(phenotype_pred[:, i, :], phenotype_true[:, i, :])

        avg_pheno_loss = pheno_loss_sum / self.num_phenotypes

        # 2. 手动加权求和
        # 直接乘以权重，不再使用 exp(-log_var)
        weighted_recon_loss = self.recon_weight * recon_loss
        weighted_pheno_loss = self.pheno_weight * avg_pheno_loss

        total_loss = weighted_recon_loss + weighted_pheno_loss

        # 返回总损失和纯表型损失（用于日志记录）

        return total_loss, avg_pheno_loss

class MSEPCCLoss(nn.Module):
    def __init__(self, pcc_weight=0.1):
        super().__init__()
        self.pcc_weight = pcc_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred: [Batch, Phenotype, Time]
        target: [Batch, Phenotype, Time]
        """
        # 1. MSE Loss (数值准确性)
        mse_loss = self.mse(pred, target)

        # 2. PCC Loss (排序准确性) - 修正版
        # 我们需要在 Batch 维度 (dim=0) 计算相关性
        # 也就是：对于每一个 (Phenotype, Time) 组合，计算 400 个样本的 PCC

        # 减去每个 (P, T) 的 Batch 均值
        # mean shape: [1, P, T]
        pred_mean = torch.mean(pred, dim=0, keepdim=True)
        target_mean = torch.mean(target, dim=0, keepdim=True)

        vx = pred - pred_mean  # [B, P, T]
        vy = target - target_mean  # [B, P, T]

        # 计算分子分母 (在 Batch 维度求和)
        # sum shape: [P, T]
        numerator = torch.sum(vx * vy, dim=0)
        denominator = torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0))

        # 计算每个 (P, T) 的 PCC
        # 添加 eps 防止除零 (如果某时刻所有样本值都一样，std为0)
        pcc_per_point = numerator / (denominator + 1e-8)

        # 对所有表型和时间点取平均，得到最终 PCC Loss
        # 我们希望 PCC 接近 1，所以 Loss = 1 - PCC
        pcc_loss = 1.0 - torch.mean(pcc_per_point)

        # 打印调试 (可选)
        # print(f"Batch PCC: {torch.mean(pcc_per_point).item():.4f}")

        return mse_loss + self.pcc_weight * pcc_loss
