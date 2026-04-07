import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader

from config import TFPPConfig
from dataloader import TFPPDataset, get_dataloader, get_dataloader_with_kfold
from model import TFPP
from utils import MSEPCCLoss, calculate_metrics_by_average, calculate_phenotype_metrics_by_time, \
    denormalize_phenotype, load_scalers, save_scalers
from .Base_Trainer import BaseTrainer


class TFPPTrainer(BaseTrainer):
    """
    表型预测Transformer训练器
    主要功能：
    1. 训练基于Transformer的SNP表型预测模型
    2. 使用多种评估指标（PCC、R²、MSE、RMSE）全面评估模型性能
    3. 自动保存最佳模型和训练统计信息
    4. 支持早停和学习率调度
    """

    def __init__(self, config: TFPPConfig):
        """
        初始化训练器

        Args:
            config: 训练器的配置类
        """
        # 价值父类中的公共属性
        super().__init__(config)

        # 实验参数
        self.experiment_name = config.experiment_name  # 实验名称

        # 路径参数
        self.snp_file = config.snp_file  # SNP存储目录
        self.pheno_dir = config.pheno_dir  # 表型存储目录
        self.scaler_file = str(self.scaler_save_dir / "scaler.pkl")

        # 数据集初始化
        self.phenotype_names = config.phenotype_names
        self.dataset = self._create_dataset()  # 数据集
        self.num_sample = len(self.dataset)  # 样本数量
        self.norm_method = config.norm_method
        self.scaler_type = config.scaler_type

        # 模型配置
        self.snp_dim: int = self.dataset.snp.shape[1]
        self.d_model = config.d_model  # 模型隐藏维度
        self.num_heads = config.num_heads  # 多头注意力头数
        self.num_layers = config.num_layers  # 编码器层数
        self.num_embeddings = 4
        self.d_ff = config.d_ff  # 前馈网络隐藏维度
        self.time_coordinates = config.time_coordinates  # 真实采用时间
        self.dropout = config.dropout
        self.model = self._create_model()  # 创建模型

        # K折交叉验证参数
        self.use_kfold = config.use_kfold  # 是否使用K折交叉验证
        self.n_folds = config.n_folds  # 交叉验证的折数

    def _create_model(self):
        """获取模型"""
        return TFPP(
            snp_dim=self.snp_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            num_embeddings=self.num_embeddings,
            time_coordinates=self.time_coordinates,
            phenotype_names=self.phenotype_names,
            dropout=self.dropout
            ).to(self.device)

        # return NetGP_Adapted(
        #     snp_dim=self.snp_dim,  # 50000
        #     num_phenotypes=len(self.phenotype_names),  # 表型数量
        #     time_points=len(self.time_coordinates),  # 时间点数量 (14)
        #     hidden_dim=2000  # 对应原版 m 维度
        #     ).to(self.device)

        # return DSGWAS_Adapted(
        #     snp_dim=self.snp_dim,  # 50000
        #     num_phenotypes=len(self.phenotype_names),
        #     time_points=len(self.time_coordinates)
        #     ).to(self.device)

        # return PKDP_Adapted(
        #     snp_dim=self.snp_dim,  # 通常是 50000
        #     num_phenotypes=len(self.phenotype_names),
        #     time_points=len(self.time_coordinates)
        #     ).to(self.device)

        # return DNAWhisper_Adapted(
        #     snp_dim=self.snp_dim,  # 50000
        #     num_phenotypes=len(self.phenotype_names),  # 表型数
        #     time_points=len(self.time_coordinates)  # 14
        #     ).to(self.device)

    def _create_dataset(self):
        """
        获取数据集对象
        """
        return TFPPDataset(self.snp_file, self.pheno_dir, self.phenotype_names)

    def _create_optimizer_and_scheduler(self):
        """
        创建优化器：实现分层学习率 + 分层权重衰减
        """

        # 1. 定义基础参数
        base_lr = self.lr  # 比如 1e-4

        # 2. 参数分组
        wide_params = []
        deep_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # 分组逻辑
            if 'wide_layer' in name:
                wide_params.append(param)
            else:
                deep_params.append(param)  # 其他都算 Deep

        # 3. 创建优化器 (AdamW)
        optimizer = optim.AdamW([
            # Group A: 宽通路
            {
                'params':       wide_params,
                'lr':           base_lr * 0.1,
                'weight_decay': self.weight_decay * 100
                },

            # Group B: 深通路
            {
                'params':       deep_params,
                'lr':           base_lr,
                'weight_decay': self.weight_decay
                },
            ])

        # 4. 调度器 (Scheduler)
        # CosineAnnealingLR 会自动处理 optimizer 中的所有 groups
        # 两个 group 的 lr 都会按照各自的初始值同步下降
        if self.use_warmup:
            warmup = LinearLR(
                optimizer, start_factor=self.warmup_start_factor,
                end_factor=1.0, total_iters=self.warmup_epochs
                )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.epochs - self.warmup_epochs, eta_min=self.min_lr
                )
            return optimizer, warmup, scheduler
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
            return optimizer, None, scheduler

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> \
            Tuple[float, Dict[str, float]]:
        """
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            pos: 位置编码或其他辅助输入 (可选)
        Returns:
            平均训练损失和评估指标
        """
        self.model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        all_preds = []
        all_targets = []

        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()

            # === 常规训练路径  ===
            phenotype_pred = self.model(X)

            # 1. 基础预测 Loss (MSE/PCC)
            loss = criterion(phenotype_pred, y)

            # 为了统计指标统一，这里 targets 保持原样
            loss.backward()

            # === 梯度监控 (如果需要) ===
            # self._monitor_gradients()

            # 4. 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()

            # 5. 收集预测值和真实值用于计算指标
            all_preds.append(phenotype_pred.detach())
            all_targets.append(y.detach())

            # 计算整体指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 调用你的指标计算函数
        phenotype_metrics = calculate_phenotype_metrics_by_time(all_preds, all_targets, self.phenotype_names)
        phenotype_metrics_avg = calculate_metrics_by_average(phenotype_metrics)

        return train_loss / num_batches, phenotype_metrics_avg

    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> \
            Tuple[float, Dict[str, Dict[str, float]], Dict[str, float]]:  # <--- 修改了返回类型提示，增加了 float
        """
        验证一个epoch (已修改：支持记录 Gate 值)
        """
        self.model.eval()
        num_batches = len(val_loader)
        val_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                phenotype_pred = self.model(X)

                # 计算 Loss (只用预测值)
                loss = criterion(phenotype_pred, y)
                val_loss += loss.item()

                # 收集预测值和真实值
                all_preds.append(phenotype_pred)
                all_targets.append(y)

        # 计算整体指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算各表型指标
        phenotype_metrics = calculate_phenotype_metrics_by_time(all_preds, all_targets, self.phenotype_names)
        phenotype_metrics_avg = calculate_metrics_by_average(phenotype_metrics)

        # 返回值增加了 avg_gate
        return val_loss / num_batches, phenotype_metrics, phenotype_metrics_avg

    def _evaluate_test_set(self, test_loader: DataLoader, scalers: Dict) -> Dict[str, Any]:
        """
        在测试集上进行最终评估（使用反归一化后的数据）

        Args:
            test_loader: 测试数据加载器
            scalers: 归一化器字典

        Returns:
            测试集评估结果
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                phenotype_pred = self.model(X)

                all_preds.append(phenotype_pred.cpu())
                all_targets.append(y.cpu())

        # 合并所有批次
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 反归一化
        all_preds_denorm = denormalize_phenotype(all_preds, scalers)
        all_targets_denorm = denormalize_phenotype(all_targets, scalers)

        # 计算归一化数据的指标
        norm_pheno_metrics = calculate_phenotype_metrics_by_time(all_preds, all_targets, self.phenotype_names)
        norm_phenotype_metrics_avg = calculate_metrics_by_average(norm_pheno_metrics)

        # 计算反归一化数据的指标
        denorm_pheno_metrics = calculate_phenotype_metrics_by_time(all_preds_denorm, all_targets_denorm, self.phenotype_names)
        denorm_phenotype_metrics_avg = calculate_metrics_by_average(denorm_pheno_metrics)

        return {
            'normalized':   {
                'phenotypes': norm_pheno_metrics,
                'overall':    norm_phenotype_metrics_avg
                },
            'denormalized': {
                'phenotypes': denorm_pheno_metrics,
                'overall':    denorm_phenotype_metrics_avg
                }
            }

    def _monitor_gradients(self):
        """监控梯度统计信息"""
        grad_norms = []
        grad_means = []
        grad_stds = []
        layer_names = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()

                grad_norms.append(grad_norm)
                grad_means.append(grad_mean)
                grad_stds.append(grad_std)
                layer_names.append(name)

                # 记录异常梯度
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"异常梯度: {name} - 包含NaN或Inf")

        if grad_norms:  # 确保有梯度数据
            # 记录整体统计
            self.logger.info(
                f"梯度范数: 平均={np.mean(grad_norms):.6f}, "
                f"最大={np.max(grad_norms):.6f}, "
                f"最小={np.min(grad_norms):.6f}"
                )

            # 记录梯度最大的前3层
            if len(grad_norms) > 3:
                top_indices = np.argsort(grad_norms)[-3:][::-1]
                for idx in top_indices:
                    self.logger.info(f"  梯度最大层: {layer_names[idx]} - 范数={grad_norms[idx]:.6f}")

    def _train_regular(self) -> Dict[str, Any]:
        """
        常规训练模式（单次训练验证测试划分）

        Returns:
            训练统计信息字典
        """
        self.logger.info("=" * 80)
        self.logger.info("常规训练")
        self.logger.info("=" * 80)

        # 输出配置信息
        self._log_configuration()

        start_time = time.time()

        # 创建数据加载器
        train_loader, val_loader, test_loader, scalers = get_dataloader(
            dataset=self.dataset,
            norm_method=self.norm_method,
            scaler_type=self.scaler_type,
            batch_size=self.batch_size,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
            device=self.device)

        # 保存表型的归一化器，用于在预测的时候进行反归一化
        save_scalers(scalers, self.scaler_file)
        self.logger.info(f"归一化器已保存至: {self.scaler_file}")
        self.logger.info("数据加载完成")

        # 创建优化器和调度器
        optimizer, warmup, scheduler = self._create_optimizer_and_scheduler()

        criterion = MSEPCCLoss()

        # 训练状态跟踪
        best_val_loss = float('inf')
        best_val_metrics = {}
        best_val_pheno_metrics = None  # 记录各表型的最优指标
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        # 训练历史记录
        train_loss_history = []
        val_loss_history = []
        train_metrics_history = []  # 每轮训练的综合指标
        val_metrics_history = []  # 每轮验证的综合指标
        val_pheno_metrics_history = []  # 每轮验证各个表型的详细指标

        # 训练循环
        for epoch in range(self.epochs):
            # 训练一个epoch
            avg_train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            # 验证一个epoch
            avg_val_loss, val_pheno_metrics, val_metrics = self._validate_epoch(val_loader, criterion)

            # 记录损失历史
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            val_pheno_metrics_history.append(val_pheno_metrics)

            # 日志输出
            current_lr = optimizer.param_groups[0]['lr']

            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Loss[train:{avg_train_loss:.6f} / val:{avg_val_loss:.6f}] | "
                f"PCC[train:{train_metrics['pcc']:.6f} / val:{val_metrics['pcc']:.6f}] | "
                f"R²[train:{train_metrics['r2']:.6f} / val:{val_metrics['r2']:.6f}] | "
                f"lr:{current_lr:.2e}"
                f"{' ★' if avg_val_loss < best_val_loss else ''}"
                )

            # 学习率调度
            if epoch < self.warmup_epochs and self.use_warmup and warmup is not None:
                warmup.step()  # Warmup阶段
            else:
                scheduler.step()

            # 更新最佳指标
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_pheno_metrics = val_pheno_metrics  # 各个表型的最优指标
                best_val_metrics = val_metrics  # 所有表型的综合最优指标
                best_epoch = epoch + 1
                patience_counter = 0
                # 记录最佳epoch的各表型指标
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.patience:
                self.logger.info(f"连续 {self.patience} 轮无改善，触发早停！")
                break

        training_time = time.time() - start_time
        final_lr = optimizer.param_groups[0]['lr']

        # 输出训练结果摘要
        print()
        self.logger.info("训练完成摘要:")
        self.logger.info(f"    best_epoch: {best_epoch}/{self.epochs}")
        self.logger.info(f"    best_val_loss: {best_val_loss:.6f}")
        self.logger.info(f"    best_val_mse: {best_val_metrics['mse']:.6f}")
        self.logger.info(f"    best_val_rmse: {best_val_metrics['rmse']:.6f}")
        self.logger.info(f"    best_val_mae: {best_val_metrics['mae']:.6f}")
        self.logger.info(f"    best_val_pcc: {best_val_metrics['pcc']:.6f}")
        self.logger.info(f"    best_val_r²: {best_val_metrics['r2']:.6f}")
        self.logger.info(f"    final_lr: {final_lr:.2e}")
        self.logger.info(f"    training_time: {training_time / 60:.2f} 分钟\n")

        # 输出各表型的最佳指标
        self.logger.info("各表型最佳验证指标:")
        for pheno_name in self.phenotype_names:
            metrics = best_val_pheno_metrics[pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
                )
        print()

        # 在测试集上评估
        self.logger.info("=" * 80)
        self.logger.info("测试集评估")
        self.logger.info("=" * 80)

        # 加载最佳模型
        self.model.load_state_dict(best_model_state)

        # 加载scalers
        scalers = load_scalers(self.scaler_file)
        self.logger.info(f"Scaler对象已从 {self.scaler_file} 加载")

        # 测试集评估
        test_results = self._evaluate_test_set(test_loader, scalers)

        # 输出测试集结果
        self.logger.info("（归一化）数据的测试集整体指标:")
        for metric, value in test_results['normalized']['overall'].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("（反归一化）数据的测试集整体指标:")
        for metric, value in test_results['denormalized']['overall'].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("（归一化）各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results['normalized']['phenotypes'][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}，"
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
                )
        print()

        self.logger.info("（反归一化）各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results['denormalized']['phenotypes'][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"MSE={metrics['mse']:.6f}, "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}，"
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
                )
        print()

        # 保存模型和训练信息
        if self.is_save_training_info:
            self._save_model_info(
                best_model_state,
                best_val_loss,
                best_epoch,
                best_val_metrics,
                best_val_pheno_metrics,
                training_time,
                train_loss_history,
                val_loss_history,
                train_metrics_history,
                val_metrics_history,
                val_pheno_metrics_history,
                final_lr,
                test_results,
                )

        return {
            'best_val_loss':    best_val_loss,
            'best_val_metrics': best_val_metrics,
            'best_epoch':       best_epoch,
            'training_time':    training_time,
            'test_results':     test_results,
            }

    def _save_model_info(self, best_model_state: Dict,
                         best_val_loss: float,
                         best_epoch: int,
                         best_val_metrics: Dict,
                         best_val_pheno_metrics: Dict[str, Dict[str, float]],
                         training_time: float,
                         train_loss_history: list,
                         val_loss_history: list,
                         train_metrics_history: list,
                         val_metrics_history: list,
                         val_pheno_metrics_history: list,
                         final_lr: float,
                         test_results: Dict,
                         ) -> None:
        """
        保存模型和训练信息

        Args:
            best_model_state: 最佳模型状态字典
            best_val_loss: 最佳验证损失
            best_epoch: 最佳轮次
            best_val_metrics: 最佳验证集指标
            best_val_pheno_metrics：最佳验证集各表型指标
            training_time: 训练时间
            train_loss_history: 训练损失历史
            val_loss_history: 验证损失历史
            train_metrics_history: 训练指标历史
            val_metrics_history: 验证指标历史
            final_lr: 最终学习率
            test_results: 测试集结果
        """
        training_info = {
            'model_state_dict':          best_model_state,
            'train_loss_history':        train_loss_history,
            'val_loss_history':          val_loss_history,
            'val_pheno_metrics_history': val_pheno_metrics_history,
            'train_metrics_history':     train_metrics_history,
            'val_metrics_history':       val_metrics_history,
            'best_val_loss':             best_val_loss,
            'best_val_metrics':          best_val_metrics,
            'best_val_pheno_metrics':    best_val_pheno_metrics,
            'best_epoch':                best_epoch,
            'total_epochs':              self.epochs,
            'training_time':             training_time,
            'final_lr':                  final_lr,
            'test_results':              test_results
            }

        model_path = self.best_model_save_dir / "best_model.pth"
        torch.save(training_info, model_path)
        self.logger.info(f"模型已保存: {model_path}")

        # 保存训练历史为CSV
        history_df = pd.DataFrame({
            'epoch':      range(1, len(train_loss_history) + 1),
            'train_loss': train_loss_history,
            'val_loss':   val_loss_history,
            'train_mse':  [m['mse'] for m in train_metrics_history],
            'train_rmse': [m['rmse'] for m in train_metrics_history],
            'train_mae':  [m['mae'] for m in train_metrics_history],
            'train_pcc':  [m['pcc'] for m in train_metrics_history],
            'train_r2':   [m['r2'] for m in train_metrics_history],
            'val_mse':    [m['mse'] for m in val_metrics_history],
            'val_rmse':   [m['rmse'] for m in val_metrics_history],
            'val_mae':    [m['mae'] for m in val_metrics_history],
            'val_pcc':    [m['pcc'] for m in val_metrics_history],
            'val_r2':     [m['r2'] for m in val_metrics_history],

            })
        history_path = self.training_history_save_dir / "training_history_metrics.csv"
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"训练历史综合指标已保存: {history_path}")

        # 保存所有轮次的表型的所有训练指标
        metric_names = ['mse', 'rmse', 'mae', 'pcc', 'r2']
        for metric in metric_names:
            # 创建数据字典
            data = {'epoch': list(range(1, len(val_pheno_metrics_history) + 1))}

            # 为每个表型添加该指标的数据
            for phenotype in self.phenotype_names:
                phenotype_data = []
                for epoch_metrics in val_pheno_metrics_history:
                    if phenotype in epoch_metrics and metric in epoch_metrics[phenotype]:
                        phenotype_data.append(epoch_metrics[phenotype][metric])
                    else:
                        phenotype_data.append(None)  # 如果数据缺失
                data[phenotype] = phenotype_data

            # 创建DataFrame并保存
            df_metric = pd.DataFrame(data)
            metric_file = self.training_history_save_dir / f"phenotype_{metric}_history.csv"
            df_metric.to_csv(metric_file, index=False)
            self.logger.info(f"所有表型的训练{metric.upper()}指标已保存: {metric_file}")

    def _train_single_fold(self, fold_idx: int,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: DataLoader,
                           scalers: Dict) -> Dict[str, Any]:
        """
        训练单个折

        Args:
            fold_idx: 当前折的索引
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            scalers: 归一化器字典

        Returns:
            该折的训练结果
        """
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"开始训练第 {fold_idx + 1}/{self.n_folds} 折")
        self.logger.info(f"{'=' * 80}")

        # 重新初始化模型（每折使用新模型）
        self.model = self._create_model()

        # 创建优化器和调度器
        optimizer, warmup, scheduler = self._create_optimizer_and_scheduler()
        criterion = MSEPCCLoss()

        # 训练状态跟踪
        best_val_loss = float('inf')
        best_val_metrics = None
        best_pheno_metrics = None
        best_model_state = None
        patience_counter = 0
        best_epoch = -1

        # 训练历史记录
        train_loss_history = []
        val_loss_history = []
        train_metrics_history = []
        val_metrics_history = []

        # 训练循环
        for epoch in range(self.epochs):
            # 训练一个epoch
            avg_train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion)

            # 验证一个epoch
            avg_val_loss, val_pheno_metrics, val_metrics = self._validate_epoch(val_loader, criterion)

            # 记录损失历史
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)

            # 日志输出
            current_lr = optimizer.param_groups[0]['lr']

            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] "
                f"Loss[train:{avg_train_loss:.6f} / val:{avg_val_loss:.6f}] | "
                f"PCC[train:{train_metrics['pcc']:.6f} / val:{val_metrics['pcc']:.6f}] | "
                f"R²[train:{train_metrics['r2']:.6f} / val:{val_metrics['r2']:.6f}] | "
                f"lr: {current_lr:.2e}"
                f"{' ★' if avg_val_loss < best_val_loss else ''}"
                )

            # 学习率调度
            if epoch < self.warmup_epochs and self.use_warmup and warmup is not None:
                warmup.step()
            else:
                scheduler.step()

            # 更新最佳指标
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                patience_counter = 0
                best_pheno_metrics = val_pheno_metrics
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.patience:
                self.logger.info(f"连续 {self.patience} 轮无改善，触发早停！")
                break

        # 输出该折训练结果摘要
        self.logger.info(f"第 {fold_idx + 1} 折训练完成摘要:")
        self.logger.info(f"    best_epoch: {best_epoch}/{self.epochs}")
        self.logger.info(f"    best_val_loss: {best_val_loss:.6f}")
        self.logger.info(f"    best_val_mse: {best_val_metrics['mse']:.6f}")
        self.logger.info(f"    best_val_rmse: {best_val_metrics['rmse']:.6f}")
        self.logger.info(f"    best_val_pcc: {best_val_metrics['pcc']:.6f}")
        self.logger.info(f"    best_val_r²: {best_val_metrics['r2']:.6f}\n")

        self.logger.info(f"{'=' * 14} 开始测试第 {fold_idx + 1}/{self.n_folds} 折 {'=' * 14}")

        # 重要：这里的 val_loader 实际上是该折的测试集
        # 使用反归一化后的数据进行最终评估
        test_results = self._evaluate_test_set(test_loader, scalers)

        # 输出测试集结果
        self.logger.info("归一化数据的测试集整体指标:")
        for metric, value in test_results['normalized']['overall'].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("反归一化数据的测试集整体指标:")
        for metric, value in test_results['denormalized']['overall'].items():
            self.logger.info(f"    {metric.upper():>5}: {value:.6f}")
        print()

        self.logger.info("归一化各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results['normalized']['phenotypes'][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
                )
        print()

        self.logger.info("反归一化各表型测试集指标:")
        for pheno_name in self.phenotype_names:
            metrics = test_results['denormalized']['phenotypes'][pheno_name]
            self.logger.info(
                f"    {pheno_name:>5}: "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"PCC={metrics['pcc']:.6f}, "
                f"R²={metrics['r2']:.6f}"
                )
        print()

        return {
            'best_val_loss':         best_val_loss,
            'best_val_metrics':      best_val_metrics,
            'best_epoch':            best_epoch,
            'best_model_state':      best_model_state,
            'best_pheno_metrics':    best_pheno_metrics,
            'train_loss_history':    train_loss_history,
            'val_loss_history':      val_loss_history,
            'train_metrics_history': train_metrics_history,
            'val_metrics_history':   val_metrics_history,
            'test_results':          test_results  # 包含反归一化后的评估结果
            }

    def _train_with_kfold(self) -> Dict[str, Any]:
        """
        使用K折交叉验证评估模型性能

        流程说明：
        1. 将所有数据分成K折
        2. 每折训练一个独立模型
        3. 每折的验证集在训练时用于早停，训练后作为测试集评估
        4. 汇总K折结果，报告反归一化指标的 Mean ± Std

        注意：
        - 这里的"验证集"既用于训练中的模型选择，也用于最终评估
        - 由于每折训练独立模型，不存在信息泄露问题
        - 最终指标全部基于反归一化后的数据

        Returns:
            K折交叉验证的统计结果
        """
        self._log_configuration()

        # 获取所有样本索引
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)

        # 创建K折分割器
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # 存储每折的结果
        fold_results = []

        # 存储所有折的反归一化指标
        all_fold_denorm_metrics = {
            'mse':  [],
            'rmse': [],
            'mae':  [],
            'pcc':  [],
            'r2':   [],
            }
        # 存储每个表型在所有折上的反归一化指标
        phenotype_fold_metrics = {pheno: {'mse': [], 'rmse': [], 'mae': [], 'pcc': [], 'r2': []}
                                  for pheno in self.phenotype_names}
        total_start_time = time.time()

        # K折交叉验证循环
        # 注意：sklearn 的 split 返回的是 (train_index, test_index)
        # 我们将第二个返回值重命名为 test_indices，作为最终评估用的【测试集】
        for fold_idx, (train_val_indices, test_indices) in enumerate(kfold.split(indices)):
            # ======================================================
            # 核心修改：二次划分 (从 K-1 折数据中分出验证集)
            # ======================================================
            # 从 train_total_indices 中划分出 10% 作为验证集 (用于 Early Stopping)
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.11,  # 约等于 1/9，这样如果是10折，验证集和测试集大小差不多
                random_state=self.seed
                )
            # 打印检查 (确保没有重叠)
            # assert len(set(train_indices) & set(val_indices)) == 0
            # assert len(set(train_indices) & set(test_indices)) == 0
            print(f"Fold {fold_idx + 1}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
            # ======================================================
            # 获取 DataLoader (需要修改该函数以接收三个索引列表)
            # ======================================================
            train_loader, val_loader, test_loader, scalers = get_dataloader_with_kfold(train_indices=train_indices,
                                                                                       val_indices=val_indices,
                                                                                       test_indices=test_indices,
                                                                                       norm_method=self.norm_method,
                                                                                       scaler_type=self.scaler_type,
                                                                                       dataset=self.dataset,
                                                                                       batch_size=self.batch_size,
                                                                                       device=self.device)
            # 保存表型的归一化器，用于在预测的时候进行反归一化
            phenotype_scaler_save_path = self.scaler_save_dir / f"scaler_fold_{fold_idx + 1}.pkl"
            save_scalers(scalers, phenotype_scaler_save_path)
            self.logger.info(f"表型归一化器已保存至: {phenotype_scaler_save_path}")

            # 训练当前折
            fold_result = self._train_single_fold(fold_idx, train_loader, val_loader, test_loader, scalers)
            fold_results.append(fold_result)

            # 收集反归一化后的指标
            denorm_overall = fold_result['test_results']['denormalized']['overall']
            all_fold_denorm_metrics['mse'].append(denorm_overall['mse'])
            all_fold_denorm_metrics['rmse'].append(denorm_overall['rmse'])
            all_fold_denorm_metrics['mae'].append(denorm_overall['mae'])
            all_fold_denorm_metrics['pcc'].append(denorm_overall['pcc'])
            all_fold_denorm_metrics['r2'].append(denorm_overall['r2'])

            # 收集每个表型的反归一化指标
            denorm_phenotypes = fold_result['test_results']['denormalized']['phenotypes']
            for pheno_name in self.phenotype_names:
                phenotype_fold_metrics[pheno_name]['mse'].append(denorm_phenotypes[pheno_name]['mse'])
                phenotype_fold_metrics[pheno_name]['rmse'].append(denorm_phenotypes[pheno_name]['rmse'])
                phenotype_fold_metrics[pheno_name]['mae'].append(denorm_phenotypes[pheno_name]['mae'])
                phenotype_fold_metrics[pheno_name]['pcc'].append(denorm_phenotypes[pheno_name]['pcc'])
                phenotype_fold_metrics[pheno_name]['r2'].append(denorm_phenotypes[pheno_name]['r2'])

        total_time = time.time() - total_start_time

        # ========== 输出K折交叉验证总结 ==========
        self.logger.info("=" * 80)
        self.logger.info(f"{self.n_folds}折交叉验证完成!")
        self.logger.info("=" * 80)
        self.logger.info(f"总耗时: {total_time / 60:.2f} 分钟")
        self.logger.info(f"平均每折耗时: {total_time / self.n_folds / 60:.2f} 分钟")
        print()

        # 计算并输出整体反归一化指标的统计
        self.logger.info("整体反归一化指标统计 (Mean ± Std):")
        for metric_name, values in all_fold_denorm_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            self.logger.info(f"    {metric_name.upper():>5}: {mean_val:.6f} ± {std_val:.6f}")
        print()

        # 输出每个表型的反归一化指标统计
        self.logger.info("各表型反归一化指标统计 (Mean ± Std):")
        for pheno_name in self.phenotype_names:
            self.logger.info(f"  {pheno_name:>5}:")
            for metric_name, values in phenotype_fold_metrics[pheno_name].items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                self.logger.info(f"      {metric_name.upper():>5}: {mean_val:.6f} ± {std_val:.6f}")
        print()

        # 保存K折统计结果
        if self.is_save_training_info:
            self._save_kfold_results(fold_results, all_fold_denorm_metrics, phenotype_fold_metrics, total_time)

        return {
            'fold_results':             fold_results,
            'overall_denorm_metrics':   all_fold_denorm_metrics,
            'phenotype_denorm_metrics': phenotype_fold_metrics,
            'total_time':               total_time
            }

    def _save_kfold_results(self, fold_results: list, overall_metrics: dict,
                            phenotype_metrics: dict, total_time: float):
        """
        保存K折交叉验证结果

        Args:
            fold_results: 每折的详细结果
            overall_metrics: 整体反归一化指标
            phenotype_metrics: 各表型反归一化指标
            total_time: 总训练时间
        """
        # 保存整体指标统计
        overall_stats = pd.DataFrame({
            'Metric': list(overall_metrics.keys()),
            'Mean':   [np.mean(v) for v in overall_metrics.values()],
            'Std':    [np.std(v) for v in overall_metrics.values()],
            'Min':    [np.min(v) for v in overall_metrics.values()],
            'Max':    [np.max(v) for v in overall_metrics.values()]
            })
        overall_stats_path = self.training_history_save_dir / "kfold_overall_stats.csv"
        overall_stats.to_csv(overall_stats_path, index=False)
        self.logger.info(f"整体K折统计已保存: {overall_stats_path}\n")

        # 保存各表型指标统计
        phenotype_stats_list = []
        for pheno_name in self.phenotype_names:
            for metric_name, values in phenotype_metrics[pheno_name].items():
                phenotype_stats_list.append({
                    'Phenotype': pheno_name,
                    'Metric':    metric_name.upper(),
                    'Mean':      np.mean(values),
                    'Std':       np.std(values),
                    'Min':       np.min(values),
                    'Max':       np.max(values)
                    })
        phenotype_stats = pd.DataFrame(phenotype_stats_list)
        phenotype_stats_path = self.training_history_save_dir / "kfold_phenotype_stats.csv"
        phenotype_stats.to_csv(phenotype_stats_path, index=False)
        self.logger.info(f"各表型K折统计已保存: {phenotype_stats_path}\n")

        # 保存每折的详细结果和模型
        fold_details = []
        for i, result in enumerate(fold_results):
            denorm_overall = result['test_results']['denormalized']['overall']
            fold_details.append({
                'Fold':       i + 1,
                'Best_Epoch': result['best_epoch'],
                'Val_Loss':   result['best_val_loss'],
                'Val_MSE':    denorm_overall['mse'],
                'Val_RMSE':   denorm_overall['rmse'],
                'Val_MAE':    denorm_overall['mae'],
                'Val_PCC':    denorm_overall['pcc'],
                'Val_R2':     denorm_overall['r2']
                })
            # 获取最优模型
            best_model_state = result['best_model_state']
            # 最优模型保存路径
            model_path = self.best_model_save_dir / f"tfpp_best_model_kfold_{i + 1}.pth"
            torch.save(best_model_state, model_path)
        self.logger.info(f"每折最优模型结果已保存:{self.best_model_save_dir}\n")
        fold_details_df = pd.DataFrame(fold_details)
        fold_details_path = self.training_history_save_dir / "kfold_fold_details.csv"
        fold_details_df.to_csv(fold_details_path, index=False)
        self.logger.info(f"每折详细结果已保存: {fold_details_path}\n")

        # 保存K折摘要信息
        summary_path = self.training_history_save_dir / "kfold_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.n_folds}折交叉验证摘要\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"总训练时间: {total_time / 60:.2f} 分钟\n")
            f.write(f"平均每折时间: {total_time / self.n_folds / 60:.2f} 分钟\n\n")

            f.write(f"整体反归一化指标 (Mean ± Std):\n")
            for metric_name, values in overall_metrics.items():
                f.write(f"  {metric_name.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f}\n")

            f.write(f"\n各表型反归一化指标 (Mean ± Std):\n")
            for pheno_name in self.phenotype_names:
                f.write(f"  {pheno_name:>5}:\n")
                for metric_name, values in phenotype_metrics[pheno_name].items():
                    f.write(f"    {metric_name.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f}\n")

        self.logger.info(f"K折摘要已保存: {summary_path}\n")

    def train(self) -> Dict[str, Any]:
        """
        根据配置选择训练模式（K折交叉验证 或 常规训练）

        Returns:
            训练统计信息字典
        """
        if self.use_kfold:
            # 使用K折交叉验证
            return self._train_with_kfold()
        else:
            # 使用常规训练
            return self._train_regular()

    def _log_configuration(self) -> None:
        """记录配置信息"""
        self.logger.info("训练配置:")
        self.logger.info(f"    实验名称: {self.experiment_name}")
        self.logger.info(f"    样本数量: {self.num_sample}")
        self.logger.info(f"    表型数量: {len(self.phenotype_names)} {self.phenotype_names}")
        self.logger.info(f"    时间点数: {self.time_coordinates}")
        self.logger.info(f"    训练模式: {'K折交叉验证' if self.use_kfold else '常规训练'}")
        if self.use_kfold:
            self.logger.info(f"    折数: {self.n_folds}")
        else:
            self.logger.info(f"    test_ratio: {self.test_ratio}")
            self.logger.info(f"    val_ratio: {self.val_ratio}")
        self.logger.info(f"    epochs: {self.epochs}")
        self.logger.info(f"    lr: {self.lr}")
        self.logger.info(f"    min_lr: {self.min_lr}")
        self.logger.info(f"    weight_decay: {self.weight_decay}")
        self.logger.info(f"    batch_size: {self.batch_size}")
        self.logger.info(f"    patience: {self.patience}")
        self.logger.info(f"    grad_clip_norm: {self.grad_clip_norm}")
        self.logger.info(f"    seed: {self.seed}")
        self.logger.info(f"    device: {self.device}")
        self.logger.info(f"    use_warmup: {self.use_warmup}")
        self.logger.info(f"    norm_method: {self.norm_method}")
        self.logger.info(f"    scaler_type: {self.scaler_type}")
        if self.use_warmup:
            self.logger.info(f"    warmup_epochs: {self.warmup_epochs}")
        self.logger.info(f"模型配置:")
        self.logger.info(f"    snp_dim: {self.snp_dim}")
        self.logger.info(f"    d_model: {self.d_model}")
        self.logger.info(f"    num_heads: {self.num_heads}")
        self.logger.info(f"    num_layers: {self.num_layers}")
        self.logger.info(f"    d_ff: {self.d_ff}")
        self.logger.info(f"    dropout: {self.dropout}")
        self.logger.info(f"路径配置:")
        self.logger.info(f"    snp_file: {self.snp_file}")
        self.logger.info(f"    pheno_dir: {self.pheno_dir}")
        self.logger.info(f"    model_save_dir: {self.best_model_save_dir}")
        self.logger.info(f"    training_history_save_dir: {self.training_history_save_dir}")
        self.logger.info(f"    phenotype_scaler_dir: {self.scaler_save_dir}")
