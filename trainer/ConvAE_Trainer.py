import time
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from config.ConvAE_Config import ConvAEConfig
from dataloader.ConvAE_DataLoader import ConvAEDataset, get_dataloader
from model.ConvAE import ConvAE
from trainer import BaseTrainer
from utils import Logger, MSEPCCLoss


class ConvAETrainer(BaseTrainer):
    """
    ConvAE 训练器 (重构版：分块训练 + AMP + 优雅的两阶段策略)
    """

    def __init__(self, config: ConvAEConfig):
        super().__init__(config)

        # === 1. 实验与路径配置 ===
        self.filled_type = config.filled_type
        self.use_pheno_help = config.use_pheno_help

        self.snp_file = config.snp_file
        self.pheno_dir = config.pheno_dir
        self.compressed_snp_save_dir = Path(config.output_dir) / "compressed_snp"
        self.compressed_snp_save_dir.mkdir(parents=True, exist_ok=True)

        # === 2. 训练策略配置 (Curriculum Learning) ===
        self.target_recon_weight = config.recon_weight  # 目标重建权重
        self.target_pheno_weight = config.pheno_weight  # 目标表型权重

        self.warmup_epochs = 10  # 第一阶段：仅训练重建的轮数
        self.rampup_epochs = 5  # 第二阶段：表型权重爬坡的轮数

        # === 3. 数据集与模型初始化 ===
        self.phenotype_names = config.phenotype_names
        self.chunk_size = config.chunk_size
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout
        self.time_coordinates = config.time_coordinates

        self.logger.info("正在初始化数据集...")
        self.dataset = self._create_dataset()
        self.num_samples = len(self.dataset)
        self.total_snp_num = self.dataset.total_snp_num

        self.model = self._create_model()

        # === 4. 优化器与损失函数 ===
        # 优化器只负责优化参数，不再负责复杂的自适应权重逻辑
        self.optimizer, _, self.scheduler = self._create_optimizer_and_scheduler()

        self.recon_criterion = nn.MSELoss()
        self.pheno_criterion = MSEPCCLoss()

        # AMP 梯度缩放器
        self.scaler = GradScaler(enabled=(self.device == 'cuda'))

    def _create_model(self) -> ConvAE:
        return ConvAE(
            total_snp_num=self.total_snp_num,
            chunk_size=self.chunk_size,
            latent_dim=self.latent_dim,
            num_phenotypes=len(self.phenotype_names),
            time_coordinates=self.time_coordinates,
            degree=3,
            dropout=self.dropout
            ).to(self.device)

    def _create_dataset(self) -> ConvAEDataset:
        return ConvAEDataset(
            snp_file=self.snp_file,
            pheno_dir=self.pheno_dir,
            phenotype_names=self.phenotype_names,
            chunk_size=self.chunk_size,
            filled_type=self.filled_type,
            )

    # ==========================================================================
    # 核心逻辑：动态权重计算策略
    # ==========================================================================
    def _get_current_weights(self, epoch: int) -> Tuple[float, float, str]:
        """
        根据当前 Epoch 计算 Loss 权重
        Returns: (recon_weight, pheno_weight, phase_name)
        """
        # 阶段一：Warmup (纯重建)
        if epoch < self.warmup_epochs:
            return self.target_recon_weight, 0.0, "Warmup"

        # 阶段二：Ramp-up (线性增加表型权重)
        # 例如：warmup=5, rampup=5. epoch=5时ratio=0, epoch=10时ratio=1
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
            # 这里的策略是：表型权重线性增加
            curr_pheno = self.target_pheno_weight * progress
            return self.target_recon_weight, curr_pheno, f"RampUp({progress:.1%})"

        # 阶段三：Full (全速训练)
        else:
            return self.target_recon_weight, self.target_pheno_weight, "Full"

    # ==========================================================================
    # 核心循环：单轮训练
    # ==========================================================================
    def _train_epoch(self, train_loader: DataLoader,
                     recon_weight: float, pheno_weight: float) -> Tuple[float, float, float]:
        """
        训练一个 epoch
        Args:
            recon_weight: 当前轮次的重建权重
            pheno_weight: 当前轮次的表型权重
        """
        self.model.train()
        num_batches = len(train_loader)

        metrics = {'total': 0.0, 'recon': 0.0, 'pheno': 0.0}

        for indices, pheno_target in train_loader:
            indices = indices.numpy()
            pheno_target = pheno_target.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=(self.device == 'cuda')):
                # 1. 计算重建损失 (仅当权重 > 0 时计算，节省算力)
                latent_list = []
                batch_recon_loss = 0.0

                # 逐块编码
                for chunk_idx in range(self.dataset.num_chunks):
                    chunk_tensor = self.dataset.get_chunk_data(chunk_idx, indices).to(self.device)
                    latent_i = self.model.encode_one_chunk(chunk_tensor)
                    latent_list.append(latent_i)

                    if recon_weight > 0:
                        recon_i = self.model.decode_one_chunk(latent_i)
                        batch_recon_loss += self.recon_criterion(recon_i, chunk_tensor)

                # 平均化重建损失
                if recon_weight > 0:
                    recon_loss = batch_recon_loss / self.dataset.num_chunks
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)

                # 2. 计算表型损失 (仅当权重 > 0 时计算)
                all_latents = torch.stack(latent_list, dim=1)  # [B, Chunks, D]

                if pheno_weight > 0:
                    pheno_pred = self.model.predict_from_latents(all_latents)
                    pheno_loss = self.pheno_criterion(pheno_pred, pheno_target)
                else:
                    pheno_loss = torch.tensor(0.0, device=self.device)

                # 3. 加权求和
                total_loss = (recon_weight * recon_loss +
                              pheno_weight * pheno_loss)

            # 反向传播与优化
            self.scaler.scale(total_loss).backward()

            # 严格梯度裁剪 (防止 Ramp-up 期间梯度突变)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 记录指标
            metrics['total'] += total_loss.item()
            metrics['recon'] += recon_loss.item()
            metrics['pheno'] += pheno_loss.item()

        # 计算平均值
        return (metrics['total'] / num_batches,
                metrics['recon'] / num_batches,
                metrics['pheno'] / num_batches)

    def _validate_epoch(self, val_loader: DataLoader,
                        recon_weight: float, pheno_weight: float) -> Tuple[float, float, float]:
        """验证一个 epoch"""
        self.model.eval()
        num_batches = len(val_loader)
        metrics = {'total': 0.0, 'recon': 0.0, 'pheno': 0.0}

        with torch.no_grad():
            for indices, pheno_target in val_loader:
                indices = indices.numpy()
                pheno_target = pheno_target.to(self.device)

                with autocast(enabled=(self.device == 'cuda')):
                    latent_list = []
                    batch_recon_loss = 0.0

                    # 验证时通常全跑，以便观察指标
                    for chunk_idx in range(self.dataset.num_chunks):
                        chunk_tensor = self.dataset.get_chunk_data(chunk_idx, indices).to(self.device)
                        latent_i = self.model.encode_one_chunk(chunk_tensor)
                        latent_list.append(latent_i)

                        recon_i = self.model.decode_one_chunk(latent_i)
                        batch_recon_loss += self.recon_criterion(recon_i, chunk_tensor)

                    recon_loss = batch_recon_loss / self.dataset.num_chunks

                    all_latents = torch.stack(latent_list, dim=1)
                    pheno_pred = self.model.predict_from_latents(all_latents)
                    pheno_loss = self.pheno_criterion(pheno_pred, pheno_target)

                    # 验证集的 Total Loss 建议始终使用"满权重"计算，以便跨 Epoch 比较
                    # 或者跟随训练权重，这里选择跟随训练权重以保持一致性
                    total_loss = (recon_weight * recon_loss + pheno_weight * pheno_loss)

                metrics['total'] += total_loss.item()
                metrics['recon'] += recon_loss.item()
                metrics['pheno'] += pheno_loss.item()

        return (metrics['total'] / num_batches,
                metrics['recon'] / num_batches,
                metrics['pheno'] / num_batches)

    # ==========================================================================
    # 主训练流程
    # ==========================================================================
    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("开始 ConvAE 分块训练 (Curriculum Learning)")
        self.logger.info("=" * 80)
        self._log_configuration()

        train_loader, val_loader, test_loader = get_dataloader(self.dataset, self.batch_size, self.seed)

        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = -1

        start_time = time.time()

        for epoch in range(self.epochs):
            # 1. 获取当前策略权重
            curr_recon_w, curr_pheno_w, phase = self._get_current_weights(epoch)

            # 2. 训练
            t_total, t_recon, t_pheno = self._train_epoch(train_loader, curr_recon_w, curr_pheno_w)

            # 3. 验证 (使用满权重或当前权重，建议在 Full 阶段前只看当前权重 Loss，Full 后看满权重 Loss)
            # 这里简单起见，验证集使用满权重，这样指标具有可比性，不会因为权重变化而跳变
            v_total, v_recon, v_pheno = self._validate_epoch(val_loader, self.target_recon_weight, self.target_pheno_weight)

            # 4. 日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"[{epoch + 1:>3}/{self.epochs}] {phase:<12} | "
                f"Train: [Rec:{t_recon:.4f} Phe:{t_pheno:.4f}] | "
                f"Val: [Rec:{v_recon:.4f} Phe:{v_pheno:.4f}] | "
                f"V_Total: {v_total:.4f} {'★' if v_total < best_val_loss else ''}"
                )

            self.scheduler.step()

            # 5. 早停与保存
            # 只有在进入 RampUp 后期或 Full 阶段后，才开始真正根据验证集 Loss 选模型
            # 否则 Warmup 阶段肯定 Loss 最低 (因为只算 Recon)
            if epoch >= self.warmup_epochs:
                if v_total < best_val_loss:
                    best_val_loss = v_total
                    best_epoch = epoch + 1
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    self.logger.info(f"早停触发: {self.patience} 轮无改善")
                    break
            else:
                patience_counter = 0  # Warmup 期间不计数

        total_time = time.time() - start_time
        self.logger.info(f"训练完成. 最佳 Epoch: {best_epoch}, 最佳 Val Loss: {best_val_loss:.6f}")

        # 6. 保存最终结果并压缩
        if self.is_save_training_info and best_model_state is not None:
            self._save_best_model(best_model_state, best_val_loss, best_epoch, total_time)
            self._compress_data(best_model_state)

    def _compress_data(self, best_model_state: Dict[str, Any]):
        """全量数据压缩逻辑 (保持不变)"""
        self.logger.info("=" * 80)
        self.logger.info("开始全量数据压缩...")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.model.eval()

        all_indices = torch.arange(self.num_samples)
        compress_loader = DataLoader(TensorDataset(all_indices), batch_size=self.batch_size, shuffle=False, num_workers=0)
        compressed_features = []

        with torch.no_grad():
            for (batch_indices,) in compress_loader:
                batch_indices = batch_indices.numpy()
                latent_list = []
                with autocast(enabled=(self.device == 'cuda')):
                    for chunk_idx in range(self.dataset.num_chunks):
                        chunk_tensor = self.dataset.get_chunk_data(chunk_idx, batch_indices).to(self.device)
                        latent_i = self.model.encode_one_chunk(chunk_tensor)
                        latent_list.append(latent_i.float().cpu())  # 转回 float32

                # Stack & Flatten: [B, Num_Chunks * Latent_Dim]
                batch_features = torch.stack(latent_list, dim=1).view(len(batch_indices), -1)
                compressed_features.append(batch_features.numpy())

        final_array = np.concatenate(compressed_features, axis=0)
        save_path = self.compressed_snp_save_dir / "snp_compressed.npy"
        np.save(str(save_path), final_array)

        self.logger.info(f"压缩结果已保存: {save_path} (Shape: {final_array.shape})")

    def _save_best_model(self, state, loss, epoch, time):
        path = self.best_model_save_dir / "best_model.pth"
        torch.save({'model_state_dict': state, 'val_loss': loss, 'epoch': epoch}, path)
        self.logger.info(f"模型已保存: {path}")

    def _log_configuration(self):
        self.logger.info(f"Total SNPs: {self.total_snp_num} | Chunk Size: {self.chunk_size} | Latent: {self.latent_dim}")
        self.logger.info(f"Curriculum: Warmup={self.warmup_epochs}, RampUp={self.rampup_epochs}")