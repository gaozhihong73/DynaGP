"""
基础训练器抽象类
提取Trainer的共同功能
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from utils import Logger


class BaseTrainer(ABC):
    """
    基础训练器抽象类

    封装了所有Trainer的共同功能:
    1. 优化器和学习率调度器的创建
    2. 训练和验证的epoch循环
    3. 早停机制
    4. 模型保存
    5. 日志记录

    子类需要实现的抽象方法:
    - _create_model(): 创建具体的模型
    - _create_dataloaders(): 创建数据加载器
    - _log_configuration(): 记录配置信息
    - train(): 完整的训练流程
    """

    def __init__(self, config):
        """
        初始化基础训练器

        Args:
            config: 配置对象,包含所有训练参数
        """
        # 保存配置
        self.config = config

        # 保存目录参数
        self.best_model_save_dir = Path(config.best_model_save_dir)  # 最佳模型保存目录
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录(如果不存在)

        self.training_history_save_dir = Path(config.training_history_save_dir)  # 训练历史保存目录
        self.training_history_save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录(如果不存在)

        self.scaler_save_dir = Path(config.scaler_save_dir)  # 归一化器保存目录
        self.scaler_save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录(如果不存在)

        # 初始化日志记录器
        self.log_file = self.config.log_file  # 日志文件路径
        self.logger = self._create_logger()  # 创建日志记录器

        # 训练超参数
        self.epochs = config.epochs  # 训练轮数
        self.lr = config.lr  # 初始学习率
        self.weight_decay = config.weight_decay  # 权重衰减
        self.batch_size = config.batch_size  # 批次大小
        self.patience = config.patience  # 早停耐心值
        self.grad_clip_norm = config.grad_clip_norm  # 梯度裁剪阈值
        self.device = config.device  # 训练设备
        self.seed = config.seed  # 随机种子
        self.num_workers = config.num_workers  # 数据加载的工作进程数

        # 学习率调度器参数
        self.use_warmup = config.use_warmup  # 是否使用学习率预热
        self.warmup_epochs = config.warmup_epochs if self.use_warmup else 0  # 预热轮数
        self.warmup_start_factor = config.warmup_start_factor if self.use_warmup else 0.1  # 预热开始时的学习率因子
        self.warmup_end_factor = config.warmup_end_factor if self.use_warmup else 1.0  # 预热结束时的学习率因子
        self.min_lr = config.min_lr  # 最小学习率

        # 数据集参数
        self.test_ratio = config.test_ratio  # 测试集比例
        self.val_ratio = config.val_ratio  # 验证集比例

        # 是否保存训练信息
        self.is_save_training_info = config.is_save_training_info  # 是否保存训练信息标志

    @abstractmethod
    def _create_model(self):
        """
        创建模型(由子类实现,因为不同阶段的模型不同)

        Returns:
            模型对象
        """
        pass

    @abstractmethod
    def _create_dataset(self):
        """
        获取数据集对象(由子类实现)

        Returns:
            数据集对象
        """
        pass

    @abstractmethod
    def _log_configuration(self):
        """
        记录配置信息(由子类实现,因为不同阶段的配置不同)
        """
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        完整的训练流程(由子类实现,因为不同阶段的训练流程不同)

        Returns:
            训练统计信息字典
        """
        pass

    def _create_logger(self) -> Logger:
        """获取日志记录器"""
        return Logger(self.log_file, unique=False)

    def _create_optimizer_and_scheduler(self):
        """
        创建优化器和学习率调度器

        Returns:
            optimizer: AdamW优化器
            warmup: 预热调度器(可选)
            scheduler: 余弦退火学习率调度器
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
            )

        # 如果使用Warmup
        if self.use_warmup:
            warmup = LinearLR(
                optimizer,
                start_factor=self.warmup_start_factor,  # 从 base_lr * start_factor 开始
                end_factor=self.warmup_end_factor,  # 增长到 base_lr * end_factor
                total_iters=self.warmup_epochs  # warmup持续的epoch数
                )
            # 注意: T_max是warmup结束后的epoch数
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=self.min_lr
                )
            return optimizer, warmup, scheduler
        else:
            # 不使用Warmup
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
                )
            return optimizer, None, scheduler
