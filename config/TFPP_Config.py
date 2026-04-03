from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import torch


@dataclass
class TFPPConfig:
    """实验配置类"""
    # 实验组合
    exp_name: str = None  # 实验名称

    # 文件目录
    pheno_dir: str = 'dataset/maize/cleaned_pheno'
    snp_file: str = 'dataset/maize/snp/snp_50000.npy'

    # 训练参数
    epochs: int = 100
    lr: float = 0.0001
    weight_decay: float = 0.0001
    min_lr: float = 1e-7
    batch_size: int = 64
    patience: int = 10
    seed: int = 73
    dropout: float = 0.3
    grad_clip_norm: float = 1.0
    is_save_training_info: bool = True  # 是否保存训练好的模型到本地
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 训练设备

    # 学习率调度器参数
    use_warmup: bool = True  # 使用warmup
    warmup_epochs: int = 8  # warmup轮数
    warmup_start_factor: float = 0.1  # 从 base_lr * 0.1 开始
    warmup_end_factor: float = 1.0  # 持续增长到 base_lr * 1.0

    # 模型参数
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2  # 编码器层数
    d_ff: int = 256  # 前馈网络隐藏维度
    time_coordinates: List[float] = field(default_factory=lambda: [10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80,
                                                                   101])  # 真实的采样时间点 (14)
    phenotype_names: List[str] = field(default_factory=lambda: ['RGBVI', 'WI', 'ExB', 'COM', 'GreenCoverage',
                                                                'bn', 'IPCA', 'NDI'
                                                                ])

    # K折交叉验证参数
    use_kfold: bool = True  # 是否使用K折交叉验证
    n_folds: int = 10  # K折数量

    # 数据集参数
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    num_workers: int = 4
    norm_method: str = 'global'  # 归一化策略：global, timepoint, residual_global
    scaler_type: str = 'zscore'  # 归一化函数：zscore，minmax，robust

    def __post_init__(self):
        """初始化后验证参数"""
        if not self.phenotype_names:
            raise ValueError("至少需要选择一个表型")

    @property
    def experiment_name(self) -> str:
        return self.exp_name if self.exp_name else f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        return f"outputs/tfpp/{self.experiment_name}"

    @property
    def best_model_save_dir(self) -> str:
        """训练好模型的保存目录"""
        return f"{self.output_dir}/models"

    @property
    def training_history_save_dir(self) -> str:
        """训练好模型的保存目录"""
        return f"{self.output_dir}/training_history"

    @property
    def scaler_save_dir(self) -> str:
        """表型的归一化器保存目录"""
        return f"{self.output_dir}/scalers"

    @property
    def log_file(self) -> str:
        """日志文件名"""
        if self.use_kfold:
            return f"{self.output_dir}/{self.experiment_name}_kfold.log"
        else:
            return f"{self.output_dir}/{self.experiment_name}.log"
