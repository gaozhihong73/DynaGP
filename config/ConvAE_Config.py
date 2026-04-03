from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ConvAEConfig:
    """实验配置类"""
    filled_type: int = 0  # 0: -1填充 / 1: 众数填充
    use_pheno_help: bool = False  # 是否使用表型辅助压缩

    # 训练参数
    epochs: int = 100
    lr: float = 0.0001
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 16
    patience: int = 10
    grad_clip_norm: float = 1.0  # 梯度裁剪阈值
    seed: int = 73
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout: float = 0.7
    num_workers: int = 4

    # 损失权重
    recon_weight: float = 0.7
    pheno_weight: float = 0.3

    # 学习率调度器参数
    use_warmup: bool = True  # 使用warmup
    warmup_epochs: int = 10  # warmup轮数
    warmup_start_factor: float = 0.1  # 从 base_lr * 0.1 开始
    warmup_end_factor: float = 1.0  # 持续增长到 base_lr * 1.0

    # 模型参数
    # total_snp_num: int = 2860770  # 原始 SNP 总数量
    chunk_size: int = 50000  # 每个块的大小 (保持固定，不够的补0)
    latent_dim: int = 128  # 每个块被压缩后的特征维度 (Token Dimension)
    # 全局聚合 (Transformer)
    time_coordinates: List[float] = field(default_factory=lambda: [10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80,
                                                                   101])  # 真实的采样时间点 (14)

    phenotype_names: List[str] = field(default_factory=lambda: ['RGBVI', 'WI', 'ExB', 'COM', 'GreenCoverage',
                                                                'bn', 'IPCA', 'NDI'
                                                                ])

    # 数据集参数
    test_ratio: float = 0.1
    val_ratio: float = 0.1

    # 保存参数
    is_save_training_info: bool = True  # 是否保存最优模型

    # 映射表
    FILLED_MAP = {0: "filled_-1", 1: "filled_mode"}

    def __post_init__(self):
        """初始化后验证参数"""
        if self.filled_type not in self.FILLED_MAP:
            raise ValueError(f"filled_type 必须在 [0, 1, 2] 范围内")

    @property
    def filled_name(self) -> str:
        """填充方式名称"""
        return self.FILLED_MAP[self.filled_type]

    @property
    def pheno_help_name(self) -> str:
        """使用表型辅助"""
        if self.use_pheno_help:
            return "use_pheno"
        return "not_pheno"

    @property
    def num_snp_classes(self) -> int:
        """
        原始数据的类别数:
        - filled_type=0 (-1填充): -1,0,1,2 -> 4类
        - filled_type=1 (众数): 0,1,2 -> 3类
        """
        return 4 if self.filled_type == 0 else 3

    @property
    def snp_file(self) -> str:
        """训练SNP文件路径"""
        return f"dataset/maize/snp/{self.filled_name}/snp_2860770.npy"

    @property
    def pheno_dir(self) -> str:
        """表型文件存储目录"""
        return f"dataset/maize/pheno"

    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        return f"outputs/convae/{self.filled_name}/{self.pheno_help_name}"

    @property
    def best_model_save_dir(self) -> str:
        """训练好模型的保存目录"""
        return f"{self.output_dir}/models"

    @property
    def training_history_save_dir(self) -> str:
        """训练历史文件保存目录"""
        return f"{self.output_dir}/training_history"

    @property
    def compressed_snp_save_dir(self) -> str:
        """压缩后的SNP文件目录"""
        return f"dataset/maize/snp_zip/{self.filled_name}/{self.pheno_help_name}"

    @property
    def scaler_save_dir(self) -> str:
        """基因型的归一化器保存目录"""
        return f"{self.output_dir}/scalers"

    @property
    def log_file(self) -> str:
        return f"log/convae/convae_{self.filled_name}_{self.pheno_help_name}.log"
