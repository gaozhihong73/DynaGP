import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


class ConvAEDataset(Dataset):
    """
    智能分块数据集类 (集成归一化与填充处理)
    """

    def __init__(self,
                 snp_file: str,
                 pheno_dir: str,
                 phenotype_names: List[str],
                 chunk_size: int = 50000,
                 filled_type: int = 0):  # 新增 filled_type

        self.chunk_size = chunk_size
        self.phenotype_names = phenotype_names
        self.filled_type = filled_type

        # 1. 加载 SNP 数据
        if not Path(snp_file).exists():
            raise FileNotFoundError(f"SNP file not found: {snp_file}")

        try:
            self.snp_data = np.load(snp_file).astype(np.int8)
            print(f"[Dataset] SNP Matrix loaded into RAM. Shape: {self.snp_data.shape}")
        except MemoryError:
            self.snp_data = np.load(snp_file, mmap_mode='r').astype(np.int8)
            print(f"[Dataset] SNP Matrix loaded in mmap mode (Disk).")

        self.n_samples, self.total_snp_num = self.snp_data.shape

        # 2. 计算分块信息
        self.num_chunks = math.ceil(self.total_snp_num / self.chunk_size)

        # 3. 确定归一化分母 (Scale Factor)
        # filled_type 0 (-1填充): 值域 -1, 0, 1, 2 -> shift后 0, 1, 2, 3 -> Max=3
        # filled_type 1 (众数):   值域 0, 1, 2       -> Max=2
        self.scale_factor = 3.0 if self.filled_type == 0 else 2.0
        print(f"[Dataset] Config: Filled Type={filled_type}, Scale Factor={self.scale_factor}")

        # 4. 加载并归一化表型
        self.phenotype = self._load_and_norm_phenotypes(pheno_dir)

    def _load_and_norm_phenotypes(self, pheno_dir):
        """加载表型并进行 Z-Score 归一化"""
        phenos = []
        for name in self.phenotype_names:
            path = Path(pheno_dir) / f"{name}.csv"
            if not path.exists(): path = Path(f"../dataset/maize/phenotype/{name}.csv")
            if not path.exists(): path = Path(f"dataset/maize/phenotype/{name}.csv")
            df = pd.read_csv(path)
            data = df.values.astype(np.float32)  # [N, 1, T]
            # Z-Score 归一化
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            phenos.append(data_norm[:, np.newaxis, :])
        return np.concatenate(phenos, axis=1)  # [N, P, T]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 返回索引和表型，SNP通过 get_chunk_data 获取
        return idx, self.phenotype[idx]

    def get_chunk_data(self, chunk_idx: int, sample_indices: np.ndarray) -> torch.Tensor:
        """
        【核心功能】获取指定块的 SNP 数据并进行归一化
        """
        if chunk_idx >= self.num_chunks:
            raise ValueError(f"Chunk index {chunk_idx} out of bounds")

        # 1. 计算切片范围
        start_col = chunk_idx * self.chunk_size
        end_col = min((chunk_idx + 1) * self.chunk_size, self.total_snp_num)

        # 2. 切取数据
        chunk_data = self.snp_data[sample_indices, start_col:end_col]  # Numpy array

        # 3. 转 Tensor
        chunk_tensor = torch.from_numpy(chunk_data).float()

        # 4. === 核心：数据偏移与归一化 ===
        if self.filled_type == 0:
            # 如果是 -1 填充，先 +1 偏移，使其变为非负数
            chunk_tensor = chunk_tensor + 1.0

        # 归一化到 [0, 1] 区间
        # 例如: 0,1,2 -> 0, 0.5, 1.0
        chunk_tensor = chunk_tensor / self.scale_factor

        # 5. 处理 Padding (补 0)
        # 注意：补的 0 在归一化后依然是 0，代表"无信号"，这是合理的
        current_width = chunk_tensor.shape[1]
        if current_width < self.chunk_size:
            pad_len = self.chunk_size - current_width
            chunk_tensor = F.pad(chunk_tensor, (0, pad_len), "constant", 0)

        return chunk_tensor


def get_dataloader(dataset, batch_size, seed=73):
    """获取 Loader (保持不变)"""
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=seed)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=seed)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    # num_workers=0 对于大内存对象是最安全的
    kwargs = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True}
    return (DataLoader(train_ds, shuffle=True, **kwargs),
            DataLoader(val_ds, shuffle=False, **kwargs),
            DataLoader(test_ds, shuffle=False, **kwargs))
