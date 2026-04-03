import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


# ==========================================
# 1. 基础组件 (已修复尺寸对齐问题)
# ==========================================
class ConvBlock(nn.Module):
    """
    标准卷积块 (Encoder使用)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.3):
        super().__init__()
        self.conv_main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第二层卷积保持尺寸不变: k=3, s=1, p=1 (满足 k=2p+1)
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            )

        # Shortcut 处理
        if stride != 1 or in_channels != out_channels:
            # 下采样 Shortcut: k=1, s=stride
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.final_activation = nn.GELU()

    def forward(self, x):
        return self.final_activation(self.conv_main(x) + self.shortcut(x))


class ConvTransBlock(nn.Module):
    """
    标准转置卷积块 (Decoder使用)
    修复核心: 自动计算 Shortcut 的 output_padding 以对齐尺寸
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dropout=0.1):
        super().__init__()

        # 主通路
        self.conv_main = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第二层保持尺寸: k=3, s=1, p=1
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            )

        # Shortcut 处理 (关键修复位置)
        if stride != 1 or in_channels != out_channels:
            # 计算 Shortcut 需要的 output_padding
            # 主通路长度增量: (K - 1)
            # Shortcut(k=1) 长度增量: 0
            # 差异 Diff = K - 1 - 2*padding + output_padding_main
            # 这里 padding=0, op=0, 所以需要补 K-1
            shortcut_op = (kernel_size - 1) + output_padding - 2 * padding

            # 确保 output_padding 小于 stride (这是 PyTorch 的要求)
            # 如果 k=5, s=5, shortcut_op=4 < 5 (合法)
            # 如果 k=2, s=2, shortcut_op=1 < 2 (合法)

            self.shortcut = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                padding=0, output_padding=shortcut_op,
                bias=False
                )
        else:
            self.shortcut = nn.Identity()

        self.final_activation = nn.GELU()

    def forward(self, x):
        return self.final_activation(self.conv_main(x) + self.shortcut(x))


# ==========================================
# 2. 核心 Encoder & Decoder (参数经严格计算)
# ==========================================
class Encoder(nn.Module):
    """
    编码器: 50000 -> 128
    路径: 50000 ->(5)-> 10000 ->(2)-> 5000 ->(5)-> 1000 ->(2)-> 500 ->(5)-> 100 ->(2)-> 50 ->(5)-> 10
    总压缩倍数: 5*2 * 5*2 * 5*2 * 5 = 5000
    """

    def __init__(self, latent_dim=128, dropout=0.0):
        super().__init__()

        # Stage 1: 50000 -> 10000 -> 5000
        self.stage1_conv = ConvBlock(1, 16, kernel_size=5, stride=5, padding=0, dropout=dropout)
        self.stage1_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 2: 5000 -> 1000 -> 500
        self.stage2_conv = ConvBlock(16, 32, kernel_size=5, stride=5, padding=0, dropout=dropout)
        self.stage2_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 3: 500 -> 100 -> 50
        self.stage3_conv = ConvBlock(32, 64, kernel_size=5, stride=5, padding=0, dropout=dropout)
        self.stage3_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 4: 50 -> 10
        self.stage4_conv = ConvBlock(64, 128, kernel_size=5, stride=5, padding=0, dropout=dropout)

        # 投影层: [B, 128, 10] -> [B, 1280] -> [B, Latent]
        self.flatten_dim = 128 * 10
        self.proj = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)  # [B, L] -> [B, 1, L]

        x = self.stage1_pool(self.stage1_conv(x))  # -> 5000
        x = self.stage2_pool(self.stage2_conv(x))  # -> 500
        x = self.stage3_pool(self.stage3_conv(x))  # -> 50
        x = self.stage4_conv(x)  # -> 10

        x = x.view(x.size(0), -1)
        return self.proj(x)


class Decoder(nn.Module):
    """
    解码器: 128 -> 50000
    严格按照 Encoder 的逆序构建
    """

    def __init__(self, output_size=50000, latent_dim=128, dropout=0.0):
        super().__init__()
        self.output_size = output_size

        # 1. 投影回特征图
        self.flatten_dim = 128 * 10
        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        # 2. 逆向 Stage 4: 10 -> 50
        # 对应 Encoder: Conv(k=5, s=5)
        # Decoder: ConvTrans(k=5, s=5, p=0)
        self.up4 = ConvTransBlock(128, 64, kernel_size=5, stride=5, padding=0, output_padding=0, dropout=dropout)

        # 3. 逆向 Stage 3: 50 -> 100 -> 500
        # 对应 Encoder Pool(k=2, s=2) -> ConvTrans(k=2, s=2)
        self.up3_pool = ConvTransBlock(64, 64, kernel_size=2, stride=2, padding=0, output_padding=0, dropout=dropout)
        # 对应 Encoder Conv(k=5, s=5) -> ConvTrans(k=5, s=5)
        self.up3_conv = ConvTransBlock(64, 32, kernel_size=5, stride=5, padding=0, output_padding=0, dropout=dropout)

        # 4. 逆向 Stage 2: 500 -> 1000 -> 5000
        self.up2_pool = ConvTransBlock(32, 32, kernel_size=2, stride=2, padding=0, output_padding=0, dropout=dropout)
        self.up2_conv = ConvTransBlock(32, 16, kernel_size=5, stride=5, padding=0, output_padding=0, dropout=dropout)

        # 5. 逆向 Stage 1: 5000 -> 10000 -> 50000
        self.up1_pool = ConvTransBlock(16, 16, kernel_size=2, stride=2, padding=0, output_padding=0, dropout=dropout)
        self.up1_conv = ConvTransBlock(16, 1, kernel_size=5, stride=5, padding=0, output_padding=0, dropout=dropout)

    def forward(self, z):
        # [B, Latent] -> [B, 128, 10]
        h = self.fc(z).view(z.size(0), 128, 10)

        # Stage 4: 10 -> 50
        h = self.up4(h)

        # Stage 3: 50 -> 100 -> 500
        h = self.up3_pool(h)
        h = self.up3_conv(h)

        # Stage 2: 500 -> 1000 -> 5000
        h = self.up2_pool(h)
        h = self.up2_conv(h)

        # Stage 1: 5000 -> 10000 -> 50000
        h = self.up1_pool(h)
        out = self.up1_conv(h)

        # 最后的安全裁剪/插值 (一般不需要，但为了保险)
        if out.size(2) != self.output_size:
            out = F.interpolate(out, size=self.output_size, mode='linear', align_corners=False)

        return out.squeeze(1)


# ==========================================
# 3. 勒让德宽通路预测头 (保持不变)
# ==========================================
class LegendrePhenoHead(nn.Module):
    """
    输入: 全基因组压缩特征
    输出: 多项式系数 -> 表型曲线
    """

    def __init__(self, total_input_dim, num_phenos, time_coordinates, degree=3, dropout=0.5):
        super().__init__()
        self.num_phenos = num_phenos
        self.degree = degree
        self.coeff_dim = num_phenos * (degree + 1)

        # 生成时间基底 (Buffer)
        t_raw = torch.tensor(time_coordinates, dtype=torch.float32)
        t_min, t_max = t_raw.min(), t_raw.max()
        if t_max > t_min:
            t_norm = 2 * (t_raw - t_min) / (t_max - t_min) - 1
        else:
            t_norm = torch.zeros_like(t_raw)

        basis = torch.zeros(degree + 1, len(t_norm))
        basis[0] = 1.0
        if degree >= 1: basis[1] = t_norm
        for n in range(2, degree + 1):
            term1 = (2 * n - 1) * t_norm * basis[n - 1]
            term2 = (n - 1) * basis[n - 2]
            basis[n] = (term1 - term2) / n
        self.register_buffer('t_basis', basis)

        # 线性预测层
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, self.coeff_dim)
            )

    def forward(self, x):
        coeffs_flat = self.net(x)
        coeffs = coeffs_flat.view(-1, self.num_phenos, self.degree + 1)
        trend = torch.matmul(coeffs, self.t_basis)
        return trend


# ==========================================
# 4. 主模型 (ConvAE)
# ==========================================
class ConvAE(nn.Module):
    def __init__(self,
                 total_snp_num: int,
                 chunk_size: int,
                 latent_dim: int,
                 num_phenotypes: int,
                 time_coordinates: List[float],
                 degree: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        self.chunk_size = chunk_size
        self.num_phenos = num_phenotypes
        self.time_points = len(time_coordinates)

        # A. 共享组件
        self.encoder = Encoder(latent_dim=latent_dim, dropout=dropout)
        self.decoder = Decoder(output_size=chunk_size, latent_dim=latent_dim, dropout=dropout)

        # B. 宽通路预测头
        self.num_chunks = math.ceil(total_snp_num / chunk_size)
        total_feat_dim = self.num_chunks * latent_dim

        print(f"[Model Init] Chunks: {self.num_chunks}, Total Latent Dim: {total_feat_dim}")

        self.pheno_head = LegendrePhenoHead(
            total_input_dim=total_feat_dim,
            num_phenos=num_phenotypes,
            time_coordinates=time_coordinates,
            degree=degree,
            dropout=dropout
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 预测头初始化
        if isinstance(self.pheno_head.net[-1], nn.Linear):
            nn.init.normal_(self.pheno_head.net[-1].weight, mean=0.0, std=1e-5)
            nn.init.constant_(self.pheno_head.net[-1].bias, 0)

    def encode_one_chunk(self, chunk_tensor):
        return self.encoder(chunk_tensor)

    def decode_one_chunk(self, latent_tensor):
        return self.decoder(latent_tensor)

    def predict_from_latents(self, all_chunks_latents):
        B = all_chunks_latents.size(0)
        flat_features = all_chunks_latents.view(B, -1)
        return self.pheno_head(flat_features)