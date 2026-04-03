from datetime import datetime

import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from model.TFPP import TFPP
from dataloader.TFPP_DataLoader import TFPPDataset, get_dataloader
from utils import load_scalers, denormalize_phenotype


def parse_args():
    parser = argparse.ArgumentParser(description="DynaGS: TFPP 模型预测脚本 (支持外部文件或默认测试集)")

    # ================= 核心必填路径 =================
    parser.add_argument('--model_path', type=str, default='outputs/tfpp/exp_20260403_111343/models/best_model.pth',
                        help='训练好的模型文件路径 (.pth)')
    parser.add_argument('--scaler_path', type=str, default='outputs/tfpp/exp_20260403_111343/scalers/scaler.pkl',
                        help='归一化器文件路径 (.pkl)')

    # ================= 预测模式控制 =================
    parser.add_argument('--input_snp', type=str, default=None,
                        help='外部待预测的 SNP 数据路径 (.npy)。如果不传入此项，则默认预测训练时的测试集')
    parser.add_argument('--output', type=str,
                        default=f'outputs/predict/predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        help='预测结果保存路径')

    # ================= 数据划分配置 (当不传入 input_snp 时用于复现测试集) =================
    parser.add_argument('--snp_file', type=str, default='dataset/maize/snp/snp_50000.npy', help='训练时的 SNP 数据文件')
    parser.add_argument('--pheno_dir', type=str, default='dataset/maize/cleaned_pheno', help='训练时的表型数据目录')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (需与训练时一致)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例 (需与训练时一致)')
    parser.add_argument('--seed', type=int, default=73, help='随机种子 (需与训练时一致)')
    parser.add_argument('--scaler_type', type=str, default='zscore', help='归一化缩放器类型')

    # ================= 模型结构参数 (需与训练时一致) =================
    parser.add_argument('--time_coordinates', type=float, nargs='+',
                        default=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101],
                        help='采样时间点坐标列表')
    parser.add_argument('--batch_size', type=int, default=32, help='推理批次大小')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=256)

    return parser.parse_args()


def predict():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path("outputs/predict").mkdir(parents=True, exist_ok=True)

    # 1. 加载归一化器信息
    print(f"[*] 正在加载归一化器: {args.scaler_path}")
    scalers_pkg = load_scalers(args.scaler_path)
    phenotype_names = scalers_pkg['phenotype_names']
    norm_method = scalers_pkg['norm_method']

    # 2. 判断输入模式以获取 SNP 维度
    # 如果是预测测试集，需要先加载 dataset 才能知道维度；如果是外部文件，直接读取 numpy
    if args.input_snp:
        print(f"[*] 检测到外部输入文件: {args.input_snp}")
        if not Path(args.input_snp).exists():
            raise FileNotFoundError(f"SNP文件不存在: {args.input_snp}")
        external_snp_data = np.load(args.input_snp)
        snp_dim = external_snp_data.shape[1]
    else:
        print(f"[*] 未检测到外部输入文件，将复现测试集进行预测 (Seed: {args.seed})...")
        dataset = TFPPDataset(args.snp_file, args.pheno_dir, phenotype_names)
        snp_dim = dataset.snp.shape[1]

        _, _, test_loader, _ = get_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            norm_method=norm_method,
            scaler_type=args.scaler_type,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=device.type,
            num_workers=4
            )

    # 3. 初始化模型结构并加载权重
    print("[*] 正在初始化模型并加载权重...")
    model = TFPP(
        snp_dim=snp_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_embeddings=4,
        time_coordinates=args.time_coordinates,
        phenotype_names=phenotype_names,
        dropout=0.0  # 推理时关闭 Dropout
        ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. 执行推理并反归一化
    results = []

    if args.input_snp:
        # 模式 A: 预测外部文件
        snp_tensor = torch.tensor(external_snp_data + 1, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_norm = model(snp_tensor)

        preds_denorm = denormalize_phenotype(preds_norm.cpu(), scalers_pkg)
        num_samples, num_phenos, num_times = preds_denorm.shape

        for i in range(num_samples):
            for p_idx, p_name in enumerate(phenotype_names):
                row = {'sample_idx': i, 'phenotype': p_name}
                for t_idx in range(num_times):
                    row[f'T{t_idx + 1}'] = preds_denorm[i, p_idx, t_idx].item()
                results.append(row)

    else:
        # 模式 B: 预测内部测试集
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                preds_norm = model(X)
                all_preds.append(preds_norm.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        preds_denorm = denormalize_phenotype(all_preds, scalers_pkg)
        targets_denorm = denormalize_phenotype(all_targets, scalers_pkg)

        num_samples, num_phenos, num_times = preds_denorm.shape
        for i in range(num_samples):
            for p_idx, p_name in enumerate(phenotype_names):
                row = {'sample_idx': i, 'phenotype': p_name}
                for t_idx in range(num_times):
                    row[f'T{t_idx + 1}_Pred'] = preds_denorm[i, p_idx, t_idx].item()
                    row[f'T{t_idx + 1}_True'] = targets_denorm[i, p_idx, t_idx].item()
                results.append(row)

    # 5. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[*] 预测完成！结果已保存至: {args.output}")


if __name__ == "__main__":
    predict()
