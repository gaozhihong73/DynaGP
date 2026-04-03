import argparse
from config import TFPPConfig
from trainer import TFPPTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TFPP model')

    parser.add_argument('--exp_name', type=str, default=None, help='实验名称，默认为时间戳格式')
    parser.add_argument('--snp_file', type=str, default='dataset/maize/snp/snp_50000.npy',
                        help='SNP 数据文件路径 (.npy)')
    parser.add_argument('--pheno_dir', type=str, default='dataset/maize/cleaned_pheno',
                        help='表型数据所在目录')

    # ================= 数据集与特征配置 =================
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--pheno_names', type=str, nargs='+',
                        default=['RGBVI', 'WI', 'ExB', 'COM', 'GreenCoverage', 'bn', 'IPCA', 'NDI'],
                        help='需要预测的表型名称列表，需要与表型文件名对应，例如: --phenotype_names Yield PH GW ')
    parser.add_argument('--time_coordinates', type=float, nargs='+',
                        default=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101],
                        help='采样时间点坐标列表，例如: --time_coordinates 10 14 26')
    parser.add_argument('--norm_method', type=str, default='global',
                        choices=['global', 'timepoint', 'residual_global'],
                        help='表型归一化策略')
    parser.add_argument('--scaler_type', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'robust'],
                        help='归一化缩放器类型')

    # ================= 训练超参数 =================
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减 (L2正则化)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout 比例')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值 (连续多少轮无提升则停止)')
    parser.add_argument('--seed', type=int, default=73, help='随机种子')
    parser.add_argument('--is_save', type=bool, default=True, help='是否保存训练数据到本地')
    parser.add_argument('--use_warmup', type=bool, default=True, help='是否使用Warmup')
    parser.add_argument('--warmup_epochs', type=int, default=8, help='warmup轮数')

    # ================= 模型参数 =================
    parser.add_argument('--d_model', type=int, default=64, help='特征维度')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力机制的头数')
    parser.add_argument('--num_layers', type=int, default=2, help='网络深度')
    parser.add_argument('--d_ff', type=int, default=256, help='前馈网络隐藏层维度')

    # ================= 验证策略 =================
    # 使用 --use_kfold 开启 K 折交叉验证，使用 --no_kfold 则使用常规的 train/val/test 划分
    parser.add_argument('--use_kfold', type=bool, default=False, help='是否使用 K 折交叉验证')
    parser.add_argument('--n_folds', type=int, default=10, help='K 折交叉验证的折数')

    # 解析命令行输入的参数
    return parser.parse_args()


def main():
    # 1. 获取命令行参数
    args = parse_args()

    # 2. 将命令行参数转化为配置对象 (TFPPConfig)
    config = TFPPConfig(
        exp_name=args.exp_name,
        snp_file=args.snp_file,
        pheno_dir=args.pheno_dir,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        phenotype_names=args.pheno_names,
        time_coordinates=args.time_coordinates,
        norm_method=args.norm_method,
        scaler_type=args.scaler_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        is_save_training_info=args.is_save,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        use_kfold=args.use_kfold,
        n_folds=args.n_folds
        )

    # 3. 初始化训练器
    print(f"[*] 正在初始化 TFPPTrainer，实验名称: {config.experiment_name}")
    trainer = TFPPTrainer(config=config)

    # 4. 开始训练（Trainer 内部会自动在训练结束后执行测试集的评估）
    print("[*] 开始执行训练流程...")
    results = trainer.train()
    print("[*] 流程执行完毕。")


if __name__ == "__main__":
    main()
