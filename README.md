


<p align="center">
    <a href="https://github.com/gaozhihong73/DynaGP/releases">
    	<img src="https://img.shields.io/github/v/release/gaozhihong73/DynaGP?style=flat-plastic&color=007acc&include_prereleases" alt="Release">
	</a>
    <a href="https://github.com/gaozhihong73/DynaGP/blob/main/LICENSE">
    	<img src="https://img.shields.io/github/license/gaozhihong73/DynaGP?style=flat-plastic&color=42b883" alt="License">
  	</a> 
    <a href="https://github.com/gaozhihong73/DynaGP/commits/main">
    	<img src="https://img.shields.io/github/last-commit/gaozhihong73/DynaGP?style=flat-plastic&color=fb8c00" alt="Last Commit">
  	</a>
    <img src="https://img.shields.io/github/languages/top/gaozhihong73/DynaGP?style=flat-plastic&color=f1e05a" alt="languagges">
  	<a href="https://github.com/gaozhihong73/DynaGP/stargazers">
    	<img src="https://img.shields.io/github/stars/gaozhihong73/DynaGP?style=flat-plastic&color=dfb317" alt="Stars">
  	</a>
</p>



# DynaGS

## Installation

```bash
# Create a new conda environment
conda create -n DynaGS python=3.10.11

# Activate the environment
conda activate DynaGS

# Install DynaGS
source /etc/network_turbo  # 若使用的是AutoDL服务器，先执行这句再克隆
git clone https://github.com/gaozhihong73/DynaGP.git
cd ./DynaGS

# Install dependencies
pip install -r requirements.txt
```

## Requirement

```
python==3.10.11
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
numpy==1.23.5
scipy==1.10.1
pandas==2.0.3
scikit-learn==1.3.2
transformers==4.36.2
matplotlib==3.8.2
seaborn==0.13.2
opencv-python-headless==4.9.0.80
pyarrow==8.0.0
torchinfo==1.8.0
tqdm==4.66.1
```

## Options and usage

### Training

#### Usage

```bash
# Training with Default Config
python train.py

# Training with Custom Settings
python train.py \
  --exp_name maize_single_train \
  --snp_file dataset/maize/snp/snp_50000.npy \
  --pheno_dir dataset/maize/cleaned_pheno \
  --pheno_names RGBVI WI ExB COM GreenCoverage bn IPCA NDI \
  --time_coordinates 10 12 14 15 17 19 24 26 37 44 51 65 80 101 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --dropout 0.3 \
  --use_warmup True \
  --use_kfold False
```

#### Optional Parameters

| **Parameter**        | **Description**                                              | **Default Value**                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `--exp_name`         | Experiment name to distinguish differe                       | Timestamp                                        |
| `--snp_file`         | Path to the SNP genotype data file (.npy format)             | dataset/maize/snp/snp_50000.npy                  |
| `--pheno_dir`        | Directory containing the phenotype data files                | dataset/maize/cleaned_pheno                      |
| `--test_ratio`       | Ratio of the dataset to be used for testing                  | 0.1                                              |
| `--val_ratio`        | Ratio of the dataset to be used for validation               | 0.1                                              |
| `--pheno_names`      | List of phenotype names to predict (must match filenames)    | RGBVI WI ExB COM GreenCoverage bn IPCA NDI       |
| `--time_coordinates` | Real-world time coordinates for phenotype sampling           | 10 12 14 15 17 19 24 <br />26 37 44 51 65 80 101 |
| `--norm_method`      | Strategy for phenotype normalization (global/timepoint/residual) | global                                           |
| `--scaler_type`      | Type of scaler function (zscore/minmax/robust)               | zscore                                           |
| `--epochs`           | Maximum number of training iterations                        | 100                                              |
| `--batch_size`       | Number of samples per training batch                         | 16                                               |
| `--lr`               | Initial learning rate                                        | 0.0001                                           |
| `--weight_decay`     | Weight decay coefficient                                     | 1e-4                                             |
| `--dropout`          | Dropout rate for neural network layers                       | 0.3                                              |
| `--patience`         | Patience for early stopping (number of epochs without improvement) | 10                                               |
| `--seed`             | Global random seed for reproducibility                       | 73                                               |
| `--is_save`          | Whether to save the trained model and history locally        | True                                             |
| `--use_warmup`       | Whether to enable the learning rate warmup strategy          | True                                             |
| `--warmup_epochs`    | Number of epochs for the warmup phase                        | 8                                                |
| `--d_model`          | Hidden dimension size of the Transformer                     | 64                                               |
| `--num_heads`        | Number of heads in the multi-head attention mechanism        | 4                                                |
| `--num_layers`       | Number of encoder layers (network depth)                     | 2                                                |
| `--d_ff`             | Dimension of the feed-forward network (FFN) hidden layer     | 256                                              |
| `--use_kfold`        | Whether to enable K-fold cross-validation                    | False                                            |
| `--n_folds`          | Number of folds for K-fold cross-validation                  | 10                                               |

### Prediction

#### Usage

```bash
# Inference with Default Test Set
python predict.py

# Inference with External Data
python predict.py \
  --model_path outputs/tfpp/exp_2026/models/best_model.pth \
  --scaler_path outputs/tfpp/exp_2026/scalers/scaler.pkl \
  --output_csv outputs/predict/testset_predictions.csv \
  --input_snp dataset/new_samples.npy
```

#### Optional Parameters

| **Parameter**        | **Description**                                              | **Default Value**                                  |
| -------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| `--model_path`       | Path to the trained model file (.pth)                        | (Required)                                         |
| `--scaler_path`      | Path to the phenotype scaler file (.pkl)                     | (Required)                                         |
| `--input_snp`        | Path to external SNP data (.npy) for prediction; if None, uses default test set | None                                               |
| `--output_csv`       | Path to save the prediction results in CSV format            | outputs/predict/<br />predictions_{timestamp}.csv` |
| `--snp_file`         | Path to the original SNP data used during training           | dataset/maize/snp/snp_50000.npy                    |
| `--pheno_dir`        | Directory containing the original phenotype data             | dataset/maize/cleaned_pheno                        |
| `--test_ratio`       | Ratio of the dataset used for testing (must match training)  | 0.1                                                |
| `--val_ratio`        | Ratio of the dataset used for validation (must match training) | 0.1                                                |
| `--seed`             | Random seed for dataset splitting (must match training)      | 73                                                 |
| `--scaler_type`      | Type of scaler function used during training (must match training) | zscore                                             |
| `--time_coordinates` | Real-world time coordinates for phenotype sampling (must match training) | 10 12 14 15 17 19 24 <br />26 37 44 51 65 80 101   |
| `--batch_size`       | Number of samples per inference batch                        | 32                                                 |
| `--d_model`          | Hidden dimension size of the Transformer (must match training) | 64                                                 |
| `--num_heads`        | Number of heads in the multi-head attention mechanism (must match training) | 4                                                  |
| `--num_layers`       | Number of encoder layers (must match training)               | 2                                                  |
| `--d_ff`             | Dimension of the feed-forward network (FFN) hidden layer (must match training) | 256                                                |