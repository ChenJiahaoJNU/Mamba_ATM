# Model Experiment Results Analysis Report
==================================================

## Analysis Information
- Analysis Time: 2026-03-11 15:49:39
- Results Storage Directory: ./analysis/analysis_20260311_154926
- Raw Data Backup Directory: ./analysis/analysis_20260311_154926/raw_data

## Filter Conditions

## 1. Overall Statistics
- Number of Models in Comparison: 13
- Number of Tested Stocks: 6
- Average Return Range: 0.3866 ~ 1.1457
- Average Accuracy Range: 0.5037 ~ 0.8507

## 2. Best Models by Metrics

### 2.1 Best Return
- Model: MLP_Base
- Average Return: 1.1457
- Return Standard Deviation: 0.0009
- Accuracy: 0.6189

### 2.2 Best Accuracy
- Model: MLP-mamba2
- Average Accuracy: 0.8507
- Average Return: 0.9209

### 2.3 Lowest RMSE
- Model: Attention-MambaAMT
- Average RMSE: 0.0321
- Average Return: 0.7383

## 3. Model Category Comparison

### 3.1 MLP Category Models
- Average Return: 1.0210 (±0.1031)
- Maximum Return: 1.1457
- Average Accuracy: 0.7122 (±0.1195)
- Average RMSE: 0.4395 (±0.0592)

### 3.1 Attention Category Models
- Average Return: 0.7717 (±0.2417)
- Maximum Return: 1.0386
- Average Accuracy: 0.5410 (±0.0415)
- Average RMSE: 0.0582 (±0.0513)

### 3.1 LSTM Category Models
- Average Return: 0.6892 (±0.2214)
- Maximum Return: 0.9110
- Average Accuracy: 0.5309 (±0.0459)
- Average RMSE: 0.0381 (±0.0104)

## 4. Risk-Return Analysis

### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)
- Model: Attention-mamba2
- Sharpe Ratio: 0.6665
- Return: 0.8192
- Risk (Standard Deviation): 1.2292

### 5.1 Performance of Best Model (MLP_Base) on Different Stocks
- Best Performing Stock: 7.0 (Return: 4.9898)
- Worst Performing Stock: 2.0 (Return: 0.0000)
- Return Standard Deviation Across Stocks: 1.9170