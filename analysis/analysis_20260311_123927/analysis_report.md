# Model Experiment Results Analysis Report
==================================================

## Analysis Information
- Analysis Time: 2026-03-11 12:39:37
- Results Storage Directory: ./analysis/analysis_20260311_123927
- Raw Data Backup Directory: ./analysis/analysis_20260311_123927/raw_data

## Filter Conditions
- Excluded Specific Models: ['.']

## 1. Overall Statistics
- Number of Models in Comparison: 13
- Number of Tested Stocks: 6
- Average Return Range: 0.4291 ~ 1.8173
- Average Accuracy Range: 0.5037 ~ 0.8507

## 2. Best Models by Metrics

### 2.1 Best Return
- Model: MLP-S4
- Average Return: 1.8173
- Return Standard Deviation: 0.0578
- Accuracy: 0.6061

### 2.2 Best Accuracy
- Model: MLP-mamba2
- Average Accuracy: 0.8507
- Average Return: 1.0678

### 2.3 Lowest RMSE
- Model: Attention-MambaAMT
- Average RMSE: 0.0321
- Average Return: 0.9107

## 3. Model Category Comparison

### 3.1 MLP Category Models
- Average Return: 1.4231 (±0.3617)
- Maximum Return: 1.8173
- Average Accuracy: 0.7122 (±0.1195)
- Average RMSE: 0.4395 (±0.0592)

### 3.1 Attention Category Models
- Average Return: 0.6608 (±0.1823)
- Maximum Return: 0.9107
- Average Accuracy: 0.5407 (±0.0417)
- Average RMSE: 0.0582 (±0.0513)

### 3.1 LSTM Category Models
- Average Return: 1.0569 (±0.4868)
- Maximum Return: 1.4885
- Average Accuracy: 0.5309 (±0.0459)
- Average RMSE: 0.0381 (±0.0104)

## 4. Risk-Return Analysis

### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)
- Model: LSTM-mamba2
- Sharpe Ratio: 0.7486
- Return: 0.4758
- Risk (Standard Deviation): 0.6356

### 5.1 Performance of Best Model (MLP-S4) on Different Stocks
- Best Performing Stock: 7.0 (Return: 7.6926)
- Worst Performing Stock: 2.0 (Return: 0.0000)
- Return Standard Deviation Across Stocks: 2.9733