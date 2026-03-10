# Model Experiment Results Analysis Report
==================================================

## Analysis Information
- Analysis Time: 2026-03-10 22:00:21
- Results Storage Directory: ./analysis/analysis_20260310_220010
- Raw Data Backup Directory: ./analysis/analysis_20260310_220010/raw_data

## Filter Conditions

## 1. Overall Statistics
- Number of Models in Comparison: 13
- Number of Tested Stocks: 6
- Average Return Range: 0.8518 ~ 1.9643
- Average Accuracy Range: 0.4896 ~ 0.8883

## 2. Best Models by Metrics

### 2.1 Best Return
- Model: MLP-S4
- Average Return: 1.9643
- Return Standard Deviation: 0.0642
- Accuracy: 0.6091

### 2.2 Best Accuracy
- Model: MLP-mamba2
- Average Accuracy: 0.8883
- Average Return: 1.7190

### 2.3 Lowest RMSE
- Model: Attention-MambaAMT
- Average RMSE: 0.0321
- Average Return: 1.3259

## 3. Model Category Comparison

### 3.1 MLP Category Models
- Average Return: 1.8457 (±0.1016)
- Maximum Return: 1.9643
- Average Accuracy: 0.7251 (±0.1384)
- Average RMSE: 0.4395 (±0.0592)

### 3.1 Attention Category Models
- Average Return: 1.2341 (±0.3703)
- Maximum Return: 1.7843
- Average Accuracy: 0.5585 (±0.0596)
- Average RMSE: 0.0582 (±0.0513)

### 3.1 LSTM Category Models
- Average Return: 1.3184 (±0.2395)
- Maximum Return: 1.5478
- Average Accuracy: 0.5209 (±0.0368)
- Average RMSE: 0.0381 (±0.0104)

## 4. Risk-Return Analysis

### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)
- Model: Attention-S4
- Sharpe Ratio: 0.7228
- Return: 0.8518
- Risk (Standard Deviation): 1.1785

### 5.1 Performance of Best Model (MLP-S4) on Different Stocks
- Best Performing Stock: 7.0 (Return: 8.1228)
- Worst Performing Stock: 2.0 (Return: 0.0000)
- Return Standard Deviation Across Stocks: 3.1298