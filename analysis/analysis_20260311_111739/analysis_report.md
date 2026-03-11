# Model Experiment Results Analysis Report
==================================================

## Analysis Information
- Analysis Time: 2026-03-11 11:17:50
- Results Storage Directory: ./analysis/analysis_20260311_111739
- Raw Data Backup Directory: ./analysis/analysis_20260311_111739/raw_data

## Filter Conditions

## 1. Overall Statistics
- Number of Models in Comparison: 13
- Number of Tested Stocks: 6
- Average Return Range: 0.4638 ~ 2.5368
- Average Accuracy Range: 0.5037 ~ 0.8507

## 2. Best Models by Metrics

### 2.1 Best Return
- Model: MLP-S4
- Average Return: 2.5368
- Return Standard Deviation: 0.0936
- Accuracy: 0.6061

### 2.2 Best Accuracy
- Model: MLP-mamba2
- Average Accuracy: 0.8507
- Average Return: 1.9368

### 2.3 Lowest RMSE
- Model: Attention-MambaAMT
- Average RMSE: 0.0321
- Average Return: 1.1812

## 3. Model Category Comparison

### 3.1 MLP Category Models
- Average Return: 2.1878 (±0.2522)
- Maximum Return: 2.5368
- Average Accuracy: 0.7122 (±0.1195)
- Average RMSE: 0.4395 (±0.0592)

### 3.1 Attention Category Models
- Average Return: 0.7830 (±0.2592)
- Maximum Return: 1.1812
- Average Accuracy: 0.5407 (±0.0417)
- Average RMSE: 0.0581 (±0.0512)

### 3.1 LSTM Category Models
- Average Return: 1.2807 (±0.5976)
- Maximum Return: 1.8854
- Average Accuracy: 0.5309 (±0.0459)
- Average RMSE: 0.0381 (±0.0104)

## 4. Risk-Return Analysis

### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)
- Model: LSTM-mamba2
- Sharpe Ratio: 1.0787
- Return: 0.5749
- Risk (Standard Deviation): 0.5330

### 5.1 Performance of Best Model (MLP-S4) on Different Stocks
- Best Performing Stock: 7.0 (Return: 10.0648)
- Worst Performing Stock: 2.0 (Return: 0.0000)
- Return Standard Deviation Across Stocks: 3.9069