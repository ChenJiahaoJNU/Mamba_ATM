# Model Experiment Results Analysis Report
==================================================

## Analysis Information
- Analysis Time: 2026-03-11 16:52:43
- Results Storage Directory: ./analysis/analysis_20260311_165228
- Raw Data Backup Directory: ./analysis/analysis_20260311_165228/raw_data

## Filter Conditions

## 1. Overall Statistics
- Number of Models in Comparison: 13
- Number of Tested Stocks: 6
- Average Return Range: 0.3897 ~ 1.1473
- Average Accuracy Range: 0.5037 ~ 0.8507

## 2. Best Models by Metrics

### 2.1 Best Return
- Model: MLP_Base
- Average Return: 1.1473
- Return Standard Deviation: 0.0025
- Accuracy: 0.6189

### 2.2 Best Accuracy
- Model: MLP-mamba2
- Average Accuracy: 0.8507
- Average Return: 0.9310

### 2.3 Lowest RMSE
- Model: Attention-MambaAMT
- Average RMSE: 0.0321
- Average Return: 0.7405

## 3. Model Category Comparison

### 3.1 MLP Category Models
- Average Return: 1.0227 (±0.0967)
- Maximum Return: 1.1473
- Average Accuracy: 0.7122 (±0.1195)
- Average RMSE: 0.4395 (±0.0592)

### 3.1 Attention Category Models
- Average Return: 0.7682 (±0.2364)
- Maximum Return: 1.0277
- Average Accuracy: 0.5406 (±0.0419)
- Average RMSE: 0.0581 (±0.0512)

### 3.1 LSTM Category Models
- Average Return: 0.7007 (±0.2047)
- Maximum Return: 0.9015
- Average Accuracy: 0.5309 (±0.0459)
- Average RMSE: 0.0381 (±0.0104)

## 4. Risk-Return Analysis

### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)
- Model: Attention-MambaAMT
- Sharpe Ratio: 0.6658
- Return: 0.7405
- Risk (Standard Deviation): 1.1121

### 5.1 Performance of Best Model (MLP_Base) on Different Stocks
- Best Performing Stock: 7.0 (Return: 4.9898)
- Worst Performing Stock: 2.0 (Return: 0.0000)
- Return Standard Deviation Across Stocks: 1.9162