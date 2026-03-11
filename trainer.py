# 导入必要的模块（需要放在文件末尾避免循环导入）
import random
from models import AdvancedMambaAMT, MLP, CombinedMLP, SelfAttention, SelfAttentionMamba, LSTMModel, CombinedLSTM
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# 导入独立的交易模拟器模块
from trading_simulator import get_trading_simulator
from trading_simulator import EnhancedModelTrainer


# ===================== 模型评估和实验执行模块 =====================
def evaluate_model(model, test_x, test_y, criterion, device):
    model.eval()
    with torch.no_grad():
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        
        outputs = model(test_x)
        
        if outputs.shape != test_y.shape:
            outputs = outputs.view(test_y.shape)
        
        loss = criterion(outputs, test_y)
        
        outputs_np = outputs.cpu().numpy()
        targets_np = test_y.cpu().numpy()
        
        rmse = np.sqrt(np.mean((outputs_np - targets_np) ** 2))
        mae = np.mean(np.abs(outputs_np - targets_np))
        r2 = 1 - (np.sum((targets_np - outputs_np) ** 2) / np.sum((targets_np - np.mean(targets_np)) ** 2))
        
        if isinstance(criterion, nn.BCELoss):
            binary_outputs = np.where(outputs_np > 0.5, 1, 0)
            binary_targets = np.where(targets_np > 0.5, 1, 0)
        else:
            binary_outputs = np.where(outputs_np > 0, 1, 0)
            binary_targets = np.where(targets_np > 0, 1, 0)
        
        accuracy = np.mean(binary_outputs == binary_targets)
        
        tp = np.sum((binary_outputs == 1) & (binary_targets == 1))
        tn = np.sum((binary_outputs == 0) & (binary_targets == 0))
        fp = np.sum((binary_outputs == 1) & (binary_targets == 0))
        fn = np.sum((binary_outputs == 0) & (binary_targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'loss': loss.item(),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"Test Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return binary_outputs, metrics
from datetime import datetime  # 新增：导入时间模块
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式：年月日_时分秒

def run_experiment(model_name, model_class, input_dim, train_x, train_y, test_x, test_y, criterion, 
                   is_combined=False, model_type=None, device=None, hyper_params=None, 
                   mamba_config=None, repeat_times=5, test_size=0.25, data_dir='./pre_data', 
                   output_dir=f'./out{TIMESTAMP}', logger=None, trading_config=None):
    all_returns = []
    all_metrics = []
    all_preds = []
    
    for repeat in range(repeat_times):
        logger.log(f"\n=== {model_name} 第 {repeat+1}/{repeat_times} 次实验 ===")
        
        seed = hyper_params['seed'] + repeat
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 特殊处理MambaAMT
        if model_name == "Attention-MambaAMT-Enhanced":
            model = AdvancedMambaAMT(input_dim, input_dim, input_dim, 1, hyper_params['dropout_rate'], device)
        elif is_combined:
            if model_name.startswith('MLP'):
                model = model_class(input_dim, hyper_params['hidden_dim'], hyper_params['dropout_rate'], 
                                   model_type, mamba_config, device)
            elif model_name.startswith('Attention'):
                model = model_class(input_dim, input_dim, input_dim, 1, hyper_params['dropout_rate'], 
                                   model_type, mamba_config, device)
            elif model_name.startswith('LSTM'):
                model = model_class(input_dim, hyper_params['hidden_dim'], hyper_params['lstm_layers'], 1, 
                                   hyper_params['dropout_rate'], model_type, mamba_config, device)
        else:
            if model_name.startswith('LSTM'):
                model = model_class(input_dim, hyper_params['hidden_dim'], hyper_params['lstm_layers'], 1, 
                                   hyper_params['dropout_rate'])
            elif model_name.startswith('Attention'):
                model = model_class(input_dim, input_dim, input_dim, 1, hyper_params['dropout_rate'])
            else:
                model = model_class(input_dim, hyper_params['hidden_dim'], hyper_params['dropout_rate'])
        
        # 使用增强版训练器
        trainer = EnhancedModelTrainer(model, train_x, train_y, criterion, f"{model_name}_repeat_{repeat+1}", 
                                      device, hyper_params, logger.logger)
        trained_model, train_losses = trainer.train()
        
        test_preds, metrics = evaluate_model(trained_model, test_x, test_y, criterion, device)
        all_metrics.append(metrics)
        all_preds.append(test_preds)
        
        returns = []
        CSV_FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('_processed_data.csv')])
        STOCK_CODES = [f.split('_')[0] for f in CSV_FILES]
# ===============================
# 将预测 reshape 成 (股票数, 测试长度)
# ===============================

        num_stocks = len(CSV_FILES)
        test_preds = np.array(test_preds).reshape(num_stocks, -1)

        for i, file_name in enumerate(CSV_FILES):

            # 读取股票数据
            df = pd.read_csv(os.path.join(data_dir, file_name))

            train_size = int(df.shape[0] * (1 - test_size))

            df_test = df.iloc[train_size:].reset_index(drop=True)

            stock_code = file_name.split('_')[0]

            # ===============================
            # 直接按股票取预测
            # ===============================

            pred_flat = test_preds[i].flatten()

            # 如果长度不一致，进行截断
            if len(pred_flat) > len(df_test):
                pred_flat = pred_flat[:len(df_test)]

            # print(stock_code, len(pred_flat))
            # print(pred_flat[:10])

            if len(pred_flat) > 0:

                sim = get_trading_simulator(
                    model_name=model_name,
                    df_scaled=df_test.copy(),
                    y_pred=pred_flat,
                    stock_code=stock_code,
                    trading_config=trading_config,
                    logger=logger.logger
                )

                df_result, total_return = sim.simulate()

                df_result.to_excel(
                    os.path.join(
                        output_dir,
                        f"{model_name}_repeat_{repeat+1}_{stock_code}.xlsx"
                    ),
                    index=False
                )

                returns.append(total_return)

        all_returns.append(returns)
    
    avg_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    
    repeat_results = pd.DataFrame({
        'stock_code': STOCK_CODES,
        'avg_return': avg_returns,
        'std_return': std_returns,
        'avg_accuracy': [avg_metrics['accuracy']] * len(STOCK_CODES),
        'avg_rmse': [avg_metrics['rmse']] * len(STOCK_CODES)
    })
    repeat_results.to_excel(os.path.join(output_dir, f"{model_name}_repeat_results.xlsx"), index=False)
    
    return avg_returns, avg_metrics, all_returns

# 导入必要的模块（需要放在文件末尾避免循环导入）
import random
from models import AdvancedMambaAMT, MLP, CombinedMLP, SelfAttention, SelfAttentionMamba, LSTMModel, CombinedLSTM