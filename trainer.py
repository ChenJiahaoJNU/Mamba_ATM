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

# ===================== 增强版模型训练器（适配MambaAMT） =====================
class EnhancedModelTrainer:
    """增强版训练器，为MambaAMT定制优化策略"""
    def __init__(self, model, train_x, train_y, criterion, model_name, device, hyper_params, logger):
        self.model = model.to(device)
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.criterion = criterion
        self.model_name = model_name
        self.device = device
        self.hyper_params = hyper_params
        self.logger = logger
        
    def train(self):
        # MambaAMT专属优化器配置
        if "MambaAMT" in self.model_name:
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.hyper_params['mambaamt_lr'],
                weight_decay=self.hyper_params['mambaamt_weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.hyper_params['num_epochs'],
                eta_min=self.hyper_params['mambaamt_lr'] * 0.01
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.hyper_params['learning_rate'],
                weight_decay=self.hyper_params['weight_decay']
            )
            scheduler = StepLR(
                optimizer, 
                step_size=self.hyper_params['step_size'],
                gamma=self.hyper_params['gamma']
            )
        
        train_losses = []
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.hyper_params['num_epochs']):
            self.model.train()
            optimizer.zero_grad()
            
            if epoch == 0:
                self.logger.write(f"{self.model_name} 输入形状: {self.train_x.shape}\n")
            
            outputs = self.model(self.train_x)
            
            if epoch == 0:
                self.logger.write(f"{self.model_name} 输出形状: {outputs.shape}\n")
                self.logger.write(f"{self.model_name} 标签形状: {self.train_y.shape}\n")
                if torch.isnan(outputs).any():
                    self.logger.write(f"警告：{self.model_name} 输出包含NaN值！\n")
                    print(f"警告：{self.model_name} 输出包含NaN值！")
                self.logger.write(f"{self.model_name} 输出范围: [{torch.min(outputs):.4f}, {torch.max(outputs):.4f}]\n")
                self.logger.write(f"{self.model_name} 标签范围: [{torch.min(self.train_y):.4f}, {torch.max(self.train_y):.4f}]\n")
            
            if outputs.shape != self.train_y.shape:
                outputs = outputs.view(self.train_y.shape)
            
            loss = self.criterion(outputs, self.train_y)
            
            # MambaAMT梯度裁剪更严格
            if "MambaAMT" in self.model_name:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            # 早停机制（仅MambaAMT）
            if "MambaAMT" in self.model_name:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), f'best_mambaamt_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.write(f"早停触发，停止训练在epoch {epoch+1}\n")
                        break
            
            if (epoch + 1) % 10 == 0:
                self.logger.write(f"[{self.model_name}] Epoch {epoch + 1}/{self.hyper_params['num_epochs']}, Loss: {loss.item():.4f}\n")
                self.logger.flush()
        
        # 加载MambaAMT最佳模型
        if "MambaAMT" in self.model_name and os.path.exists('best_mambaamt_model.pth'):
            self.model.load_state_dict(torch.load('best_mambaamt_model.pth'))
            self.logger.write(f"加载MambaAMT最佳模型，最佳损失: {best_loss:.4f}\n")
        
        return self.model, train_losses


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


def run_experiment(model_name, model_class, input_dim, train_x, train_y, test_x, test_y, criterion, 
                   is_combined=False, model_type=None, device=None, hyper_params=None, 
                   mamba_config=None, repeat_times=5, test_size=0.25, data_dir='./pre_data', 
                   output_dir='./out', logger=None, trading_config=None):
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
        
        for i, file_name in enumerate(CSV_FILES):
            df = pd.read_csv(os.path.join(data_dir, file_name))
            train_size = int(df.shape[0] * (1 - test_size))
            df_test = df.iloc[train_size:, :].reset_index(drop=True)
            
            stock_code = file_name.split('_')[0]
            pred_flat = test_preds[i].flatten()[:len(df_test)]
            
            if len(pred_flat) > 0:
                # 【核心修改】使用工厂函数获取对应的交易模拟器
                sim = get_trading_simulator(
                    model_name=model_name,
                    df_scaled=df_test.copy(),
                    y_pred=pred_flat,
                    stock_code=stock_code,
                    trading_config=trading_config,
                    logger=logger.logger
                )
                df_result, total_return = sim.simulate()
                
                df_result.to_excel(os.path.join(output_dir, f"{model_name}_repeat_{repeat+1}_{stock_code}.xlsx"), index=False)
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