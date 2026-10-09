import pandas as pd
import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import os
import random
warnings.filterwarnings('ignore')

# ===================== 数据预处理工具（修复BCELoss标签问题） =====================
def convert_returns_to_binary_labels(returns, threshold=0.0):
    binary_labels = np.where(returns > threshold, 1.0, 0.0)
    return binary_labels

def validate_label_range(labels, loss_type='bce'):
    if loss_type == 'bce':
        labels = np.clip(labels, 0.0, 1.0)
        unique_vals = np.unique(labels)
        print(f"BCE标签值范围: [{np.min(labels):.4f}, {np.max(labels):.4f}]")
        print(f"唯一标签值: {unique_vals[:10]}")
    return labels

# ===================== 数据统计与日志模块 =====================
class DataStatistics:
    def __init__(self, log_dir):
        self.stats = {}
        self.logger = self._init_logger(log_dir)
    
    def _init_logger(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        return open(log_file, 'w', encoding='utf-8')
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        self.logger.write(log_msg + '\n')
        self.logger.flush()
    
    def collect_data_stats(self, stock_data, stock_labels, file_names, stock_codes):
        self.log("=== 数据统计信息 ===")
        
        self.stats['total_stocks'] = len(stock_codes)
        self.stats['total_files'] = len(file_names)
        self.stats['feature_dim'] = stock_data.shape[2]
        self.stats['total_time_steps'] = stock_data.shape[1]
        
        stock_stats = {}
        for i, code in enumerate(stock_codes):
            df = pd.read_csv(os.path.join('./pre_data', file_names[i]))
            
            date_col = df['日期'] if '日期' in df.columns else None
            if date_col is not None:
                start_date = pd.to_datetime(date_col.min())
                end_date = pd.to_datetime(date_col.max())
                time_span = (end_date - start_date).days
            else:
                start_date = end_date = "未知"
                time_span = "未知"
            
            sample_size = len(df)
            returns = df['涨跌幅'].values if '涨跌幅' in df.columns else None
            if returns is not None:
                return_stats = {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'positive_ratio': np.mean(returns > 0)
                }
                binary_returns = convert_returns_to_binary_labels(returns)
                return_stats['binary_positive_ratio'] = np.mean(binary_returns)
            else:
                return_stats = None
            
            stock_stats[code] = {
                'file_name': file_names[i],
                'sample_size': sample_size,
                'start_date': start_date,
                'end_date': end_date,
                'time_span_days': time_span,
                'return_statistics': return_stats
            }
        
        self.stats['stock_details'] = stock_stats
        
        self.log(f"总股票数量: {self.stats['total_stocks']}")
        self.log(f"总特征维度: {self.stats['feature_dim']}")
        self.log(f"总时间步数: {self.stats['total_time_steps']}")
        self.log(f"股票列表: {stock_codes}")
        
        for code, details in stock_stats.items():
            self.log(f"\n股票 {code}:")
            self.log(f"  文件名: {details['file_name']}")
            self.log(f"  样本数量: {details['sample_size']}")
            self.log(f"  时间区间: {details['start_date']} 至 {details['end_date']}")
            self.log(f"  时间跨度: {details['time_span_days']} 天")
            if details['return_statistics']:
                rs = details['return_statistics']
                self.log(f"  涨跌幅统计 - 均值: {rs['mean']:.4f}, 标准差: {rs['std']:.4f}")
                self.log(f"  涨跌幅统计 - 最小值: {rs['min']:.4f}, 最大值: {rs['max']:.4f}")
                self.log(f"  上涨比例: {rs['positive_ratio']:.2%}")
                self.log(f"  二值化后上涨比例: {rs['binary_positive_ratio']:.2%}")
        
        os.makedirs('./out', exist_ok=True)
        stats_df = pd.DataFrame.from_dict(stock_stats, orient='index')
        stats_df.to_excel(os.path.join('./out', 'data_statistics.xlsx'))
        
        return self.stats

# ===================== 数据加载模块（增强时间序列划分） =====================
class DataLoader:
    @staticmethod
    def load_and_process_data(data_dir, return_type='binary', seed=42):
        # 创建测试数据（如果没有真实数据）
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            # 生成2个测试股票文件
            for code in ['000001', '000002']:
                dates = pd.date_range('2020-01-01', periods=500)
                df = pd.DataFrame({
                    '日期': dates,
                    '涨跌幅': np.random.randn(500) * 0.01,
                    '当日开盘': np.cumsum(np.random.randn(500) * 0.01) + 10,
                    'feature1': np.random.randn(500),
                    'feature2': np.random.randn(500),
                    'feature3': np.random.randn(500)
                })
                df.to_csv(os.path.join(data_dir, f'{code}_processed_data.csv'), index=False)
        
        CSV_FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('_processed_data.csv')])
        STOCK_CODES = [f.split('_')[0] for f in CSV_FILES]
        
        stock_data, stock_labels = [], []
        scaler = StandardScaler()
        
        for file_name in CSV_FILES:
            file_path = os.path.join(data_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                
                if '涨跌幅' not in df.columns or '日期' not in df.columns:
                    raise ValueError(f"文件 {file_name} 缺少必要列：涨跌幅 或 日期")
                
                raw_returns = df['涨跌幅'].values.reshape(-1, 1)
                
                if return_type == 'binary':
                    labels = convert_returns_to_binary_labels(raw_returns)
                    labels = validate_label_range(labels, loss_type='bce')
                else:
                    labels = raw_returns
                
                features = df.drop(columns=['涨跌幅', '日期']).values
                features = scaler.fit_transform(features)
                
                stock_labels.append(labels)
                stock_data.append(features)
                
            except Exception as e:
                print(f"读取文件 {file_name} 时出错: {e}")
                continue
        
        stock_data_np = np.stack(stock_data, axis=0)
        stock_labels_np = np.stack(stock_labels, axis=0)
        
        data_stats = DataStatistics('./logs')
        data_stats.collect_data_stats(stock_data_np, stock_labels_np, CSV_FILES, STOCK_CODES)
        
        print(f"成功加载 {len(stock_data)} 个股票文件")
        print(f"数据形状: {stock_data_np.shape}, 标签形状: {stock_labels_np.shape}")
        print(f"标签类型: {return_type}, 标签范围: [{np.min(stock_labels_np):.4f}, {np.max(stock_labels_np):.4f}]")
        
        return stock_data_np, stock_labels_np, CSV_FILES, STOCK_CODES
    
    @staticmethod
    def split_data(data, labels, test_size=0.25, time_series_split=True, seed=42):
        num_stocks, num_timesteps, num_features = data.shape
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        
        np.random.seed(seed)
        random.seed(seed)
        
        for i in range(num_stocks):
            stock_data = data[i]
            stock_labels = labels[i]
            
            if time_series_split:
                split_idx = int(num_timesteps * (1 - test_size))
                train_d = stock_data[:split_idx]
                test_d = stock_data[split_idx:]
                train_l = stock_labels[:split_idx]
                test_l = stock_labels[split_idx:]
            else:
                train_d, test_d, train_l, test_l = train_test_split(
                    stock_data, stock_labels, test_size=test_size, random_state=seed, shuffle=True
                )
            
            train_data.append(train_d)
            test_data.append(test_d)
            train_labels.append(train_l)
            test_labels.append(test_l)
        
        train_data = np.stack(train_data, axis=0)
        test_data = np.stack(test_data, axis=0)
        train_labels = np.stack(train_labels, axis=0)
        test_labels = np.stack(test_labels, axis=0)
        
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)
        
        return train_data, test_data, train_labels, test_labels