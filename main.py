import torch
import torch.nn as nn
import numpy as np
import os
import random
import pandas as pd

# 导入自定义模块
from data_utils import DataLoader, DataStatistics
from models import MLP, CombinedMLP, SelfAttention, SelfAttentionMamba, LSTMModel, CombinedLSTM, AdvancedMambaAMT
from trainer import run_experiment
from stats_test import StatisticalTest

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # ===================== 增强版配置 =====================
    SEED = 42
    REPEAT_TIMES = 5
    TEST_SIZE = 0.25
    TIME_SERIES_SPLIT = True
    
    DATA_DIR = './pre_data'
    OUTPUT_DIR = './out'
    LOG_DIR = './logs'
    
    # 增强版超参数（新增MambaAMT专属配置）
    HYPER_PARAMS = {
        'seed': SEED,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 32,
        'hidden_dim': 256,
        'lstm_layers': 2,
        'dropout_rate': 0.2,
        'weight_decay': 1e-5,
        'step_size': 30,
        'gamma': 0.1,
        'mamba_d_model': 256,
        'mamba_d_state': 16,
        'mamba_d_conv': 4,
        'mamba_expand': 2,
        # MambaAMT专属超参数
        'mambaamt_lr': 0.0005,          # 更小的学习率
        'mambaamt_weight_decay': 1e-6,  # 更小的权重衰减
    }
    
    # 增强版交易配置
    TRADING_CONFIG = {
        'commission_rate': 0.0003,
        'slippage_rate': 0.0002,
        'stamp_tax': 0.001,
        'min_commission': 5.0,
        'buy_threshold': 0.2,
        'sell_threshold': 0.08,
        'base_quantity': 100,
        'add_quantity': 100,
        # MambaAMT专属交易参数（更保守）
        'mambaamt_buy_threshold': 0.15,   # 更低的加仓阈值
        'mambaamt_sell_threshold': 0.05,  # 更低的卖出阈值
        'mambaamt_base_quantity': 200,    # 更大的基础仓位
        'mambaamt_add_quantity': 50,      # 更小的加仓量
    }
    
    ALPHA = 0.05
    
    # 创建目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    set_seed(SEED)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats_logger = DataStatistics(LOG_DIR)
    stats_logger.log("=== 增强版实验开始 ===")
    stats_logger.log(f"设备: {DEVICE}")
    stats_logger.log(f"随机种子: {SEED}")
    stats_logger.log(f"重复实验次数: {REPEAT_TIMES}")
    stats_logger.log(f"时间序列划分: {TIME_SERIES_SPLIT}")
    stats_logger.log(f"超参数配置: {HYPER_PARAMS}")
    stats_logger.log(f"交易配置: {TRADING_CONFIG}")
    
    # 加载数据
    stock_data_binary, stock_labels_binary, CSV_FILES, STOCK_CODES = DataLoader.load_and_process_data(
        DATA_DIR, return_type='binary', seed=SEED
    )
    stock_data_raw, stock_labels_raw, _, _ = DataLoader.load_and_process_data(
        DATA_DIR, return_type='raw', seed=SEED
    )
    
    INPUT_DIM = stock_data_binary.shape[2]
    
    # 分割数据
    train_x_binary, test_x_binary, train_y_binary, test_y_binary = DataLoader.split_data(
        stock_data_binary, stock_labels_binary, 
        test_size=TEST_SIZE, 
        time_series_split=TIME_SERIES_SPLIT,
        seed=SEED
    )
    train_x_raw, test_x_raw, train_y_raw, test_y_raw = DataLoader.split_data(
        stock_data_raw, stock_labels_raw, 
        test_size=TEST_SIZE, 
        time_series_split=TIME_SERIES_SPLIT,
        seed=SEED
    )
    
    stat_test = StatisticalTest(alpha=ALPHA)
    
    MAMBA_CONFIG = {
        'd_model': HYPER_PARAMS['mamba_d_model'],
        'd_state': HYPER_PARAMS['mamba_d_state'],
        'd_conv': HYPER_PARAMS['mamba_d_conv'],
        'expand': HYPER_PARAMS['mamba_expand']
    }
    
    # ===================== 原有实验保持不变 =====================
    stats_logger.log("\n=== 开始MLP系列实验 ===")
    mlp_criterion = nn.BCELoss()
    mlp_base_returns, mlp_base_metrics, mlp_base_all_returns = run_experiment(
        "MLP_Base", MLP, INPUT_DIM, train_x_binary, train_y_binary, test_x_binary, test_y_binary, mlp_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    
    mlp_variants = [
        ("MLP-Mamba1", "mamba"),
        ("MLP-S4", "s4"),
        ("MLP-mamba2", "mamba2_original")
    ]
    
    mlp_all_returns = {'MLP_Base': mlp_base_returns}
    mlp_all_raw_returns = {'MLP_Base': mlp_base_all_returns}
    
    for name, model_type in mlp_variants:
        returns, metrics, raw_returns = run_experiment(
            name, CombinedMLP, INPUT_DIM, train_x_binary, train_y_binary, test_x_binary, test_y_binary, mlp_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        mlp_all_returns[name] = returns
        mlp_all_raw_returns[name] = raw_returns
    
    mlp_tests = []
    for name in mlp_variants:
        base_flat = np.array(mlp_base_all_returns).flatten()
        variant_flat = np.array(mlp_all_raw_returns[name[0]]).flatten()
        
        ttest_result = stat_test.paired_ttest(base_flat, variant_flat, "MLP_Base", name[0])
        mlp_tests.append(ttest_result)
        
        wilcoxon_result = stat_test.wilcoxon_test(base_flat, variant_flat, "MLP_Base", name[0])
        mlp_tests.append(wilcoxon_result)
    
    mlp_results = pd.DataFrame(mlp_all_returns, index=STOCK_CODES)
    mlp_results.index.name = 'Stock Code'
    mlp_results.to_excel(os.path.join(OUTPUT_DIR, 'MLP_combined_results.xlsx'))
    
    # ===================== Attention系列实验（新增增强版MambaAMT） =====================
    stats_logger.log("\n=== 开始Attention系列实验 ===")
    attn_criterion = nn.MSELoss()
    attn_base_returns, attn_base_metrics, attn_base_all_returns = run_experiment(
        "Attention_Base", SelfAttention, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, attn_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    
    attn_variants = [
        ("Attention-Mamba1", "mamba"),
        ("Attention-S4", "s4"),
        ("Attention-mamba2", "mamba2_original"),
        ("Attention-MambaAMT", "MambaAMT")
    ]
    
    attn_all_returns = {'Attention_Base': attn_base_returns}
    attn_all_raw_returns = {'Attention_Base': attn_base_all_returns}
    
    for name, model_type in attn_variants:
        returns, metrics, raw_returns = run_experiment(
            name, SelfAttentionMamba, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, attn_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        attn_all_returns[name] = returns
        attn_all_raw_returns[name] = raw_returns
    
    # 运行增强版MambaAMT
    mambaamt_returns, mambaamt_metrics, mambaamt_all_returns = run_experiment(
        "Attention-MambaAMT-Enhanced", AdvancedMambaAMT, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, 
        attn_criterion, device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    attn_all_returns["Attention-MambaAMT-Enhanced"] = mambaamt_returns
    attn_all_raw_returns["Attention-MambaAMT-Enhanced"] = mambaamt_all_returns
    
    # Attention统计检验（包含增强版MambaAMT）
    attn_tests = []
    all_attn_models = attn_variants + [("Attention-MambaAMT-Enhanced", None)]
    for name, _ in all_attn_models:
        base_flat = np.array(attn_base_all_returns).flatten()
        variant_flat = np.array(attn_all_raw_returns[name]).flatten()
        
        ttest_result = stat_test.paired_ttest(base_flat, variant_flat, "Attention_Base", name)
        attn_tests.append(ttest_result)
        
        wilcoxon_result = stat_test.wilcoxon_test(base_flat, variant_flat, "Attention_Base", name)
        attn_tests.append(wilcoxon_result)
    
    attn_results = pd.DataFrame(attn_all_returns, index=STOCK_CODES)
    attn_results.index.name = 'Stock Code'
    attn_results.to_excel(os.path.join(OUTPUT_DIR, 'Attention_combined_results.xlsx'))
    
    # ===================== LSTM系列实验 =====================
    stats_logger.log("\n=== 开始LSTM系列实验 ===")
    lstm_criterion = nn.MSELoss()
    lstm_base_returns, lstm_base_metrics, lstm_base_all_returns = run_experiment(
        "LSTM_Base", LSTMModel, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, lstm_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    
    lstm_variants = [
        ("LSTM-Mamba1", "mamba"),
        ("LSTM-S4", "s4"),
        ("LSTM-mamba2", "mamba2_original")
    ]
    
    lstm_all_returns = {'LSTM_Base': lstm_base_returns}
    lstm_all_raw_returns = {'LSTM_Base': lstm_base_all_returns}
    
    for name, model_type in lstm_variants:
        returns, metrics, raw_returns = run_experiment(
            name, CombinedLSTM, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, lstm_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        lstm_all_returns[name] = returns
        lstm_all_raw_returns[name] = raw_returns
    
    lstm_tests = []
    for name in lstm_variants:
        base_flat = np.array(lstm_base_all_returns).flatten()
        variant_flat = np.array(lstm_all_raw_returns[name[0]]).flatten()
        
        ttest_result = stat_test.paired_ttest(base_flat, variant_flat, "LSTM_Base", name[0])
        lstm_tests.append(ttest_result)
        
        wilcoxon_result = stat_test.wilcoxon_test(base_flat, variant_flat, "LSTM_Base", name[0])
        lstm_tests.append(wilcoxon_result)
    
    lstm_results = pd.DataFrame(lstm_all_returns, index=STOCK_CODES)
    lstm_results.index.name = 'Stock Code'
    lstm_results.to_excel(os.path.join(OUTPUT_DIR, 'LSTM_combined_results.xlsx'))
    
    # ===================== 汇总所有统计检验结果 =====================
    all_tests = mlp_tests + attn_tests + lstm_tests
    stat_test.summarize_significance(all_tests, OUTPUT_DIR)
    
    # ===================== 最终结果汇总 =====================
    stats_logger.log("\n=== 实验完成，最终结果汇总 ===")
    
    print("\n=== 各模型平均收益率 ===")
    print("MLP系列:")
    for model, returns in mlp_all_returns.items():
        print(f"  {model}: {np.mean(returns):.4f} (±{np.std(returns):.4f})")
    
    print("\nAttention系列:")
    for model, returns in attn_all_returns.items():
        print(f"  {model}: {np.mean(returns):.4f} (±{np.std(returns):.4f})")
    
    print("\nLSTM系列:")
    for model, returns in lstm_all_returns.items():
        print(f"  {model}: {np.mean(returns):.4f} (±{np.std(returns):.4f})")
    
    # 保存综合结果
    all_models = list(mlp_all_returns.keys()) + list(attn_all_returns.keys()) + list(lstm_all_returns.keys())
    all_results_data = {'Stock Code': STOCK_CODES}
    for model in all_models:
        if model in mlp_all_returns:
            all_results_data[model] = mlp_all_returns[model]
        elif model in attn_all_returns:
            all_results_data[model] = attn_all_returns[model]
        elif model in lstm_all_returns:
            all_results_data[model] = lstm_all_returns[model]
    
    all_results = pd.DataFrame(all_results_data)
    all_results.to_excel(os.path.join(OUTPUT_DIR, 'all_models_combined_results.xlsx'), index=False)
    
    stats_logger.log("=== 所有实验完成 ===")
    stats_logger.logger.close()