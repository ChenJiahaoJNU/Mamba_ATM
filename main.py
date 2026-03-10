import torch
import torch.nn as nn
import numpy as np
import os
import random
import pandas as pd
from itertools import combinations
import scipy.stats as stats  # 确保导入stats库

# 导入自定义模块
from data_utils import DataLoader, DataStatistics
from models import MLP, CombinedMLP, SelfAttention, SelfAttentionMamba, LSTMModel, CombinedLSTM, AdvancedMambaAMT
from trainer import run_experiment
from stats_test import StatisticalTest

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # ===================== 配置项 =====================
    SEED = 42
    REPEAT_TIMES = 5
    TEST_SIZE = 0.25
    TIME_SERIES_SPLIT = True
    
    DATA_DIR = './pre_data'
    OUTPUT_DIR = './out'
    LOG_DIR = './logs'
    
    # 超参数配置
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
        'mambaamt_lr': 0.0005,          
        'mambaamt_weight_decay': 1e-6,  
    }
    
    # 交易配置
    TRADING_CONFIG = {
        'commission_rate': 0.0003,
        'slippage_rate': 0.0002,
        'stamp_tax': 0.001,
        'min_commission': 5.0,
        'buy_threshold': 0.2,
        'sell_threshold': 0.08,
        'base_quantity': 100,
        'add_quantity': 100,
        'mambaamt_buy_threshold': 0.15,
        'mambaamt_sell_threshold': 0.05,
        'mambaamt_base_quantity': 200,
        'mambaamt_add_quantity': 200,
    }
    
    ALPHA = 0.05
    
    # 创建目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ===================== 工具函数 =====================
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
    
    def wilcoxon_test(self, group1, group2, model1_name, model2_name):
        """补充Wilcoxon检验方法（适配你的StatisticalTest类）"""
        w_stat, p_value = stats.wilcoxon(group1, group2)
        
        result = {
            'test_type': 'Wilcoxon signed-rank test',
            'model1': model1_name,
            'model2': model2_name,
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }
        
        print(f"\n{model1_name} vs {model2_name} Wilcoxon检验结果:")
        print(f"W统计量: {w_stat:.4f}, p值: {p_value:.4f}")
        if p_value < self.alpha:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能存在显著差异")
        else:
            print(f"结论: 在{self.alpha}显著性水平下，两个模型性能无显著差异")
        
        return result
    
    # 为StatisticalTest类动态添加Wilcoxon方法（如果原类没有）
    if not hasattr(StatisticalTest, 'wilcoxon_test'):
        StatisticalTest.wilcoxon_test = wilcoxon_test
    
    # ===================== 初始化 =====================
    set_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats_logger = DataStatistics(LOG_DIR)
    stats_logger.log("=== 实验开始 ===")
    stats_logger.log(f"设备: {DEVICE}")
    stats_logger.log(f"随机种子: {SEED}")
    
    stat_test = StatisticalTest(alpha=ALPHA)
    
    # ===================== 数据加载与处理 =====================
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
    
    # ===================== 模型训练与结果收集 =====================
    # 存储所有模型的原始收益率数据（用于检验）
    all_models_raw_returns = {}
    # 存储所有模型的最终收益率（用于汇总）
    all_models_returns = {}
    
    MAMBA_CONFIG = {
        'd_model': HYPER_PARAMS['mamba_d_model'],
        'd_state': HYPER_PARAMS['mamba_d_state'],
        'd_conv': HYPER_PARAMS['mamba_d_conv'],
        'expand': HYPER_PARAMS['mamba_expand']
    }
    
    # 1. MLP系列
    stats_logger.log("\n=== 开始MLP系列实验 ===")
    mlp_criterion = nn.BCELoss()
    # MLP Base
    mlp_base_returns, mlp_base_metrics, mlp_base_all_returns = run_experiment(
        "MLP_Base", MLP, INPUT_DIM, train_x_binary, train_y_binary, test_x_binary, test_y_binary, mlp_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    all_models_raw_returns["MLP_Base"] = mlp_base_all_returns
    all_models_returns["MLP_Base"] = mlp_base_returns
    
    # MLP变体
    mlp_variants = [
        ("MLP-Mamba1", "mamba"),
        ("MLP-S4", "s4"),
        ("MLP-mamba2", "mamba2_original")
    ]
    for name, model_type in mlp_variants:
        returns, metrics, raw_returns = run_experiment(
            name, CombinedMLP, INPUT_DIM, train_x_binary, train_y_binary, test_x_binary, test_y_binary, mlp_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        all_models_raw_returns[name] = raw_returns
        all_models_returns[name] = returns
    
    # 2. Attention系列
    stats_logger.log("\n=== 开始Attention系列实验 ===")
    attn_criterion = nn.MSELoss()
    # Attention Base
    attn_base_returns, attn_base_metrics, attn_base_all_returns = run_experiment(
        "Attention_Base", SelfAttention, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, attn_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    all_models_raw_returns["Attention_Base"] = attn_base_all_returns
    all_models_returns["Attention_Base"] = attn_base_returns
    
    # Attention变体
    attn_variants = [
        ("Attention-Mamba1", "mamba"),
        ("Attention-S4", "s4"),
        ("Attention-mamba2", "mamba2_original"),
        ("Attention-MambaAMT", "MambaAMT")
    ]
    for name, model_type in attn_variants:
        returns, metrics, raw_returns = run_experiment(
            name, SelfAttentionMamba, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, attn_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        all_models_raw_returns[name] = raw_returns
        all_models_returns[name] = returns
    
    # 3. LSTM系列
    stats_logger.log("\n=== 开始LSTM系列实验 ===")
    lstm_criterion = nn.MSELoss()
    # LSTM Base
    lstm_base_returns, lstm_base_metrics, lstm_base_all_returns = run_experiment(
        "LSTM_Base", LSTMModel, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, lstm_criterion,
        device=DEVICE, hyper_params=HYPER_PARAMS, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
        data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
    )
    all_models_raw_returns["LSTM_Base"] = lstm_base_all_returns
    all_models_returns["LSTM_Base"] = lstm_base_returns
    
    # LSTM变体
    lstm_variants = [
        ("LSTM-Mamba1", "mamba"),
        ("LSTM-S4", "s4"),
        ("LSTM-mamba2", "mamba2_original")
    ]
    for name, model_type in lstm_variants:
        returns, metrics, raw_returns = run_experiment(
            name, CombinedLSTM, INPUT_DIM, train_x_raw, train_y_raw, test_x_raw, test_y_raw, lstm_criterion,
            is_combined=True, model_type=model_type, device=DEVICE, hyper_params=HYPER_PARAMS,
            mamba_config=MAMBA_CONFIG, repeat_times=REPEAT_TIMES, test_size=TEST_SIZE,
            data_dir=DATA_DIR, output_dir=OUTPUT_DIR, logger=stats_logger, trading_config=TRADING_CONFIG
        )
        all_models_raw_returns[name] = raw_returns
        all_models_returns[name] = returns
    
    # ===================== 全量两两显著性检验 =====================
    stats_logger.log("\n=== 开始所有模型全量两两显著性检验 ===")
    
    # 生成所有模型的两两组合
    model_names = list(all_models_raw_returns.keys())
    model_pairs = list(combinations(model_names, 2))
    stats_logger.log(f"参与检验的模型总数: {len(model_names)}")
    stats_logger.log(f"需执行的检验组合数: {len(model_pairs)}")
    
    # 存储检验结果
    test_results = []
    
    # 执行每一对模型的检验
    for model_a, model_b in model_pairs:
        stats_logger.log(f"\n--- 检验 {model_a} vs {model_b} ---")
        
        # 展平收益率数据
        returns_a = np.array(all_models_raw_returns[model_a]).flatten()
        returns_b = np.array(all_models_raw_returns[model_b]).flatten()
        
        # 配对t检验
        ttest_res = stat_test.paired_ttest(returns_a, returns_b, model_a, model_b)
        # Wilcoxon检验
        wilcoxon_res = stat_test.wilcoxon_test(returns_a, returns_b, model_a, model_b)
        
        # 整理结果（适配你的StatisticalTest返回格式）
        test_results.append({
            '模型A': model_a,
            '模型B': model_b,
            't检验统计量': ttest_res['t_statistic'],
            't检验p值': ttest_res['p_value'],
            't检验是否显著': '是' if ttest_res['significant'] else '否',
            'Wilcoxon统计量': wilcoxon_res['w_statistic'],
            'Wilcoxon p值': wilcoxon_res['p_value'],
            'Wilcoxon是否显著': '是' if wilcoxon_res['significant'] else '否',
            '显著性水平(α)': ALPHA
        })
    
    # ===================== 保存检验结果到Excel =====================
    if test_results:
        # 转换为DataFrame
        df_results = pd.DataFrame(test_results)
        # 保存为Excel
        excel_path = os.path.join(OUTPUT_DIR, 'all_models_pairwise_tests_results.xlsx')
        df_results.to_excel(excel_path, index=False, sheet_name='全量两两检验')
        stats_logger.log(f"\n所有模型两两检验结果已保存至: {excel_path}")
    else:
        stats_logger.log("\n无检验结果可保存")
    
    # ===================== 保存收益率汇总 =====================
    # 整理收益率数据
    all_results_data = {'Stock Code': STOCK_CODES}
    for model in model_names:
        all_results_data[model] = all_models_returns[model]
    
    # 保存为Excel
    returns_excel_path = os.path.join(OUTPUT_DIR, 'all_models_returns_summary.xlsx')
    pd.DataFrame(all_results_data).to_excel(returns_excel_path, index=False)
    stats_logger.log(f"所有模型收益率汇总已保存至: {returns_excel_path}")
    
    # ===================== 打印最终汇总 =====================
    stats_logger.log("\n=== 实验完成，各模型平均收益率汇总 ===")
    print("\n=== 各模型平均收益率 ===")
    for model, returns in all_models_returns.items():
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        print(f"  {model}: {mean_ret:.4f} (±{std_ret:.4f})")
        stats_logger.log(f"{model}: 均值={mean_ret:.4f}, 标准差={std_ret:.4f}")
    
    stats_logger.log("=== 所有实验完成 ===")
    stats_logger.logger.close()