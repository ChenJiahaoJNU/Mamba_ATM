import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import shutil
from datetime import datetime
import traceback

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 移除中文字体依赖
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = './out'
ANALYSIS_BASE_DIR = './analysis'
ANALYSIS_OUTPUT_DIR = os.path.join(ANALYSIS_BASE_DIR, f'analysis_{TIMESTAMP}')
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

RAW_DATA_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, 'raw_data')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

CATEGORY_DIRS = {
    'MLP': os.path.join(ANALYSIS_OUTPUT_DIR, 'MLP'),
    'LSTM': os.path.join(ANALYSIS_OUTPUT_DIR, 'LSTM'),
    'Attention': os.path.join(ANALYSIS_OUTPUT_DIR, 'Attention'),
    'MambaAMT': os.path.join(ANALYSIS_OUTPUT_DIR, 'MambaAMT'),
    'combined': os.path.join(ANALYSIS_OUTPUT_DIR, 'combined'),
    'trade_points': os.path.join(ANALYSIS_OUTPUT_DIR, 'trade_points')
}

for dir_path in CATEGORY_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

MODEL_CATEGORIES = {
    'MLP': ['MLP_Base', 'MLP-Mamba1', 'MLP-mamba2', 'MLP-S4'],
    'LSTM': ['LSTM_Base', 'LSTM-Mamba1', 'LSTM-mamba2', 'LSTM-S4'],
    'Attention': ['Attention_Base', 'Attention-Mamba1', 'Attention-mamba2',
                  'Attention-S4', 'Attention-MambaAMT', 'Attention-MambaAMT-Enhanced'],
    'MambaAMT': ['Attention-MambaAMT', 'Attention-MambaAMT-Enhanced']
}

COLOR_MAP = {
    'MLP_Base': '#9467bd',
    'MLP-Mamba1': '#8c564b',
    'MLP-mamba2': '#e377c2',
    'MLP-S4': '#7f7f7f',
    'LSTM_Base': '#9467bd',
    'LSTM-Mamba1': '#8c564b',
    'LSTM-mamba2': '#e377c2',
    'LSTM-S4': '#7f7f7f',
    'Attention_Base': '#9467bd',
    'Attention-Mamba1': '#8c564b',
    'Attention-mamba2': '#e377c2',
    'Attention-S4': '#7f7f7f',
    'Attention-MambaAMT': '#98df8a',
    'Attention-MambaAMT-Enhanced': '#ff9896'
}

# 移除所有中文，只保留英文标签
TRADE_STYLES = {
    'Buy': {'marker': 'v', 'color': '#e74c3c', 'size': 90, 'label': 'Buy', 'zorder': 6},
    'Sell': {'marker': '^', 'color': '#2ecc71', 'size': 90, 'label': 'Sell', 'zorder': 5},
    'StopLoss': {'marker': 'X', 'color': '#000000', 'size': 110, 'label': 'StopLoss', 'zorder': 8},
    'TakeProfit': {'marker': 'p', 'color': '#f39c12', 'size': 100, 'label': 'TakeProfit', 'zorder': 7},
    'Sell (Force)': {'marker': '*', 'color': '#9b59b6', 'size': 120, 'label': 'Sell (Force)', 'zorder': 9}
}

# ===================== 工具函数 =====================
def copy_result_files(source_dir, target_dir):
    print(f"\nCopying result files to {target_dir} ...")
    result_files = [f for f in os.listdir(source_dir)
                   if 'result' in f.lower() and f.endswith('.xlsx') and not f.startswith('~')]

    if not result_files:
        print("⚠️  No Excel files containing 'result' found")
        return

    copied_count = 0
    for file_name in result_files:
        try:
            shutil.copy2(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
            copied_count += 1
            print(f"✅ Copied: {file_name}")
        except Exception as e:
            print(f"❌ Failed to copy {file_name}: {e}")

    print(f"\n📄 File copy completed! Total copied: {copied_count} result files")

def filter_models(all_results, exclude_models=None, exclude_categories=None):
    exclude_models = exclude_models or []
    exclude_categories = exclude_categories or []

    models_to_exclude = set(exclude_models)
    for category in exclude_categories:
        if category in MODEL_CATEGORIES:
            models_to_exclude.update(MODEL_CATEGORIES[category])
            print(f"Excluding all models in {category} category: {MODEL_CATEGORIES[category]}")

    filtered_results = {}
    excluded_count = 0
    for model_name, df in all_results.items():
        if model_name not in models_to_exclude:
            filtered_results[model_name] = df
        else:
            excluded_count += 1
            print(f"Excluded model: {model_name}")

    print(f"\nFiltering completed! Original:{len(all_results)} | Excluded:{excluded_count} | Remaining:{len(filtered_results)}")
    return filtered_results

def get_user_filter_choices():
    print("\n" + "="*60)
    print("Model Filter Options")
    print("="*60)

    print("\nAvailable model categories:")
    category_list = list(MODEL_CATEGORIES.keys())
    for i, (category, models) in enumerate(MODEL_CATEGORIES.items(), 1):
        print(f"{i}. {category}: {models}")

    exclude_categories = []
    category_choice = input("\nEnter model categories to exclude (ID/name, comma separated, press enter for none): ").strip()
    if category_choice:
        for item in category_choice.split(','):
            item = item.strip()
            try:
                idx = int(item) - 1
                if 0 <= idx < len(category_list):
                    exclude_categories.append(category_list[idx])
            except:
                if item in MODEL_CATEGORIES:
                    exclude_categories.append(item)

    exclude_models = []
    model_choice = input("\nEnter specific model names to exclude (comma separated, press enter for none): ").strip()
    if model_choice:
        exclude_models = [m.strip() for m in model_choice.split(',')]

    print("\nFilter confirmation:")
    if exclude_categories:
        print(f"- Excluded categories: {exclude_categories}")
    if exclude_models:
        print(f"- Excluded specific models: {exclude_models}")
    if not exclude_categories and not exclude_models:
        print("- No models excluded, using all data")

    return exclude_models, exclude_categories

def check_model_files(target_stock="000008"):
    """检查每个模型对应的文件是否存在并正确命名"""
    print("\n=== File Check ===")
    all_models = [
        'MLP_Base', 'MLP-Mamba1', 'MLP-mamba2', 'MLP-S4',
        'LSTM_Base', 'LSTM-Mamba1', 'LSTM-mamba2', 'LSTM-S4',
        'Attention_Base', 'Attention-Mamba1', 'Attention-mamba2', 'Attention-S4',
        'Attention-MambaAMT', 'Attention-MambaAMT-Enhanced'
    ]
    
    for model_name in all_models:
        pattern = os.path.join(OUTPUT_DIR, f"{model_name}_repeat_*{target_stock}.xlsx")
        files = glob.glob(pattern)
        files = [f for f in files if os.path.basename(f).startswith(f"{model_name}_repeat_")]
        
        if files:
            print(f"✅ {model_name}: Found {len(files)} files")
            for f in files:
                print(f"   - {os.path.basename(f)}")
        else:
            print(f"❌ {model_name}: No files found (Pattern: {pattern})")

# ===================== 数据加载 =====================
def load_all_results():
    all_results = {}
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.endswith('_repeat_results.xlsx') and not file_name.startswith('~'):
            model_name = file_name.replace('_repeat_results.xlsx', '')
            file_path = os.path.join(OUTPUT_DIR, file_name)
            try:
                df = pd.read_excel(file_path)
                all_results[model_name] = df
                print(f"Loaded successfully: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    return all_results

def aggregate_results(all_results):
    summary_data = []
    for model_name, df in all_results.items():
        summary_data.append({
            'model_name': model_name,
            'avg_return': df['avg_return'].mean(),
            'std_return': df['std_return'].mean(),
            'avg_accuracy': df['avg_accuracy'].mean(),
            'avg_rmse': df['avg_rmse'].mean(),
            'max_return': df['avg_return'].max(),
            'min_return': df['avg_return'].min(),
            'return_std': df['avg_return'].std(),
            'num_stocks': len(df)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df['category'] = summary_df['model_name'].apply(
        lambda x: next((cat for cat, models in MODEL_CATEGORIES.items() if x in models), 'Other')
    )
    return summary_df

# ===================== 交易点可视化（支持分批卖出 · 修复版） =====================
def generate_trade_points_plots(all_results=None, target_stock_code=None):
    print("\n6. Generating trade points visualization charts (support partial selling)...")
    os.makedirs(CATEGORY_DIRS['trade_points'], exist_ok=True)

    target_stock = "000008" if target_stock_code is None else str(target_stock_code)
    print(f"Analyzing stock: {target_stock}")

    all_models = [
        'MLP_Base', 'MLP-Mamba1', 'MLP-mamba2', 'MLP-S4',
        'LSTM_Base', 'LSTM-Mamba1', 'LSTM-mamba2', 'LSTM-S4',
        'Attention_Base', 'Attention-Mamba1', 'Attention-mamba2', 'Attention-S4',
        'Attention-MambaAMT', 'Attention-MambaAMT-Enhanced'
    ]

    for model_name in all_models:
        try:
            # 修复1：更精确的文件匹配规则
            # 匹配格式：模型名_repeat_任意字符_股票代码.xlsx
            pattern = os.path.join(OUTPUT_DIR, f"{model_name}_repeat_*{target_stock}.xlsx")
            raw_files = glob.glob(pattern)
            
            # 修复2：严格的文件过滤，确保文件名完全匹配模型名
            raw_files = [f for f in raw_files if os.path.basename(f).startswith(f"{model_name}_repeat_")]
            
            if not raw_files:
                print(f"❌ {model_name} - No raw data found for {target_stock} (Pattern: {pattern})")
                continue
            
            # 修复3：只取第一个匹配的文件，并验证
            raw_file_path = raw_files[0]
            file_name = os.path.basename(raw_file_path)
            
            # 额外校验：确保文件名包含正确的模型名和股票代码
            if model_name not in file_name or target_stock not in file_name:
                print(f"⚠️ {model_name} - File {file_name} does not match model/stock, skipping")
                continue

            df = pd.read_excel(raw_file_path)
            
            # 修复4：数据校验，确保DataFrame非空且包含必要列
            if df.empty:
                print(f"⚠️ {model_name} - Empty data frame, skipping")
                continue
                
            # 检查必要的列
            required_cols = ['日期', '收盘价', 'Strategy']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ {model_name} - Missing columns: {missing_cols}, skipping")
                continue

            df['Date'] = pd.to_datetime(df['日期'])  # 保留原始列名但图表只用英文
            df = df.sort_values('Date').reset_index(drop=True)

            # 取出所有交易
            trade_mask = df['Strategy'].isin(TRADE_STYLES.keys())
            trade_df = df[trade_mask].copy()

            if trade_df.empty:
                print(f"⚠️ {model_name} - No trading records, skipping plot generation")
                continue

            # 找到第一次买入时间，只保留第一次买入及之后的所有交易
            buy_rows = trade_df[trade_df['Strategy'] == 'Buy']
            if buy_rows.empty:
                print(f"⚠️ {model_name} - No buy records, skipping")
                continue

            first_buy_time = buy_rows['Date'].min()
            valid_df = trade_df[trade_df['Date'] >= first_buy_time].copy()

            # 保留所有交易：允许分批卖出、多次止盈、多次止损
            final_trades = valid_df.copy()

            # 绘图
            plt.figure(figsize=(16, 8))
            plt.plot(df['Date'], df['收盘价'], color='#3498db', linewidth=1.5, label='Close Price')

            for tag in TRADE_STYLES:
                sub = final_trades[final_trades['Strategy'] == tag]
                if sub.empty:
                    continue
                style = TRADE_STYLES[tag]
                plt.scatter(sub['Date'], sub['收盘价'],
                            marker=style['marker'],
                            color=style['color'],
                            s=style['size'],
                            label=style['label'],
                            zorder=style['zorder'])

            plt.xticks(rotation=45)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'{model_name} - {target_stock} Trade Points (Support Partial Selling)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(CATEGORY_DIRS['trade_points'], f"{model_name}_{target_stock}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 修复5：更精确的交易统计，避免统计错误
            cnt = {}
            for t in TRADE_STYLES:
                cnt[t] = len(final_trades[final_trades['Strategy'] == t])
            
            # 打印详细信息，便于调试
            print(f"✅ {model_name} | File: {file_name} | B:{cnt['Buy']} S:{cnt['Sell']} SL:{cnt['StopLoss']} TP:{cnt['TakeProfit']} ForceSell:{cnt['Sell (Force)']}")

        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)[:100]}")
            # 打印完整的异常堆栈，便于调试
            traceback.print_exc()
            continue

# ===================== 绘图 =====================
def plot_category_specific_charts(summary_df):
    valid_categories = [cat for cat in MODEL_CATEGORIES.keys() if cat in summary_df['category'].unique()]

    for category in valid_categories:
        cat_df = summary_df[summary_df['category'] == category].copy()
        if len(cat_df) == 0:
            continue

        plt.rcParams.update({
            'axes.labelsize': 15,
            'axes.titlesize': 16,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

        plt.figure(figsize=(12, 8))
        models = cat_df['model_name'].values
        returns = cat_df['avg_return'].values
        stds = cat_df['std_return'].values

        bars = plt.bar(range(len(models)), returns, yerr=stds, alpha=0.8, capsize=5)
        for i, bar in enumerate(bars):
            model_name = models[i]
            if model_name in COLOR_MAP:
                bar.set_color(COLOR_MAP[model_name])

        for i, (return_val, std_val) in enumerate(zip(returns, stds)):
            plt.text(i, return_val + std_val + 0.005,
                    f'{return_val:.3f}', ha='center', va='bottom', fontsize=12)

        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel('Average Return', fontsize=18)
        plt.xlabel('Models', fontsize=18)
        plt.title(f'{category} Model Return Comparison', fontsize=20)
        y_max = max(returns) + max(stds) + 0.01
        plt.ylim(0, y_max * 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CATEGORY_DIRS[category], f'{category}_return_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        cat_df_sorted = cat_df.sort_values('avg_accuracy', ascending=False)
        bars = plt.bar(cat_df_sorted['model_name'], cat_df_sorted['avg_accuracy'], alpha=0.8)

        for i, bar in enumerate(bars):
            model_name = cat_df_sorted['model_name'].iloc[i]
            if model_name in COLOR_MAP:
                bar.set_color(COLOR_MAP[model_name])

        for i, acc in enumerate(cat_df_sorted['avg_accuracy']):
            plt.text(i, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Accuracy', fontsize=15)
        plt.title(f'{category} Model Accuracy Comparison', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(CATEGORY_DIRS[category], f'{category}_accuracy_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        cat_df_sorted = cat_df.sort_values('avg_rmse')
        bars = plt.bar(cat_df_sorted['model_name'], cat_df_sorted['avg_rmse'], alpha=0.8)

        for i, bar in enumerate(bars):
            model_name = cat_df_sorted['model_name'].iloc[i]
            if model_name in COLOR_MAP:
                bar.set_color(COLOR_MAP[model_name])

        for i, rmse in enumerate(cat_df_sorted['avg_rmse']):
            plt.text(i, rmse + 0.001, f'{rmse:.4f}', ha='center', va='bottom', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average RMSE', fontsize=15)
        plt.title(f'{category} Model RMSE Comparison', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CATEGORY_DIRS[category], f'{category}_rmse_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated charts for {category} category, saved to: {CATEGORY_DIRS[category]}")

def create_combined_plots(summary_df):
    plt.rcParams.update({
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    categories = summary_df['category'].unique()

    plt.figure(figsize=(16, 10))
    x_pos = 0
    offsets = []
    all_returns = []
    all_stds = []

    for category in categories:
        cat_data = summary_df[summary_df['category'] == category]
        cat_models = cat_data['model_name'].values
        cat_returns = cat_data['avg_return'].values
        cat_std = cat_data['std_return'].values

        all_returns.extend(cat_returns)
        all_stds.extend(cat_std)

        bars = plt.bar(range(x_pos, x_pos + len(cat_models)),
                      cat_returns, yerr=cat_std, alpha=0.8, capsize=5, label=category)

        for i, bar in enumerate(bars):
            model_name = cat_models[i]
            if model_name in COLOR_MAP:
                bar.set_color(COLOR_MAP[model_name])

        for i, (return_val, std_val) in enumerate(zip(cat_returns, cat_std)):
            plt.text(x_pos + i, return_val + std_val + 0.01,
                    f'{return_val:.3f}', ha='center', va='bottom', fontsize=14, rotation=90)

        offsets.extend([x_pos + i for i in range(len(cat_models))])
        x_pos += len(cat_models) + 1

    plt.xticks(offsets, summary_df['model_name'].values, rotation=45, ha='right', fontsize=18)
    plt.ylabel('Average Return', fontsize=20)
    plt.xlabel('Models', fontsize=20)
    plt.title('Model Average Return Comparison', fontsize=22)

    if all_returns:
        y_max = max(all_returns) + max(all_stds) + 0.01
        plt.ylim(0, y_max * 1.1)

    plt.yticks(fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CATEGORY_DIRS['combined'], 'return_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(16, 8))
    summary_df_sorted = summary_df.sort_values('avg_accuracy', ascending=False)
    bars = plt.bar(summary_df_sorted['model_name'], summary_df_sorted['avg_accuracy'], alpha=0.8)

    for i, bar in enumerate(bars):
        model_name = summary_df_sorted['model_name'].iloc[i]
        if model_name in COLOR_MAP:
            bar.set_color(COLOR_MAP[model_name])

    for i, acc in enumerate(summary_df_sorted['avg_accuracy']):
        plt.text(i, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Accuracy', fontsize=15)
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(CATEGORY_DIRS['combined'], 'accuracy_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(16, 8))
    summary_df_sorted = summary_df.sort_values('avg_rmse')
    bars = plt.bar(summary_df_sorted['model_name'], summary_df_sorted['avg_rmse'], alpha=0.8)

    for i, bar in enumerate(bars):
        model_name = summary_df_sorted['model_name'].iloc[i]
        if model_name in COLOR_MAP:
            bar.set_color(COLOR_MAP[model_name])

    for i, rmse in enumerate(summary_df_sorted['avg_rmse']):
        plt.text(i, rmse + 0.001, f'{rmse:.4f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average RMSE', fontsize=15)
    plt.title('Model RMSE Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CATEGORY_DIRS['combined'], 'rmse_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    for category in categories:
        cat_data = summary_df[summary_df['category'] == category]
        plt.scatter(cat_data['return_std'], cat_data['avg_return'],
                   s=cat_data['avg_accuracy']*200, alpha=0.7, label=category)

    for idx, row in summary_df.iterrows():
        plt.annotate(row['model_name'], (row['return_std'], row['avg_return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=12, alpha=0.8)

    plt.xlabel('Return Standard Deviation (Risk)', fontsize=15)
    plt.ylabel('Average Return (Reward)', fontsize=15)
    plt.title('Risk-Return Analysis (Bubble size = Accuracy)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CATEGORY_DIRS['combined'], 'risk_return_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated combined charts, saved to: {CATEGORY_DIRS['combined']}")

def create_comparison_plots(summary_df):
    plot_category_specific_charts(summary_df)
    create_combined_plots(summary_df)

# ===================== 报告 =====================
def generate_detailed_report(summary_df, all_results, exclude_info=None):
    report = []
    report.append("# Model Experiment Results Analysis Report")
    report.append("="*50)

    report.append(f"\n## Analysis Information")
    report.append(f"- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- Results Storage Directory: {ANALYSIS_OUTPUT_DIR}")
    report.append(f"- Raw Data Backup Directory: {RAW_DATA_DIR}")

    if exclude_info:
        report.append(f"\n## Filter Conditions")
        if exclude_info['exclude_categories']:
            report.append(f"- Excluded Model Categories: {exclude_info['exclude_categories']}")
        if exclude_info['exclude_models']:
            report.append(f"- Excluded Specific Models: {exclude_info['exclude_models']}")

    categories = summary_df['category'].unique()

    report.append("\n## 1. Overall Statistics")
    report.append(f"- Number of Models in Comparison: {len(summary_df)}")
    if len(summary_df) > 0:
        report.append(f"- Number of Tested Stocks: {summary_df.iloc[0]['num_stocks']}")
        report.append(f"- Average Return Range: {summary_df['avg_return'].min():.4f} ~ {summary_df['avg_return'].max():.4f}")
        report.append(f"- Average Accuracy Range: {summary_df['avg_accuracy'].min():.4f} ~ {summary_df['avg_accuracy'].max():.4f}")

    if len(summary_df) > 0:
        report.append("\n## 2. Best Models by Metrics")

        best_return = summary_df.loc[summary_df['avg_return'].idxmax()]
        report.append(f"\n### 2.1 Best Return")
        report.append(f"- Model: {best_return['model_name']}")
        report.append(f"- Average Return: {best_return['avg_return']:.4f}")
        report.append(f"- Return Standard Deviation: {best_return['std_return']:.4f}")
        report.append(f"- Accuracy: {best_return['avg_accuracy']:.4f}")

        best_accuracy = summary_df.loc[summary_df['avg_accuracy'].idxmax()]
        report.append(f"\n### 2.2 Best Accuracy")
        report.append(f"- Model: {best_accuracy['model_name']}")
        report.append(f"- Average Accuracy: {best_accuracy['avg_accuracy']:.4f}")
        report.append(f"- Average Return: {best_accuracy['avg_return']:.4f}")

        best_rmse = summary_df.loc[summary_df['avg_rmse'].idxmin()]
        report.append(f"\n### 2.3 Lowest RMSE")
        report.append(f"- Model: {best_rmse['model_name']}")
        report.append(f"- Average RMSE: {best_rmse['avg_rmse']:.4f}")
        report.append(f"- Average Return: {best_rmse['avg_return']:.4f}")

        report.append("\n## 3. Model Category Comparison")
        category_stats = summary_df.groupby('category').agg({
            'avg_return': ['mean', 'std', 'max'],
            'avg_accuracy': ['mean', 'std'],
            'avg_rmse': ['mean', 'std']
        }).round(4)

        for category in categories:
            if category in category_stats.index:
                stats = category_stats.loc[category]
                report.append(f"\n### 3.1 {category} Category Models")
                report.append(f"- Average Return: {stats['avg_return']['mean']:.4f} (±{stats['avg_return']['std']:.4f})")
                report.append(f"- Maximum Return: {stats['avg_return']['max']:.4f}")
                report.append(f"- Average Accuracy: {stats['avg_accuracy']['mean']:.4f} (±{stats['avg_accuracy']['std']:.4f})")
                report.append(f"- Average RMSE: {stats['avg_rmse']['mean']:.4f} (±{stats['avg_rmse']['std']:.4f})")

        report.append("\n## 4. Risk-Return Analysis")
        summary_df['sharpe_ratio'] = summary_df['avg_return'] / summary_df['return_std']
        best_sharpe = summary_df.loc[summary_df['sharpe_ratio'].idxmax()]

        report.append(f"\n### 4.1 Best Sharpe Ratio (Risk-Adjusted Return)")
        report.append(f"- Model: {best_sharpe['model_name']}")
        report.append(f"- Sharpe Ratio: {best_sharpe['sharpe_ratio']:.4f}")
        report.append(f"- Return: {best_sharpe['avg_return']:.4f}")
        report.append(f"- Risk (Standard Deviation): {best_sharpe['return_std']:.4f}")

        best_model_name = best_return['model_name']
        best_model_df = all_results[best_model_name]

        top_stock = best_model_df.loc[best_model_df['avg_return'].idxmax()]
        bottom_stock = best_model_df.loc[best_model_df['avg_return'].idxmin()]

        report.append(f"\n### 5.1 Performance of Best Model ({best_model_name}) on Different Stocks")
        report.append(f"- Best Performing Stock: {top_stock['stock_code']} (Return: {top_stock['avg_return']:.4f})")
        report.append(f"- Worst Performing Stock: {bottom_stock['stock_code']} (Return: {bottom_stock['avg_return']:.4f})")
        report.append(f"- Return Standard Deviation Across Stocks: {best_model_df['avg_return'].std():.4f}")

    report_text = "\n".join(report)
    with open(os.path.join(ANALYSIS_OUTPUT_DIR, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text

# ===================== 主函数 =====================
def main():
    print("Starting experiment results analysis...")
    print(f"📁 Analysis results will be saved to: {ANALYSIS_OUTPUT_DIR}")

    copy_result_files(OUTPUT_DIR, RAW_DATA_DIR)

    print("\n1. Loading all result files...")
    all_results = load_all_results()

    if not all_results:
        print("No result files found! Program exiting.")
        return

    exclude_models, exclude_categories = get_user_filter_choices()

    print("\n2. Filtering models...")
    filtered_results = filter_models(all_results, exclude_models, exclude_categories)

    if not filtered_results:
        print("No models remaining after filtering! Program exiting.")
        return

    print("\n3. Aggregating analysis data...")
    summary_df = aggregate_results(filtered_results)
    summary_df.to_excel(os.path.join(ANALYSIS_OUTPUT_DIR, 'model_performance_summary.xlsx'), index=False)

    print("\n4. Generating visualization charts...")
    create_comparison_plots(summary_df)

    print("\n5. Generating detailed analysis report...")
    exclude_info = {
        'exclude_models': exclude_models,
        'exclude_categories': exclude_categories
    }
    generate_detailed_report(summary_df, filtered_results, exclude_info)

    # 添加文件检查
    check_model_files("000008")
    
    generate_trade_points_plots(filtered_results)

    print("\n" + "="*50)
    print("Analysis completed! Key Findings:")
    if len(summary_df) > 0:
        print(f"- Best Return Model: {summary_df.loc[summary_df['avg_return'].idxmax()]['model_name']} ({summary_df['avg_return'].max():.4f})")
        print(f"- Best Accuracy Model: {summary_df.loc[summary_df['avg_accuracy'].idxmax()]['model_name']} ({summary_df['avg_accuracy'].max():.4f})")
        print(f"- Best RMSE Model: {summary_df.loc[summary_df['avg_rmse'].idxmin()]['model_name']} ({summary_df['avg_rmse'].min():.4f})")
    print("="*50)

    print(f"\nAll analysis results have been saved to: {ANALYSIS_OUTPUT_DIR}")
    print("- model_performance_summary.xlsx: Model performance summary table")
    print("- analysis_report.md: Detailed analysis report")
    print("- raw_data/: Raw result files backup")
    print("- MLP/LSTM/Attention/MambaAMT/: Category-specific charts")
    print("- combined/: Combined comparison charts")
    print("- trade_points/: Trade points visualization charts")

def quick_analysis(exclude_models=None, exclude_categories=None, target_stock_code=None):
    print("Starting quick experiment results analysis...")
    print(f"📁 Analysis results will be saved to: {ANALYSIS_OUTPUT_DIR}")

    copy_result_files(OUTPUT_DIR, RAW_DATA_DIR)

    all_results = load_all_results()

    if not all_results:
        print("No result files found! Program exiting.")
        return

    filtered_results = filter_models(all_results, exclude_models, exclude_categories)

    if not filtered_results:
        print("No models remaining after filtering! Program exiting.")
        return

    summary_df = aggregate_results(filtered_results)
    summary_df.to_excel(os.path.join(ANALYSIS_OUTPUT_DIR, 'model_performance_summary.xlsx'), index=False)

    create_comparison_plots(summary_df)

    exclude_info = {
        'exclude_models': exclude_models or [],
        'exclude_categories': exclude_categories or []
    }
    generate_detailed_report(summary_df, filtered_results, exclude_info)

    # 添加文件检查
    check_model_files(target_stock_code or "000008")
    
    generate_trade_points_plots(filtered_results, target_stock_code)

    print("\nQuick analysis completed! Results saved to:", ANALYSIS_OUTPUT_DIR)

if __name__ == "__main__":
    main()