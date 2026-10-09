
# ===============================
# 主函数（可直接运行）
# ===============================
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# 保留原始模型导入语句
from models import AdvancedMambaAMT, MLP, CombinedMLP, SelfAttention, SelfAttentionMamba, LSTMModel, CombinedLSTM

# 导入交易模拟器
from trading_simulator import get_trading_simulator

def main():
    """
    Main function for trading strategy comparison
    Generates two different prediction sets and compares their trading results
    """
    # 1. Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("TradingSimulator")

    # 2. Generate consistent stock price data (fixed random seed)
    np.random.seed(42)
    n_days = 200
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days)][::-1]
    
    # Base price trend (same for both strategies)
    price_trend = np.cumsum(np.random.randn(n_days) * 0.01) + 10
    # 关键修复：保留中文列名兼容交易引擎，同时增加英文列名用于可视化
    df_scaled = pd.DataFrame({
        '日期': dates,          # 保留中文列名供交易引擎使用
        '当日开盘': price_trend + np.random.randn(n_days) * 0.05,  # 交易引擎依赖的核心列
        '当日最高': price_trend + np.random.randn(n_days) * 0.08 + 0.1,
        '当日最低': price_trend + np.random.randn(n_days) * 0.08 - 0.1,
        '当日收盘': price_trend + np.random.randn(n_days) * 0.06,
        'Date': dates,         # 英文列名用于可视化
        'Open': price_trend + np.random.randn(n_days) * 0.05,
        'High': price_trend + np.random.randn(n_days) * 0.08 + 0.1,
        'Low': price_trend + np.random.randn(n_days) * 0.08 - 0.1,
        'Close': price_trend + np.random.randn(n_days) * 0.06
    })

    # 3. Generate two different prediction sets (different noise patterns)
    # y_pred1: High correlation with price trend (conservative prediction)
    y_pred1 = price_trend * 0.1 + np.random.randn(n_days) * 0.02  # Low noise
    # y_pred2: Lower correlation with price trend (aggressive prediction)
    y_pred2 = 12+price_trend * 0.2 - np.random.randn(n_days) * 0.08 + \
              np.where(np.arange(n_days) % 30 < 10, 0.1, -0.1)  # High noise + periodic bias

    # 4. Trading configuration (same for both strategies)
    trading_config = {
        "initial_capital": 1000000,
        "base_position": 0.2,
        "max_position_ratio": 0.8,
        "vol_target": 0.02,
        "signal_multiplier": 4.0,
        "min_trade_ratio": 0.01,
        "stop_loss": -0.05,
        "take_profit": 0.08,
        "signal_window": 20,
        "signal_buffer": 0.03
    }
    stock_code = "600000.SH"

    # 5. Run backtest for both prediction sets
    def run_backtest(y_pred, name):
        """Helper function to run single backtest"""
        simulator = get_trading_simulator(
            model_name=name,
            df_scaled=df_scaled.copy(),  # 传入包含中文列名的DataFrame
            y_pred=y_pred,
            stock_code=stock_code,
            trading_config=trading_config,
            logger=logger
        )
        result_df, total_return = simulator.simulate()
        return {
            'simulator': simulator,
            'result_df': result_df,
            'total_return': total_return,
            'equity_curve': simulator.equity_curve,
            'trade_count': simulator.trade_count,
            'max_drawdown': min((np.array(simulator.equity_curve) - 
                               np.maximum.accumulate(simulator.equity_curve)) / 
                               np.maximum.accumulate(simulator.equity_curve))
        }

    # Run backtests
    print("=== Running Backtest for Strategy 1 (Low Noise Prediction) ===")
    strategy1 = run_backtest(y_pred1, "Strategy1_LowNoise")
    
    print("\n=== Running Backtest for Strategy 2 (High Noise Prediction) ===")
    strategy2 = run_backtest(y_pred2, "Strategy2_HighNoise")

    # 6. Print comparison results
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    comparison_data = pd.DataFrame({
        'Metric': ['Total Return (%)', 'Total Trades', 'Max Drawdown (%)', 'Final Capital (CNY)'],
        'Strategy 1 (Low Noise)': [
            f"{strategy1['total_return']*100:.2f}",
            strategy1['trade_count'],
            f"{strategy1['max_drawdown']*100:.2f}",
            f"{strategy1['simulator'].cash:,.2f}"
        ],
        'Strategy 2 (High Noise)': [
            f"{strategy2['total_return']*100:.2f}",
            strategy2['trade_count'],
            f"{strategy2['max_drawdown']*100:.2f}",
            f"{strategy2['simulator'].cash:,.2f}"
        ]
    })
    print(comparison_data.to_string(index=False))

    # 7. Generate English-only comparison charts
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Ensure English font compatibility
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Equity Curve Comparison
    ax1.plot(df_scaled['Date'], strategy1['equity_curve'], label='Strategy 1 (Low Noise)', color='blue', linewidth=2)
    ax1.plot(df_scaled['Date'], strategy2['equity_curve'], label='Strategy 2 (High Noise)', color='red', linewidth=2)
    ax1.axhline(y=trading_config['initial_capital'], color='black', linestyle='--', label='Initial Capital (1,000,000 CNY)')
    ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Capital (CNY)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Cumulative Return
    cum_ret1 = (np.array(strategy1['equity_curve']) - trading_config['initial_capital']) / trading_config['initial_capital']
    cum_ret2 = (np.array(strategy2['equity_curve']) - trading_config['initial_capital']) / trading_config['initial_capital']
    ax2.plot(df_scaled['Date'], cum_ret1*100, color='blue', label='Strategy 1 (Low Noise)')
    ax2.plot(df_scaled['Date'], cum_ret2*100, color='red', label='Strategy 2 (High Noise)')
    ax2.set_title('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Drawdown Comparison
    dd1 = (np.array(strategy1['equity_curve']) - np.maximum.accumulate(strategy1['equity_curve'])) / np.maximum.accumulate(strategy1['equity_curve'])
    dd2 = (np.array(strategy2['equity_curve']) - np.maximum.accumulate(strategy2['equity_curve'])) / np.maximum.accumulate(strategy2['equity_curve'])
    ax3.fill_between(df_scaled['Date'], dd1*100, 0, color='blue', alpha=0.3, label='Strategy 1')
    ax3.fill_between(df_scaled['Date'], dd2*100, 0, color='red', alpha=0.3, label='Strategy 2')
    ax3.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Trade Count & Return Bar Chart
    metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Total Trades']
    values1 = [
        strategy1['total_return']*100,
        strategy1['max_drawdown']*100,
        strategy1['trade_count']
    ]
    values2 = [
        strategy2['total_return']*100,
        strategy2['max_drawdown']*100,
        strategy2['trade_count']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, values1, width, label='Strategy 1', color='blue', alpha=0.7)
    ax4.bar(x + width/2, values2, width, label='Strategy 2', color='red', alpha=0.7)
    ax4.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Adjust layout and save/show plot
    plt.tight_layout()
    plt.savefig('trading_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 8. Save detailed comparison to CSV
    comparison_df = pd.DataFrame({
        'Date': df_scaled['Date'],
        'Price': df_scaled['Open'],
        'Strategy1_Equity': strategy1['equity_curve'],
        'Strategy2_Equity': strategy2['equity_curve'],
        'Strategy1_Daily_Return': strategy1['result_df']['Return Rate'],
        'Strategy2_Daily_Return': strategy2['result_df']['Return Rate'],
        'Strategy1_Action': strategy1['result_df']['Strategy'],
        'Strategy2_Action': strategy2['result_df']['Strategy']
    })
    comparison_df.to_csv('strategy_comparison_results.csv', index=False)
    print("\nDetailed results saved to 'strategy_comparison_results.csv'")
    print("Comparison chart saved to 'trading_strategy_comparison.png'")

if __name__ == "__main__":
    main()