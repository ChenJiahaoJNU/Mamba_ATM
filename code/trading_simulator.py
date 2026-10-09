import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging

class StandardTradingSimulator:
    """量化交易引擎（修复版）- 自适应阈值 + 强制卖出"""
    
    def __init__(self, df_scaled, y_pred, stock_code, trading_config, logger):
        self._init_trading_simulator(df_scaled, y_pred, stock_code, trading_config, logger)

    def _init_trading_simulator(self, df_scaled, y_pred, stock_code, trading_config, logger):
        self.df_scaled = df_scaled.reset_index(drop=True)
        self.y_pred_original = np.array(y_pred)
        self.stock_code = stock_code
        self.trading_config = trading_config
        self.logger = logger

        # 资金配置
        self.initial_capital = trading_config.get("initial_capital", 1_000_000)
        self.cash = self.initial_capital
        self.position = 0
        self.avg_price = 0

        # ====== 核心修复：自适应阈值（基于y_pred分布） ======
        self.y_pred_mean = np.mean(self.y_pred_original)
        self.y_pred_std = np.std(self.y_pred_original)
        # 自适应买卖阈值（基于1个标准差）
        self.buy_sigma = trading_config.get("buy_sigma", 0.5)    # 均值+0.5σ为买入阈值
        self.sell_sigma = trading_config.get("sell_sigma", -0.5) # 均值-0.5σ为卖出阈值
        self.buy_threshold = self.y_pred_mean + self.buy_sigma * self.y_pred_std
        self.sell_threshold = self.y_pred_mean + self.sell_sigma * self.y_pred_std

        # 仓位配置
        self.position_ratio = trading_config.get("position_ratio", 0.8)

        # 风控（强化：强制止盈止损）
        self.stop_loss = trading_config.get("stop_loss", -0.08)
        self.take_profit = trading_config.get("take_profit", 0.10)
        self.hold_days_limit = trading_config.get("hold_days_limit", 10)  # 持仓超10天强制卖出
        self.hold_days = 0  # 持仓天数计数器

        # 成本配置
        self.commission_rate = trading_config.get("commission_rate", 0.0003)
        self.slippage_rate = trading_config.get("slippage_rate", 0.0002)
        self.stamp_tax = trading_config.get("stamp_tax", 0.001)
        self.min_commission = trading_config.get("min_commission", 5)

        # 统计
        self.equity_curve = []
        self.trade_count = 0
        self.total_trading_cost = 0
        self.buy_count = 0
        self.sell_count = 0

    def _log_message(self, msg):
        if self.logger is None:
            return
        clean_msg = msg.strip()
        if hasattr(self.logger, 'log'):
            import inspect
            sig = inspect.signature(self.logger.log)
            params = list(sig.parameters.values())
            if len(params) <= 1:
                self.logger.log(clean_msg)
            else:
                self.logger.log(logging.INFO, clean_msg)

    def _generate_trading_signal(self):
        """简化：仅平滑，不修改分布"""
        y_pred = self.y_pred_original.copy()
        if len(y_pred) > 5:
            y_pred = savgol_filter(y_pred, window_length=5, polyorder=1)
        return y_pred

    def calculate_trading_cost(self, trade_value, is_sell=True):
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        stamp_tax = trade_value * self.stamp_tax if is_sell else 0
        return commission + slippage + stamp_tax

    def risk_control(self, price):
        """强化风控：增加持仓天数限制"""
        if self.position == 0:
            self.hold_days = 0
            return None
        
        # 持仓天数+1
        self.hold_days += 1
        
        # 止损
        pnl = (price - self.avg_price) / self.avg_price
        if pnl <= self.stop_loss:
            return "stop_loss"
        # 止盈
        if pnl >= self.take_profit:
            return "take_profit"
        # 持仓超时强制卖出
        if self.hold_days >= self.hold_days_limit:
            return "hold_timeout"
        
        return None

    def buy(self, price):
        """修复：避免重复买入"""
        if self.position > 0:
            self._log_message(f"[{self.stock_code}] 已有仓位，跳过买入")
            return False
        
        usable_cash = self.cash * self.position_ratio
        max_possible_shares = int(usable_cash / (price * (1 + self.commission_rate + self.slippage_rate)))
        
        if max_possible_shares <= 0:
            self._log_message(f"[{self.stock_code}] 现金不足，无法买入")
            return False
            
        trade_value = max_possible_shares * price
        cost = self.calculate_trading_cost(trade_value, False)
        total = trade_value + cost
        
        if total > self.cash:
            self._log_message(f"[{self.stock_code}] 交易成本超现金，无法买入")
            return False
        
        self.cash -= total
        self.avg_price = price
        self.position += max_possible_shares
        self.trade_count += 1
        self.buy_count += 1
        self.total_trading_cost += cost
        self.hold_days = 0  # 重置持仓天数
        
        self._log_message(f"[{self.stock_code}] 买入 {max_possible_shares} 股，价格 {price:.2f}，剩余现金 {self.cash:.2f}")
        return True

    def sell(self, price, reason="normal"):
        """修复：明确卖出原因，确保清仓"""
        if self.position == 0:
            self._log_message(f"[{self.stock_code}] 无仓位，跳过卖出")
            return 0
        
        shares = self.position
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, True)
        profit = (price - self.avg_price) * shares - cost
        
        self.cash += trade_value - cost
        self.position = 0
        self.avg_price = 0
        self.hold_days = 0  # 重置持仓天数
        
        self.trade_count += 1
        self.sell_count += 1
        self.total_trading_cost += cost
        
        self._log_message(f"[{self.stock_code}] {reason} 卖出 {shares} 股，价格 {price:.2f}，获利 {profit:.2f}")
        return profit

    def simulate(self):
        profits = []
        strategies = []
        self.y_pred = self._generate_trading_signal()

        # 打印阈值信息（调试用）
        self._log_message(f"\n[{self.stock_code}] 信号阈值：均值={self.y_pred_mean:.4f}，标准差={self.y_pred_std:.4f}")
        self._log_message(f"买入阈值={self.buy_threshold:.4f}，卖出阈值={self.sell_threshold:.4f}")

        for i, row in self.df_scaled.iterrows():
            price = row['当日开盘'] if '当日开盘' in row else row.iloc[0]
            pred = self.y_pred[i]

            strategy = "Hold"
            daily_return = 0

            # 风控检查（优先执行）
            risk = self.risk_control(price)
            if risk == "stop_loss":
                profit = self.sell(price, reason="止损")
                daily_return = profit / self.initial_capital
                strategy = "StopLoss"
            elif risk == "take_profit":
                profit = self.sell(price, reason="止盈")
                daily_return = profit / self.initial_capital
                strategy = "TakeProfit"
            elif risk == "hold_timeout":
                profit = self.sell(price, reason="持仓超时")
                daily_return = profit / self.initial_capital
                strategy = "HoldTimeout"
            else:
                # 基于自适应阈值的买卖决策
                if pred > self.buy_threshold:
                    success = self.buy(price)
                    if success:
                        strategy = "Buy"
                elif pred < self.sell_threshold:
                    profit = self.sell(price, reason="信号卖出")
                    daily_return = profit / self.initial_capital
                    strategy = "Sell"
                else:
                    strategy = "Hold"

            # 记录资产曲线
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            profits.append(daily_return)
            strategies.append(strategy)

        # 强制平仓剩余仓位
        if self.position > 0:
            last_price = self.df_scaled.iloc[-1]['当日开盘'] if '当日开盘' in self.df_scaled.columns else self.df_scaled.iloc[-1].iloc[0]
            profit = self.sell(last_price, reason="强制平仓")
            profits[-1] += profit / self.initial_capital
            strategies[-1] = "ForceSell"

        # 计算回测指标
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        cumulative_return = (equity - self.initial_capital) / self.initial_capital
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(252) if np.std(returns) > 0 else 0
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax)/cummax
        max_drawdown = drawdown.min()
        total_return = cumulative_return[-1] if len(cumulative_return) > 0 else 0
        avg_trade_cost = self.total_trading_cost / self.trade_count if self.trade_count > 0 else 0

        # 保存结果
        self.df_scaled['Return Rate'] = profits
        self.df_scaled['Strategy'] = strategies
        self.df_scaled['Original Prediction'] = self.y_pred_original
        self.df_scaled['Trading Signal'] = self.y_pred
        self.df_scaled['Cumulative Return'] = cumulative_return

        # 日志
        self._log_message(f"\n{self.stock_code} 量化交易统计（修复版）:")
        self._log_message(f"预测值均值={self.y_pred_mean:.4f}，标准差={self.y_pred_std:.4f}")
        self._log_message(f"买入阈值={self.buy_threshold:.4f}，卖出阈值={self.sell_threshold:.4f}")
        self._log_message(f"总交易次数: {self.trade_count} (买:{self.buy_count} 卖:{self.sell_count})")
        self._log_message(f"总收益率: {total_return:.4f} | 最大回撤: {max_drawdown:.3f} | Sharpe: {sharpe:.3f}")

        return self.df_scaled, total_return

# 工厂函数
def get_trading_simulator(model_name, df_scaled, y_pred, stock_code, trading_config, logger):
    default_config = {
        "initial_capital": 1_000_000,
        "buy_sigma": 0.5,        # 买入阈值=均值+0.5σ
        "sell_sigma": -0.5,      # 卖出阈值=均值-0.5σ
        "position_ratio": 0.8,
        "stop_loss": -0.08,
        "take_profit": 0.10,
        "hold_days_limit": 10,   # 持仓超10天强制卖出
        "commission_rate": 0.0003,
        "slippage_rate": 0.0002,
        "stamp_tax": 0.001,
        "min_commission": 5
    }
    for key, value in default_config.items():
        if key not in trading_config:
            trading_config[key] = value
    return StandardTradingSimulator(df_scaled, y_pred, stock_code, trading_config, logger)