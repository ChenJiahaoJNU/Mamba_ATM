import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

    

class StandardTradingSimulator:
    """量化基金级交易引擎（完全兼容原接口）"""

    def __init__(self, df_scaled, y_pred, stock_code, trading_config, logger):
        self.df_scaled = df_scaled.reset_index(drop=True)
        self.y_pred = np.array(y_pred)
        self.stock_code = stock_code
        self.trading_config = trading_config
        self.logger = logger

        # ====== 资金 ======
        self.initial_capital = trading_config.get("initial_capital", 1_000_000)
        self.cash = self.initial_capital
        self.position = 0
        self.avg_price = 0

        # ====== 仓位管理 ======
        self.base_position = trading_config.get("base_position", 0.05)
        self.max_position_ratio = trading_config.get("max_position_ratio", 0.8)
        self.vol_target = trading_config.get("vol_target", 0.02)  # 目标日波动率

        # ====== 风控 ======
        self.stop_loss = trading_config.get("stop_loss", -0.03)
        self.take_profit = trading_config.get("take_profit", 0.05)
        self.max_drawdown = trading_config.get("max_drawdown", -0.2)
        self.cooldown_days = trading_config.get("cooldown_days", 1)
        self.last_trade_day = -100

        # ====== 成本 ======
        self.commission_rate = trading_config.get("commission_rate", 0.0003)
        self.slippage_rate = trading_config.get("slippage_rate", 0.0002)
        self.stamp_tax = trading_config.get("stamp_tax", 0.001)
        self.min_commission = trading_config.get("min_commission", 5)

        # ====== 统计 ======
        self.equity_curve = []
        self.trade_count = 0
        self.total_trading_cost = 0

    # =========================
    # 交易成本
    # =========================
    def calculate_trading_cost(self, trade_value, is_sell=True):
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        stamp_tax = trade_value * self.stamp_tax if is_sell else 0
        return commission + slippage + stamp_tax

    # =========================
    # 风控
    # =========================
    def risk_control(self, price):
        if self.position == 0:
            return None
        pnl = (price - self.avg_price) / self.avg_price
        if pnl <= self.stop_loss:
            return "stop_loss"
        if pnl >= self.take_profit:
            return "take_profit"
        return None

    # =========================
    # 波动率调仓
    # =========================
    def vol_adjusted_position(self, signal, hist_prices):
        if len(hist_prices) < 2:
            return self.base_position
        returns = np.diff(hist_prices) / hist_prices[:-1]
        realized_vol = np.std(returns)
        if realized_vol == 0:
            return self.base_position
        scaling = self.vol_target / realized_vol
        return np.clip(self.base_position + signal * scaling,
                       -self.max_position_ratio,
                       self.max_position_ratio)

    # =========================
    # 动态仓位计算（Kelly + 波动率调仓）
    # =========================
    def target_position_shares(self, price, signal, hist_prices):
        target_ratio = self.vol_adjusted_position(signal, hist_prices)
        capital = self.cash + self.position * price
        target_value = capital * target_ratio
        current_value = self.position * price
        delta_value = target_value - current_value
        shares = int(delta_value // price)
        return shares

    # =========================
    # 买入
    # =========================
    def buy(self, price, shares):
        if shares <= 0:
            return False
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, False)
        total = trade_value + cost
        if total > self.cash:
            shares = int(self.cash // (price * 1.001))  # 保守买入
            if shares <= 0:
                return False
            trade_value = shares * price
            cost = self.calculate_trading_cost(trade_value, False)
            total = trade_value + cost
        self.cash -= total
        self.avg_price = ((self.avg_price * self.position) + (price * shares)) / (self.position + shares) if self.position > 0 else price
        self.position += shares
        self.trade_count += 2
        self.total_trading_cost += cost
        return True

    # =========================
    # 卖出
    # =========================
    def sell(self, price, fraction=1.0):
        if self.position == 0:
            return 0
        shares = int(self.position * fraction)
        if shares == 0:
            return 0
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, True)
        profit = (price - self.avg_price) * shares - cost
        self.cash += trade_value - cost
        self.position -= shares
        if self.position == 0:
            self.avg_price = 0
        self.trade_count += 2
        self.total_trading_cost += cost
        return profit

    # =========================
    # 核心回测
    # =========================
    def simulate(self):
        hist_prices = []
        profits = []
        strategies = []

        # 平滑预测信号
        if len(self.y_pred) > 5:
            self.y_pred = savgol_filter(self.y_pred, 5, 2)

        for i, row in self.df_scaled.iterrows():
            price = row['当日开盘'] if '当日开盘' in row else row.iloc[0]
            pred = self.y_pred[i]

            strategy = "Hold"
            daily_return = 0

            # 累积历史价格
            hist_prices.append(price)

            # 风控
            risk = self.risk_control(price)
            if risk == "stop_loss":
                profit = self.sell(price)
                daily_return = profit / self.initial_capital
                strategy = "StopLoss"
            elif risk == "take_profit":
                profit = self.sell(price, 0.5)
                daily_return = profit / self.initial_capital
                strategy = "TakeProfit"
            else:
                if i - self.last_trade_day >= self.cooldown_days:
                    shares = self.target_position_shares(price, pred, hist_prices)
                    if shares > 0:
                        self.buy(price, shares)
                        strategy = "Buy"
                        self.last_trade_day = i
                    elif shares < 0:
                        profit = self.sell(price, fraction=min(1, -shares / max(1, self.position)))
                        daily_return = profit / self.initial_capital
                        strategy = "Sell"
                        self.last_trade_day = i
                    else:
                        strategy = "Hold"

            # 更新资金曲线
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            profits.append(daily_return)
            strategies.append(strategy)

        # 强制平仓
        if self.position > 0:
            last_price = self.df_scaled.iloc[-1]['当日开盘']
            profit = self.sell(last_price)
            profits[-1] += profit / self.initial_capital
            strategies[-1] = "Sell (Force)"

        # 计算指标
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        cumulative_return = (equity - self.initial_capital) / self.initial_capital
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(252) if np.std(returns) > 0 else 0
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax)/cummax
        max_drawdown = drawdown.min()
        total_return = cumulative_return[-1]
        avg_trade_cost = self.total_trading_cost / self.trade_count if self.trade_count > 0 else 0

        # 写回 dataframe
        self.df_scaled['Return Rate'] = profits
        self.df_scaled['Strategy'] = strategies
        self.df_scaled['Predicted Value'] = self.y_pred
        self.df_scaled['Cumulative Return'] = cumulative_return

        # 日志
        if self.logger:
            self.logger.write(f"\n{self.stock_code} 量化基金级交易统计:\n")
            self.logger.write(f"总交易次数: {self.trade_count}\n")
            self.logger.write(f"总交易成本: {self.total_trading_cost:.2f}\n")
            self.logger.write(f"平均交易成本: {avg_trade_cost:.2f}\n")
            self.logger.write(f"总收益率: {total_return:.4f}\n")
            self.logger.write(f"Sharpe Ratio: {sharpe:.3f}\n")
            self.logger.write(f"最大回撤: {max_drawdown:.3f}\n\n")

        return self.df_scaled, total_return
    
    
    
    



# ===============================
# 工厂函数
# ===============================

def get_trading_simulator(model_name, df_scaled, y_pred, stock_code, trading_config, logger):

    return StandardTradingSimulator(df_scaled, y_pred, stock_code, trading_config, logger)