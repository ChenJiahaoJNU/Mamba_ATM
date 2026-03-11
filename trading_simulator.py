import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging  # 确保导入logging模块

class StandardTradingSimulator:
    """量化基金级交易引擎（纯回测版本）- 简化版：基于预测值直接生成买卖信号"""
    
    def __init__(self, df_scaled, y_pred, stock_code, trading_config, logger):
        """
        交易引擎初始化
        参数:
            df_scaled: 标准化后的股票数据DataFrame
            y_pred: 原始预测值数组
            stock_code: 股票代码
            trading_config: 交易配置字典
            logger: 日志记录器
        """
        self._init_trading_simulator(df_scaled, y_pred, stock_code, trading_config, logger)

    def _init_trading_simulator(self, df_scaled, y_pred, stock_code, trading_config, logger):
        """交易引擎初始化逻辑"""
        self.df_scaled = df_scaled.reset_index(drop=True)
        self.y_pred_original = np.array(y_pred)  # 保存原始预测值
        self.stock_code = stock_code
        self.trading_config = trading_config
        self.logger = logger

        # ====== 资金 ======
        self.initial_capital = trading_config.get("initial_capital", 1_000_000)
        self.cash = self.initial_capital
        self.position = 0
        self.avg_price = 0

        # ====== 信号阈值（核心：由预测值直接决定买卖） ======
        self.buy_threshold = trading_config.get("buy_threshold", 0.1)    # 正向信号阈值
        self.sell_threshold = trading_config.get("sell_threshold", -0.1) # 负向信号阈值
        self.position_ratio = trading_config.get("position_ratio", 0.8)  # 每次买入用多少比例的现金（0-1）

        # ====== 风控 ======
        self.stop_loss = trading_config.get("stop_loss", -0.08)
        self.take_profit = trading_config.get("take_profit", 0.10)

        # ====== 成本 ======
        self.commission_rate = trading_config.get("commission_rate", 0.0003)
        self.slippage_rate = trading_config.get("slippage_rate", 0.0002)
        self.stamp_tax = trading_config.get("stamp_tax", 0.001)
        self.min_commission = trading_config.get("min_commission", 5)

        # ====== 统计 ======
        self.equity_curve = []
        self.trade_count = 0
        self.total_trading_cost = 0
        self.buy_count = 0
        self.sell_count = 0

    def _log_message(self, msg):
        """兼容不同日志对象的统一日志方法"""
        if self.logger is None:
            return
        
        # 清理消息格式
        clean_msg = msg.strip()
        
        # 适配不同类型的日志对象
        if hasattr(self.logger, 'log'):
            import inspect
            sig = inspect.signature(self.logger.log)
            params = list(sig.parameters.values())
            # 如果log方法只有self/1个参数，直接传消息
            if len(params) <= 1:
                self.logger.log(clean_msg)
            # 如果是标准Logger的log方法（需要级别+消息）
            else:
                self.logger.log(logging.INFO, clean_msg)

    def _generate_trading_signal(self):
        """
        简化版：仅做信号平滑，不修改核心预测值趋势
        返回: 平滑后的预测信号数组
        """
        y_pred = self.y_pred_original.copy()
        
        # 简单平滑（减少噪音）
        if len(y_pred) > 5:
            y_pred = savgol_filter(y_pred, window_length=5, polyorder=1)
        
        return y_pred

    def calculate_trading_cost(self, trade_value, is_sell=True):
        """计算交易成本"""
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        stamp_tax = trade_value * self.stamp_tax if is_sell else 0
        return commission + slippage + stamp_tax

    def risk_control(self, price):
        """风控检查（止损/止盈）"""
        if self.position == 0:
            return None
        pnl = (price - self.avg_price) / self.avg_price
        if pnl <= self.stop_loss:
            return "stop_loss"
        if pnl >= self.take_profit:
            return "take_profit"
        return None

    def buy(self, price):
        """买入股票：基于固定比例买入，核心由信号决定"""
        if self.position > 0:  # 已有仓位则不重复买入
            return False
        
        # 计算可买入的最大数量（使用设定比例的现金）
        usable_cash = self.cash * self.position_ratio
        max_possible_shares = int(usable_cash / (price * (1 + self.commission_rate + self.slippage_rate)))
        
        if max_possible_shares <= 0:
            return False
            
        trade_value = max_possible_shares * price
        cost = self.calculate_trading_cost(trade_value, False)
        total = trade_value + cost
        
        if total > self.cash:
            return False
        
        self.cash -= total
        self.avg_price = price
        self.position += max_possible_shares
        self.trade_count += 1
        self.buy_count += 1
        self.total_trading_cost += cost
        
        self._log_message(f"[{self.stock_code}] 买入 {max_possible_shares} 股，价格 {price:.2f}，剩余现金 {self.cash:.2f}")
        
        return True

    def sell(self, price):
        """卖出股票：清仓，核心由信号决定"""
        if self.position == 0:
            return 0
        
        shares = self.position
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, True)
        profit = (price - self.avg_price) * shares - cost
        
        self.cash += trade_value - cost
        self.position = 0
        self.avg_price = 0
        
        self.trade_count += 1
        self.sell_count += 1
        self.total_trading_cost += cost
        
        self._log_message(f"[{self.stock_code}] 卖出 {shares} 股，价格 {price:.2f}，获利 {profit:.2f}")
        
        return profit

    def simulate(self):
        """执行交易模拟回测（简化版：由y_pred直接驱动）"""
        profits = []
        strategies = []

        # 生成平滑后的交易信号（核心：保留y_pred的原始趋势）
        self.y_pred = self._generate_trading_signal()

        # 逐行执行交易
        for i, row in self.df_scaled.iterrows():
            price = row['当日开盘'] if '当日开盘' in row else row.iloc[0]
            pred = self.y_pred[i]  # 核心：直接使用预测值

            strategy = "Hold"
            daily_return = 0

            # 第一步：风控检查（止损/止盈优先）
            risk = self.risk_control(price)
            if risk == "stop_loss":
                profit = self.sell(price)
                daily_return = profit / self.initial_capital
                strategy = "StopLoss"
            elif risk == "take_profit":
                profit = self.sell(price)
                daily_return = profit / self.initial_capital
                strategy = "TakeProfit"
            else:
                # 第二步：由预测值直接决定买卖
                if pred > self.buy_threshold:  # 正向信号：买入
                    success = self.buy(price)
                    if success:
                        strategy = "Buy"
                elif pred < self.sell_threshold:  # 负向信号：卖出
                    profit = self.sell(price)
                    daily_return = profit / self.initial_capital
                    strategy = "Sell"
                else:  # 中性信号：持仓
                    strategy = "Hold"

            # 记录资产曲线
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            profits.append(daily_return)
            strategies.append(strategy)

        # 强制平仓剩余仓位
        if self.position > 0:
            last_price = self.df_scaled.iloc[-1]['当日开盘'] if '当日开盘' in self.df_scaled.columns else self.df_scaled.iloc[-1].iloc[0]
            profit = self.sell(last_price)
            profits[-1] += profit / self.initial_capital
            strategies[-1] = "Sell (Force)"

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

        # 保存结果到DataFrame
        self.df_scaled['Return Rate'] = profits
        self.df_scaled['Strategy'] = strategies
        self.df_scaled['Original Prediction'] = self.y_pred_original
        self.df_scaled['Trading Signal'] = self.y_pred
        self.df_scaled['Cumulative Return'] = cumulative_return

        # 记录日志
        self._log_message(f"\n{self.stock_code} 量化交易统计（简化版）:")
        self._log_message(f"总交易次数: {self.trade_count}")
        self._log_message(f"买入次数: {self.buy_count}")
        self._log_message(f"卖出次数: {self.sell_count}")
        self._log_message(f"总交易成本: {self.total_trading_cost:.2f}")
        self._log_message(f"总收益率: {total_return:.4f}")
        self._log_message(f"Sharpe Ratio: {sharpe:.3f}")
        self._log_message(f"最大回撤: {max_drawdown:.3f}")

        return self.df_scaled, total_return

# ===============================
# 工厂函数
# ===============================
def get_trading_simulator(model_name, df_scaled, y_pred, stock_code, trading_config, logger):
    """
    创建交易模拟器实例的工厂函数
    参数:
        model_name: 模型名称（仅用于日志）
        df_scaled: 标准化数据
        y_pred: 原始预测值
        stock_code: 股票代码
        trading_config: 交易配置
        logger: 日志器
    返回:
        StandardTradingSimulator实例
    """
    # 设置默认配置（简化版核心配置）
    default_config = {
        "initial_capital": 1_000_000,
        "buy_threshold": 0.1,       # 买入信号阈值
        "sell_threshold": -0.1,     # 卖出信号阈值
        "position_ratio": 0.8,      # 买入用80%现金
        "stop_loss": -0.08,
        "take_profit": 0.10,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0002,
        "stamp_tax": 0.001,
        "min_commission": 5
    }
    
    # 合并用户配置和默认配置
    for key, value in default_config.items():
        if key not in trading_config:
            trading_config[key] = value
    
    return StandardTradingSimulator(df_scaled, y_pred, stock_code, trading_config, logger)