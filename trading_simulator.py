import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging  # 确保导入logging模块

class StandardTradingSimulator:
    """量化基金级交易引擎（纯回测版本）- 集成均值信号生成"""
    
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

        # ====== 仓位管理 ======
        self.base_position = trading_config.get("base_position", 0.2)
        self.max_position_ratio = trading_config.get("max_position_ratio", 0.9)
        self.vol_target = trading_config.get("vol_target", 0.02)
        self.signal_multiplier = trading_config.get("signal_multiplier", 5.0)
        self.min_trade_ratio = trading_config.get("min_trade_ratio", 0.01)

        # ====== 风控 ======
        self.stop_loss = trading_config.get("stop_loss", -0.08)
        self.take_profit = trading_config.get("take_profit", 0.10)
        self.max_drawdown = trading_config.get("max_drawdown", -0.2)
        self.cooldown_days = trading_config.get("cooldown_days", 0)
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
        self.buy_count = 0
        self.sell_count = 0
    def _log_message(self, msg):

        """兼容不同日志对象的统一日志方法（最终修复版）"""
        if self.logger is None:
            return
        
        # 清理消息格式
        clean_msg = msg.strip()
        
        # 适配不同类型的日志对象（优先级：自定义DataStatistics > 标准Logger > 文件对象 > 打印）
        if hasattr(self.logger, 'log'):
            # 检测log方法的参数数量
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
        基于均值生成买卖信号 - 修复版（标准化统一尺度）
        返回:
            处理后的交易信号数组（1=看多, -1=看空, 0=持仓）
        """
        y_pred = self.y_pred_original.copy()
        
        # ========== 核心修复：先标准化预测值 ==========
        y_pred_mean = np.mean(y_pred)
        y_pred_std = np.std(y_pred)
        # 避免除以0
        if y_pred_std < 1e-8:
            y_pred_std = 1e-8
        # Z-score标准化：转换为均值0，标准差1的分布
        y_pred_standardized = (y_pred - y_pred_mean) / y_pred_std
        
        # 1. 剔除异常值（标准化后±3σ）
        y_pred_clean = np.clip(y_pred_standardized, -3, 3)
        
        # 2. 动态滑动窗口均值（基于标准化后的数据）
        window_size = self.trading_config.get("signal_window", 30)
        y_pred_series = pd.Series(y_pred_clean)
        y_pred_rolling_mean = y_pred_series.rolling(
            window=window_size, 
            min_periods=max(1, window_size//2)
        ).mean().fillna(0)  # 标准化后均值应为0
        
        # 3. 趋势过滤（5日均线向上）
        ma5 = y_pred_series.rolling(window=5, min_periods=1).mean()
        trend = np.where(ma5 > ma5.shift(1).fillna(ma5.iloc[0]), 1, 0)
        
        # 4. 生成三态信号（带缓冲带避免频繁交易）
        buffer_ratio = self.trading_config.get("signal_buffer", 0.05)
        # 基于标准化后的数据设置阈值（缓冲带用固定值更合理）
        upper_threshold = y_pred_rolling_mean + buffer_ratio * 1  # 1是标准化后的标准差
        lower_threshold = y_pred_rolling_mean - buffer_ratio * 1
        
        # 生成基础信号
        signal = np.where(
            (y_pred_clean > upper_threshold) & (trend == 1), 1.0,  # 买入信号
            np.where(y_pred_clean < lower_threshold, -1.0, 0.0)   # 卖出/持仓信号
        )
        
        # 5. 信号平滑（减少噪音）
        if len(signal) > 5:
            signal = savgol_filter(signal, window_length=5, polyorder=1)
        
        return signal
    # =========================
    # 交易引擎核心方法
    # =========================
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

    def vol_adjusted_position(self, signal, hist_prices):
        """波动率调整仓位"""
        if len(hist_prices) < 5:
            vol_scaling = 1.0
        else:
            returns = np.diff(hist_prices) / hist_prices[:-1]
            realized_vol = np.std(returns) + 1e-6
            vol_scaling = self.vol_target / realized_vol
        
        target_ratio = self.base_position + signal * vol_scaling * self.signal_multiplier
        return np.clip(target_ratio, 0, self.max_position_ratio)

    def target_position_shares(self, price, signal, hist_prices):
        """计算目标持仓数量"""
        target_ratio = self.vol_adjusted_position(signal, hist_prices)
        capital = self.cash + self.position * price
        
        target_value = capital * target_ratio
        current_value = self.position * price
        delta_value = target_value - current_value
        
        if abs(delta_value) < price * 10:
            min_trade_value = capital * self.min_trade_ratio
            if delta_value > 0:
                delta_value = max(delta_value, min_trade_value)
            elif delta_value < 0:
                delta_value = min(delta_value, -min_trade_value)
        
        shares = int(np.round(delta_value / price))
        return shares

    def buy(self, price, shares):
        """买入股票"""
        if shares <= 0:
            return False
        
        max_possible_shares = int((self.cash) / (price * (1 + self.commission_rate + self.slippage_rate)))
        if max_possible_shares <= 0:
            return False
        
        shares = min(shares, max_possible_shares)
        if shares <= 0:
            return False
            
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, False)
        total = trade_value + cost
        
        if total > self.cash:
            shares = int(self.cash / (price * (1 + self.commission_rate + self.slippage_rate)))
            if shares <= 0:
                return False
            trade_value = shares * price
            cost = self.calculate_trading_cost(trade_value, False)
            total = trade_value + cost
        
        self.cash -= total
        if self.position > 0:
            self.avg_price = ((self.avg_price * self.position) + (price * shares)) / (self.position + shares)
        else:
            self.avg_price = price
            
        self.position += shares
        self.trade_count += 1
        self.buy_count += 1
        self.total_trading_cost += cost
        
        self._log_message(f"[{self.stock_code}] 买入 {shares} 股，价格 {price:.2f}，剩余现金 {self.cash:.2f}")
        
        return True

    def sell(self, price, fraction=1.0):
        """卖出股票"""
        if self.position == 0:
            return 0
        
        shares = int(self.position * fraction)
        if shares <= 0:
            return 0
            
        trade_value = shares * price
        cost = self.calculate_trading_cost(trade_value, True)
        profit = (price - self.avg_price) * shares - cost
        
        self.cash += trade_value - cost
        self.position -= shares
        
        if self.position == 0:
            self.avg_price = 0
            
        self.trade_count += 1
        self.sell_count += 1
        self.total_trading_cost += cost
        
        self._log_message(f"[{self.stock_code}] 卖出 {shares} 股，价格 {price:.2f}，获利 {profit:.2f}")
        
        return profit

    def simulate(self):
        """执行交易模拟回测"""
        hist_prices = []
        profits = []
        strategies = []

        # 生成基于均值的交易信号
        self.y_pred = self._generate_trading_signal()
        
        # 趋势增强（可选）
        if len(self.y_pred) > 5:
            ma_short = pd.Series(self.y_pred).rolling(window=3).mean().fillna(0).values
            ma_long = pd.Series(self.y_pred).rolling(window=10).mean().fillna(0).values
            trend_signal = np.where(ma_short > ma_long, 1.0, -1.0)
            self.y_pred = self.y_pred * 0.7 + trend_signal * 0.3

        # 逐行执行交易
        for i, row in self.df_scaled.iterrows():
            price = row['当日开盘'] if '当日开盘' in row else row.iloc[0]
            pred = self.y_pred[i]

            strategy = "Hold"
            daily_return = 0

            hist_prices.append(price)

            # 风控检查
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
                # 计算目标仓位
                shares = self.target_position_shares(price, pred, hist_prices)
                
                # 定期调仓检查
                if i % 10 == 0:
                    current_ratio = (self.position * price) / (self.cash + self.position * price + 1e-8)
                    target_ratio = self.vol_adjusted_position(pred, hist_prices)
                    if abs(current_ratio - target_ratio) > 0.02:
                        if target_ratio > current_ratio:
                            needed_value = (self.cash + self.position * price) * (target_ratio - current_ratio)
                            shares = max(shares, int(needed_value / price))
                        else:
                            needed_value = (self.cash + self.position * price) * (current_ratio - target_ratio)
                            shares = min(shares, -int(needed_value / price))

                # 执行交易
                if shares > 0:
                    success = self.buy(price, shares)
                    if success:
                        strategy = "Buy"
                        self.last_trade_day = i
                elif shares < 0:
                    profit = self.sell(price, fraction=min(1, -shares / max(1, self.position)))
                    daily_return = profit / self.initial_capital
                    strategy = "Sell"
                    self.last_trade_day = i
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
        self._log_message(f"\n{self.stock_code} 量化基金级交易统计:")
        self._log_message(f"总交易次数: {self.trade_count}")
        self._log_message(f"买入次数: {self.buy_count}")
        self._log_message(f"卖出次数: {self.sell_count}")
        self._log_message(f"买卖比: {self.buy_count/max(1, self.sell_count):.2f}")
        self._log_message(f"总交易成本: {self.total_trading_cost:.2f}")
        self._log_message(f"平均交易成本: {avg_trade_cost:.2f}")
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
    # 设置默认配置（新增信号相关配置）
    default_config = {
        "initial_capital": 1_000_000,
        "base_position": 0.2,
        "max_position_ratio": 0.9,
        "vol_target": 0.02,
        "signal_multiplier": 5.0,
        "min_trade_ratio": 0.005,
        "stop_loss": -0.08,
        "take_profit": 0.10,
        "max_drawdown": -0.2,
        "cooldown_days": 0,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0002,
        "stamp_tax": 0.001,
        "min_commission": 5,
        "signal_window": 30,    # 信号滑动窗口大小
        "signal_buffer": 0.05   # 信号缓冲带比例
    }
    
    # 合并用户配置和默认配置
    for key, value in default_config.items():
        if key not in trading_config:
            trading_config[key] = value
    
    return StandardTradingSimulator(df_scaled, y_pred, stock_code, trading_config, logger)
