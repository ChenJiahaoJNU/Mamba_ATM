

class StandardTradingSimulator:
    """量化基金级交易引擎（兼容模型训练器初始化版本）"""
    
    def __init__(self, *args, **kwargs):
        """
        兼容两种初始化方式：
        方式1（交易引擎）: __init__(df_scaled, y_pred, stock_code, trading_config, logger)
        方式2（模型训练器）: __init__(model, train_x, train_y, criterion, model_name, device, hyper_params, logger)
        """
        # 识别初始化类型
        if len(args) >= 5 and isinstance(args[0], (pd.DataFrame, np.ndarray)):
            # 交易引擎初始化
            self._init_trading_simulator(*args, **kwargs)
        elif len(args) >= 7 and isinstance(args[0], torch.nn.Module):
            # 模型训练器初始化
            self._init_model_trainer(*args, **kwargs)
        else:
            raise ValueError("无法识别初始化参数，请使用交易引擎或模型训练器的参数格式")

    def _init_trading_simulator(self, df_scaled, y_pred, stock_code, trading_config, logger):
        """交易引擎初始化逻辑（激进版）"""
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

        # ====== 仓位管理（大幅放宽） ======
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

    def _init_model_trainer(self, model, train_x, train_y, criterion, model_name, device, hyper_params, logger):
        """模型训练器初始化逻辑（原EnhancedModelTrainer）"""
        self.model = model.to(device)
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.criterion = criterion
        self.model_name = model_name
        self.device = device
        self.hyper_params = hyper_params
        self.logger = logger

    # =========================
    # 交易引擎核心方法
    # =========================
    def calculate_trading_cost(self, trade_value, is_sell=True):
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        stamp_tax = trade_value * self.stamp_tax if is_sell else 0
        return commission + slippage + stamp_tax

    def risk_control(self, price):
        if self.position == 0:
            return None
        pnl = (price - self.avg_price) / self.avg_price
        if pnl <= self.stop_loss:
            return "stop_loss"
        if pnl >= self.take_profit:
            return "take_profit"
        return None

    def vol_adjusted_position(self, signal, hist_prices):
        if len(hist_prices) < 5:
            vol_scaling = 1.0
        else:
            returns = np.diff(hist_prices) / hist_prices[:-1]
            realized_vol = np.std(returns) + 1e-6
            vol_scaling = self.vol_target / realized_vol
        
        target_ratio = self.base_position + signal * vol_scaling * self.signal_multiplier
        return np.clip(target_ratio, 0, self.max_position_ratio)

    def target_position_shares(self, price, signal, hist_prices):
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
        
        if self.logger:
            self.logger.write(f"[{self.stock_code}] 买入 {shares} 股，价格 {price:.2f}，剩余现金 {self.cash:.2f}\n")
        
        return True

    def sell(self, price, fraction=1.0):
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
        
        if self.logger:
            self.logger.write(f"[{self.stock_code}] 卖出 {shares} 股，价格 {price:.2f}，获利 {profit:.2f}\n")
        
        return profit

    def simulate(self):
        hist_prices = []
        profits = []
        strategies = []

        y_pred_original = self.y_pred.copy()
        self.y_pred = (self.y_pred - np.mean(self.y_pred)) / (np.std(self.y_pred) + 1e-8)
        self.y_pred = self.y_pred * 2.0
        
        if len(self.y_pred) > 5:
            ma_short = pd.Series(self.y_pred).rolling(window=3).mean().fillna(0).values
            ma_long = pd.Series(self.y_pred).rolling(window=10).mean().fillna(0).values
            trend_signal = np.where(ma_short > ma_long, 1.0, -1.0)
            self.y_pred = self.y_pred * 0.7 + trend_signal * 0.3

        for i, row in self.df_scaled.iterrows():
            price = row['当日开盘'] if '当日开盘' in row else row.iloc[0]
            pred = self.y_pred[i]

            strategy = "Hold"
            daily_return = 0

            hist_prices.append(price)

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
                shares = self.target_position_shares(price, pred, hist_prices)
                
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

            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            profits.append(daily_return)
            strategies.append(strategy)

        if self.position > 0:
            last_price = self.df_scaled.iloc[-1]['当日开盘']
            profit = self.sell(last_price)
            profits[-1] += profit / self.initial_capital
            strategies[-1] = "Sell (Force)"

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        cumulative_return = (equity - self.initial_capital) / self.initial_capital
        sharpe = np.mean(returns)/np.std(returns)*np.sqrt(252) if np.std(returns) > 0 else 0
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax)/cummax
        max_drawdown = drawdown.min()
        total_return = cumulative_return[-1]
        avg_trade_cost = self.total_trading_cost / self.trade_count if self.trade_count > 0 else 0

        self.df_scaled['Return Rate'] = profits
        self.df_scaled['Strategy'] = strategies
        self.df_scaled['Predicted Value'] = y_pred_original
        self.df_scaled['Enhanced Signal'] = self.y_pred
        self.df_scaled['Cumulative Return'] = cumulative_return

        if self.logger:
            self.logger.write(f"\n{self.stock_code} 量化基金级交易统计:\n")
            self.logger.write(f"总交易次数: {self.trade_count}\n")
            self.logger.write(f"买入次数: {self.buy_count}\n")
            self.logger.write(f"卖出次数: {self.sell_count}\n")
            self.logger.write(f"买卖比: {self.buy_count/max(1, self.sell_count):.2f}\n")
            self.logger.write(f"总交易成本: {self.total_trading_cost:.2f}\n")
            self.logger.write(f"平均交易成本: {avg_trade_cost:.2f}\n")
            self.logger.write(f"总收益率: {total_return:.4f}\n")
            self.logger.write(f"Sharpe Ratio: {sharpe:.3f}\n")
            self.logger.write(f"最大回撤: {max_drawdown:.3f}\n\n")

        return self.df_scaled, total_return

    # =========================
    # 模型训练器核心方法
    # =========================
    def train(self):
        """原EnhancedModelTrainer的train方法"""
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

# 保持类名统一
EnhancedModelTrainer = StandardTradingSimulator

# ===============================
# 工厂函数
# ===============================
def get_trading_simulator(model_name, df_scaled, y_pred, stock_code, trading_config, logger):
    # 更新默认配置
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
        "min_commission": 5
    }
    for key, value in default_config.items():
        if key not in trading_config:
            trading_config[key] = value
    
    return StandardTradingSimulator(df_scaled, y_pred, stock_code, trading_config, logger)


import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os