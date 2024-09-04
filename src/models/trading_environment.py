import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, data, initial_balance=100, leverage=786, margin_call_threshold=0.5, max_position_size=0.1):
        self.data = data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.margin_call_threshold = margin_call_threshold
        self.max_position_size = max_position_size
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position_size = 0
        self.position_type = None
        self.current_step = 0
        self.entry_price = None
        self.unrealized_pnl = 0
        self.max_equity = self.initial_balance
        self.min_equity = self.initial_balance
        self.max_drawdown = 0
        self.drawdown = 0
        self.last_trade_step = 0
        self.trades_count = 0
        return self._get_observation()

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}

        current_price = round(self.data.iloc[self.current_step]['close'], 3)
        prev_equity = self.balance + self.unrealized_pnl
        
        # Execute trading action
        if action == 0 and self.position_size == 0:  # Buy Entry
            self.position_size = min(self.max_position_size, self.balance / (current_price * 100000 / self.leverage))
            self.position_type = 'long'
            self.entry_price = current_price
            self.last_trade_step = self.current_step
            self.trades_count += 1
        elif action == 1 and self.position_type == 'long':  # Buy Close
            profit_loss = (current_price - self.entry_price) * self.position_size * 100000
            self.balance += profit_loss
            self.position_size = 0
            self.position_type = None
            self.entry_price = None
            self.trades_count += 1
        elif action == 2 and self.position_size == 0:  # Sell Entry
            self.position_size = min(self.max_position_size, self.balance / (current_price * 100000 / self.leverage))
            self.position_type = 'short'
            self.entry_price = current_price
            self.last_trade_step = self.current_step
            self.trades_count += 1
        elif action == 3 and self.position_type == 'short':  # Sell Close
            profit_loss = (self.entry_price - current_price) * self.position_size * 100000
            self.balance += profit_loss
            self.position_size = 0
            self.position_type = None
            self.entry_price = None
            self.trades_count += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if self.position_size != 0:
            if self.position_type == 'long':
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size * 100000
            else:  # short
                self.unrealized_pnl = (self.entry_price - current_price) * self.position_size * 100000
        else:
            self.unrealized_pnl = 0

        new_equity = max(0, self.balance + self.unrealized_pnl)  # Ensure equity doesn't go negative
        self.max_equity = max(self.max_equity, new_equity)
        self.min_equity = min(self.min_equity, new_equity)
        self.drawdown = (self.max_equity - new_equity) / self.max_equity if self.max_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.drawdown)

        if new_equity <= self.initial_balance * self.margin_call_threshold:
            done = True

        reward = new_equity - prev_equity
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        obs = self.data.iloc[self.current_step]
        return np.array([
            obs['close'],
            obs['sma_10'],
            obs['volume'],
            obs['rsi'],
            (obs['close'] - obs['sma_10']) / obs['sma_10'],
            self.balance,
            self.position_size,
            self.unrealized_pnl,
            self.drawdown,
            self.current_step - self.last_trade_step,
            1 if self.position_type == 'long' else (-1 if self.position_type == 'short' else 0)
        ])

    def render(self):
        equity = self.balance + self.unrealized_pnl
        position_info = f"{self.position_size:.2f} {'long' if self.position_type == 'long' else 'short'}" if self.position_size > 0 else "0.00"
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Equity: {equity:.2f}, Position: {position_info}, "
              f"Unrealized P/L: {self.unrealized_pnl:.2f}, Drawdown: {self.drawdown:.2%}, Max Drawdown: {self.max_drawdown:.2%}")