import numpy as np

class ImprovedTradeAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.position_size = 0
        self.position_type = None
        self.entry_price = None
        self.volume_history = []

    def choose_action(self, observation):
        close_price, sma_10, volume, rsi, price_sma_diff, balance, current_position, unrealized_pnl = observation

        # Update volume history
        self.volume_history.append(volume)
        if len(self.volume_history) > 10:
            self.volume_history.pop(0)

        # Trend following
        uptrend = price_sma_diff > 0.0005
        downtrend = price_sma_diff < -0.0005

        # Oversold/Overbought conditions
        oversold = rsi < 40
        overbought = rsi > 60

        # Volume spike
        volume_spike = volume > np.mean(self.volume_history) * 1.2 if len(self.volume_history) > 0 else False

        # Decision making
        if current_position == 0:  # No position
            if uptrend and oversold and balance > 12.80:  # Ensure enough balance for margin
                return 0  # Buy Entry
            elif downtrend and overbought and balance > 12.80:
                return 2  # Sell Entry
        elif self.position_type == 'long':  # Long position
            profit_pips = unrealized_pnl / 1.70  # Convert unrealized P/L to pips
            if profit_pips > 10 or profit_pips < -5 or downtrend:
                return 1  # Buy Close
        elif self.position_type == 'short':  # Short position
            profit_pips = unrealized_pnl / 1.70  # Convert unrealized P/L to pips
            if profit_pips > 10 or profit_pips < -5 or uptrend:
                return 3  # Sell Close

        return 4  # Hold

    def log_trade(self, action, price, balance, position, position_type, entry_price, unrealized_pnl):
        action_name = ["Buy Entry", "Buy Close", "Sell Entry", "Sell Close", "Hold"][action]
        if action in [0, 2]:  # Entry actions
            entry_exit_price = f"Entry Price: {price:.3f}"
        else:  # Exit actions
            entry_exit_price = f"Exit Price: {price:.3f}"
        entry_price_str = f"{entry_price:.3f}" if entry_price is not None else "None"
        position_info = f"{position:.2f} {position_type}" if position > 0 else "0.00"
        equity = balance + unrealized_pnl
        print(f"Action: {action_name}, Price: {price:.3f}, Balance: {balance:.2f}, Equity: {equity:.2f}, "
              f"Position: {position_info}, {entry_exit_price}, Entry Price: {entry_price_str}, "
              f"Unrealized P/L: {unrealized_pnl:.2f}")