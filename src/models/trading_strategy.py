import numpy as np

class TradingStrategy:
    def __init__(self, genome):
        self.genome = genome
        
    def decide_action(self, observation):
        close_price, sma_10, volume, rsi, price_sma_diff, balance, position_size, unrealized_pnl, drawdown, time_since_last_trade, position_type = observation
        
        if position_size == 0:  # No position
            if rsi < self.genome['rsi_low'] and price_sma_diff < self.genome['sma_diff_buy']:
                return 0  # Buy Entry
            elif rsi > self.genome['rsi_high'] and price_sma_diff > self.genome['sma_diff_sell']:
                return 2  # Sell Entry
        elif position_size > 0:  # Long position
            if unrealized_pnl / balance > self.genome['take_profit'] or unrealized_pnl / balance < -self.genome['stop_loss']:
                return 1  # Buy Close
            elif rsi > self.genome['rsi_exit_long']:
                return 1  # Buy Close
        elif position_size < 0:  # Short position
            if unrealized_pnl / balance > self.genome['take_profit'] or unrealized_pnl / balance < -self.genome['stop_loss']:
                return 3  # Sell Close
            elif rsi < self.genome['rsi_exit_short']:
                return 3  # Sell Close
        
        return 4  # Hold

    @staticmethod
    def create_random():
        return TradingStrategy({
            'rsi_low': np.random.uniform(20, 40),
            'rsi_high': np.random.uniform(60, 80),
            'sma_diff_buy': np.random.uniform(-0.01, 0),
            'sma_diff_sell': np.random.uniform(0, 0.01),
            'take_profit': np.random.uniform(0.01, 0.05),
            'stop_loss': np.random.uniform(0.01, 0.05),
            'rsi_exit_long': np.random.uniform(50, 70),
            'rsi_exit_short': np.random.uniform(30, 50)
        })

    def mutate(self, mutation_rate=0.2):
        for key in self.genome:
            if np.random.random() < mutation_rate:
                if key.startswith('rsi'):
                    self.genome[key] = np.clip(self.genome[key] * np.random.uniform(0.8, 1.2), 0, 100)
                else:
                    self.genome[key] *= np.random.uniform(0.8, 1.2)

    @staticmethod
    def crossover(parent1, parent2):
        child_genome = {}
        for key in parent1.genome:
            if np.random.random() < 0.5:
                child_genome[key] = parent1.genome[key]
            else:
                child_genome[key] = parent2.genome[key]
        return TradingStrategy(child_genome)