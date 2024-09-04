import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_state(self, observation):
        close_price, sma_10, volume, rsi, price_sma_diff, balance, position_size, unrealized_pnl, drawdown, time_since_last_trade, position_type = observation
        
        price_sma_diff_discrete = int(np.clip(price_sma_diff * 100, -10, 10))
        rsi_discrete = int(rsi / 10)
        drawdown_discrete = int(np.clip(drawdown * 100, 0, 20))
        time_since_last_trade_discrete = min(int(time_since_last_trade / 100), 10)
        position_discrete = int(np.clip(position_size * 100, -10, 10))
        
        return (price_sma_diff_discrete, rsi_discrete, drawdown_discrete, time_since_last_trade_discrete, position_discrete)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_key = self.get_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state(state)
        next_state_key = self.get_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        current_q = self.q_table[state_key][action]
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()