import pandas as pd
import numpy as np
import os
import sys
import time

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.trading_environment import TradingEnvironment
from src.models.genetic_algorithm import GeneticAlgorithm
from src.models.trading_strategy import TradingStrategy
from src.data.data_loader import load_data, preprocess_data

def run_genetic_algorithm(env, generations=100, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
    print("Starting genetic algorithm...")
    start_time = time.time()
    ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate)
    best_strategy, best_fitness = ga.evolve(env, generations)
    end_time = time.time()
    print(f"Genetic algorithm completed in {end_time - start_time:.2f} seconds")
    return best_strategy, best_fitness

def test_strategy(env, strategy):
    print("Testing best strategy...")
    state = env.reset()
    done = False
    while not done:
        action = strategy.decide_action(state)
        state, reward, done, _ = env.step(action)
    final_equity = env.balance + env.unrealized_pnl
    max_drawdown = env.max_drawdown
    trades_count = env.trades_count
    return final_equity, max_drawdown, trades_count

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    start_time = time.time()
    file_path = os.path.join(project_root, 'data', 'raw', 'USDJPY_1m_data.csv')
    raw_data = load_data(file_path)
    processed_data = preprocess_data(raw_data)
    end_time = time.time()
    print(f"Data loading and preprocessing completed in {end_time - start_time:.2f} seconds")

    print("Creating trading environment...")
    env = TradingEnvironment(processed_data, initial_balance=100, leverage=786, max_position_size=0.1)

    best_strategy, best_fitness = run_genetic_algorithm(env, generations=100, population_size=100, mutation_rate=0.2, crossover_rate=0.8)
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print("Best strategy genome:", best_strategy.genome)

    final_equity, max_drawdown, trades_count = test_strategy(env, best_strategy)
    print(f"Test result: Final Equity = {final_equity:.2f}, Max Drawdown = {max_drawdown:.2%}, Trades = {trades_count}")