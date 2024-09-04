import numpy as np
from typing import List, Tuple
from .trading_strategy import TradingStrategy
import time
import logging
from tqdm import tqdm
import multiprocessing
import pickle
import os

class GeneticAlgorithm:
    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float):
        self.population_size = population_size
        self.initial_mutation_rate = mutation_rate
        self.initial_crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[TradingStrategy] = []
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('GeneticAlgorithm')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('ga_log.txt')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def initialize_population(self):
        self.logger.info("Initializing population...")
        self.population = [TradingStrategy.create_random() for _ in range(self.population_size)]
        self.logger.info("Population initialized.")

    def evaluate_fitness(self, env) -> List[Tuple[float, float]]:
        self.logger.info("Evaluating fitness...")
        with multiprocessing.Pool() as pool:
            results = pool.starmap(self._evaluate_strategy, [(strategy, env) for strategy in self.population])
        
        fitnesses_and_balances = [(fitness, balance) for fitness, balance in results]
        
        fitnesses, balances = zip(*fitnesses_and_balances)
        self.logger.info(f"Fitness range: Min = {min(fitnesses):.2f}, Max = {max(fitnesses):.2f}")
        self.logger.info(f"Balance range: Min = {min(balances):.2f}, Max = {max(balances):.2f}")
        return fitnesses_and_balances

    @staticmethod
    def _evaluate_strategy(strategy, env):
        state = env.reset()
        done = False
        while not done:
            action = strategy.decide_action(state)
            state, reward, done, _ = env.step(action)
        final_equity = env.balance + env.unrealized_pnl
        max_drawdown = env.max_drawdown
        trades_count = env.trades_count
        
        returns = (final_equity - env.initial_balance) / env.initial_balance
        sharpe_ratio = returns / (max_drawdown + 1e-6)
        trading_penalty = max(0, trades_count - 100) * 0.01
        
        fitness = final_equity * (1 + sharpe_ratio) - trading_penalty
        return fitness, final_equity

    def select_parents(self, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probs = [f / total_fitness for f in fitnesses]
        return np.random.choice(self.population, size=2, p=selection_probs, replace=False)

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            return TradingStrategy.crossover(parent1, parent2)
        else:
            return parent1 if np.random.random() < 0.5 else parent2

    def mutate(self, strategy):
        strategy.mutate(self.mutation_rate)

    def adjust_parameters(self, generation, max_generations):
        # Linearly decrease mutation rate and increase crossover rate
        progress = generation / max_generations
        self.mutation_rate = self.initial_mutation_rate * (1 - progress)
        self.crossover_rate = self.initial_crossover_rate + (1 - self.initial_crossover_rate) * progress
        
        self.logger.info(f"Adjusted parameters: Mutation Rate = {self.mutation_rate:.4f}, Crossover Rate = {self.crossover_rate:.4f}")

    def evolve(self, env, generations, early_stopping_generations=10):
        self.initialize_population()
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in tqdm(range(generations), desc="Generations"):
            start_time = time.time()
            fitnesses_and_balances = self.evaluate_fitness(env)
            fitnesses, balances = zip(*fitnesses_and_balances)
            new_population = []
            
            best_index = np.argmax(fitnesses)
            current_best_strategy = self.population[best_index]
            current_best_fitness = fitnesses[best_index]
            current_best_balance = balances[best_index]
            avg_fitness = np.mean(fitnesses)
            avg_balance = np.mean(balances)
            
            self.logger.info(f"Generation {generation + 1}: Best Fitness = {current_best_fitness:.2f}, "
                             f"Best Balance = {current_best_balance:.2f}, "
                             f"Avg Fitness = {avg_fitness:.2f}, "
                             f"Avg Balance = {avg_balance:.2f}")
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_strategy = current_best_strategy
                generations_without_improvement = 0
                
                # Save the new best strategy
                self._save_strategy(best_strategy, generation, current_best_fitness)
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= early_stopping_generations:
                self.logger.info(f"Early stopping triggered after {generation + 1} generations")
                break
            
            new_population.append(best_strategy)  # Elitism
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(fitnesses)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 10 == 0:
                num_random = max(1, self.population_size // 10)
                for _ in range(num_random):
                    self.population[np.random.randint(1, self.population_size)] = TradingStrategy.create_random()
            
            self.adjust_parameters(generation, generations)
            
            end_time = time.time()
            self.logger.info(f"Generation {generation + 1} completed in {end_time - start_time:.2f} seconds")
        
        return best_strategy, best_fitness

    def _save_strategy(self, strategy, generation, fitness):
        if not os.path.exists('best_strategies'):
            os.makedirs('best_strategies')
        
        filename = f'best_strategies/strategy_gen_{generation}_fitness_{fitness:.2f}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(strategy, f)
        self.logger.info(f"Saved new best strategy: {filename}")