import litellm
import random
import string
import numpy as np
from typing import List, Tuple, Callable

def evaluate_prompt(prompt: str) -> float:
    """
    Evaluate a prompt based on hidden criteria.
    Returns a score.
    """
    score = 0
    for i, char in enumerate(prompt):
        if i < 23 and char == 'a':
            score += 1
        elif i >= 23:
            score -= 1
    
    return score

class Individual:
    def __init__(self, prompt: str = None, length: int = 30):
        if prompt is None:
            self.prompt = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        else:
            self.prompt = prompt
        self.fitness = None
    
    def evaluate(self, fitness_func: Callable):
        self.fitness = fitness_func(self.prompt)
        return self.fitness
    
    def mutate(self, mutation_rate: float = 0.1):
        chars = list(self.prompt)
        for i in range(len(chars)):
            if random.random() < mutation_rate:
                chars[i] = random.choice(string.ascii_lowercase)
        self.prompt = ''.join(chars)
        self.fitness = None  # Reset fitness after mutation
        return self

    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        if len(self.prompt) != len(other.prompt):
            raise ValueError("Prompts must be of the same length")
        
        crossover_point = random.randint(1, len(self.prompt) - 1)
        child1_prompt = self.prompt[:crossover_point] + other.prompt[crossover_point:]
        child2_prompt = other.prompt[:crossover_point] + self.prompt[crossover_point:]
        
        return Individual(child1_prompt), Individual(child2_prompt)

class EvolutionaryOptimizer:
    def __init__(self, 
                 population_size: int = 100, 
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 5,
                 prompt_length: int = 30):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.prompt_length = prompt_length
        self.population = [Individual(length=prompt_length) for _ in range(population_size)]
        self.best_individual = None
        self.generation = 0
        
    def evaluate_population(self, fitness_func: Callable):
        for individual in self.population:
            if individual.fitness is None:
                individual.evaluate(fitness_func)
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0]
        
    def select_parent(self) -> Individual:
        # Tournament selection
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def evolve(self, fitness_func: Callable):
        self.evaluate_population(fitness_func)
        
        new_population = []
        
        # Elitism - keep the best individuals
        new_population.extend(self.population[:self.elitism_count])
        
        # Create the rest of the population through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = parent1.crossover(parent2)
                
                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # Just mutation
                parent = self.select_parent()
                child = Individual(parent.prompt).mutate(self.mutation_rate)
                new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        self.evaluate_population(fitness_func)
        
        return self.best_individual

def run_optimization(generations: int = 50):
    optimizer = EvolutionaryOptimizer(
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_count=5,
        prompt_length=30
    )
    
    print("Starting evolutionary optimization...")
    print("Generation 0: Initializing population")
    
    for gen in range(1, generations + 1):
        best = optimizer.evolve(evaluate_prompt)
        print(f"Generation {gen}: Best fitness = {best.fitness}, Prompt = '{best.prompt}'")
    
    print("\nOptimization complete!")
    print(f"Best prompt found: '{optimizer.best_individual.prompt}'")
    print(f"Fitness score: {optimizer.best_individual.fitness}")
    
    # Analyze the best prompt
    print("\nAnalysis of best prompt:")
    print(f"Length: {len(optimizer.best_individual.prompt)}")
    print(f"Number of 'a's: {optimizer.best_individual.prompt.count('a')}")
    
    # Try to use the best prompt with the LLM
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": optimizer.best_individual.prompt}
        ]
        
        response = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=messages
        )
        print("\nLLM Response to best prompt:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling LLM: {e}")

def main():
    run_optimization(generations=50)

if __name__ == "__main__":
    main()
