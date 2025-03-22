import litellm
import random
import string
import numpy as np
import concurrent.futures
from typing import List, Tuple, Callable

def evaluate_response(response: str) -> float:
    """
    Evaluate an LLM response based on hidden criteria.
    Returns a score.
    """
    score = 0
    for i, char in enumerate(response):
        if i < 23 and char == 'a':
            score += 1
        elif i >= 23:
            score -= 1
    
    return score

def get_llm_response(system_prompt: str) -> str:
    """
    Get a response from the LLM using the provided system prompt and an empty user message.
    Limited to 10 tokens.
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""}
        ]
        
        response = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=messages,
            max_tokens=10  # Limit to 10 tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""  # Return empty string on error

def get_llm_mutation(system_prompt: str) -> str:
    """
    Get a mutation from the LLM by using its response as the new prompt.
    Limited to 10 tokens.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates random text. Your response will be used as a system prompt."},
            {"role": "user", "content": f"Generate a new system prompt based on this one: {system_prompt}"}
        ]
        
        response = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=messages,
            max_tokens=10  # Limit to 10 tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM for mutation: {e}")
        return system_prompt  # Return original prompt on error

class Individual:
    def __init__(self, prompt: str = None, length: int = 30):
        if prompt is None:
            self.prompt = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        else:
            self.prompt = prompt
        self.fitness = None
    
    def evaluate(self, fitness_func: Callable, get_response_func: Callable = None):
        if get_response_func:
            response = get_response_func(self.prompt)
            self.response = response
            self.fitness = fitness_func(response)
        else:
            self.fitness = fitness_func(self.prompt)
        return self.fitness
    
    def mutate(self, mutation_rate: float = 0.1, mutation_func: Callable = None):
        if mutation_func and random.random() < mutation_rate:
            # Use LLM to generate a mutation
            self.prompt = mutation_func(self.prompt)
        else:
            # Fallback to random character mutation
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
        
    def evaluate_population(self, fitness_func: Callable, get_response_func: Callable = None, num_threads: int = 10):
        # Evaluate individuals in parallel
        unevaluated = [ind for ind in self.population if ind.fitness is None]
        
        if unevaluated and get_response_func:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Create a list of futures
                futures = {executor.submit(ind.evaluate, fitness_func, get_response_func): ind for ind in unevaluated}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Get the result (or exception)
                    except Exception as e:
                        print(f"Error evaluating individual: {e}")
        else:
            # Fallback to sequential evaluation
            for individual in unevaluated:
                individual.evaluate(fitness_func, get_response_func)
        
        self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'), reverse=True)
        self.best_individual = self.population[0]
        
    def select_parent(self) -> Individual:
        # Tournament selection
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]
    
    def evolve(self, fitness_func: Callable, get_response_func: Callable = None, num_threads: int = 10):
        self.evaluate_population(fitness_func, get_response_func, num_threads)
        
        new_population = []
        
        # Elitism - keep the best individuals
        new_population.extend(self.population[:self.elitism_count])
        
        # Create the rest of the population through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = parent1.crossover(parent2)
                
                child1.mutate(self.mutation_rate, get_llm_mutation)
                child2.mutate(self.mutation_rate, get_llm_mutation)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # Just mutation
                parent = self.select_parent()
                child = Individual(parent.prompt).mutate(self.mutation_rate, get_llm_mutation)
                new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        self.evaluate_population(fitness_func)
        
        return self.best_individual

def run_optimization(generations: int = 50, num_threads: int = 100):
    optimizer = EvolutionaryOptimizer(
        population_size=20,  # Further reduced population size due to more API calls
        mutation_rate=0.8,   # Higher mutation rate to encourage LLM mutations
        crossover_rate=0.5,  # Lower crossover rate to favor mutations
        elitism_count=2,     # Keep fewer elites
        prompt_length=30     # System prompts can be a bit longer
    )
    
    print("Starting evolutionary optimization with LLM-based mutations...")
    print(f"Using {num_threads} threads for parallel evaluation (max 10 tokens per completion)")
    print("Generation 0: Initializing population with random prompts")
    
    for gen in range(1, generations + 1):
        best = optimizer.evolve(evaluate_response, get_llm_response, num_threads)
        print(f"Generation {gen}: Best fitness = {best.fitness}")
        print(f"Best system prompt: '{best.prompt[:50]}...' (truncated)")
        print(f"Response: '{best.response[:50]}...' (truncated)")
        print("-" * 50)
    
    print("\nOptimization complete!")
    print(f"Best system prompt found: '{optimizer.best_individual.prompt}'")
    print(f"Fitness score: {optimizer.best_individual.fitness}")
    
    # Analyze the best response
    print("\nAnalysis of best response:")
    print(f"Response length: {len(optimizer.best_individual.response)}")
    print(f"Number of 'a's in first 23 chars: {optimizer.best_individual.response[:23].count('a')}")
    print(f"Total number of 'a's: {optimizer.best_individual.response.count('a')}")
    
    # Print the full response
    print("\nFull response to best system prompt:")
    print(optimizer.best_individual.response)

def main():
    run_optimization(generations=5, num_threads=100)  # Using 100 threads by default

if __name__ == "__main__":
    main()
