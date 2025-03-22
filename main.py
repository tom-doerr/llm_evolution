import litellm
import random
import string
import numpy as np
import concurrent.futures
import time
from typing import List, Tuple, Callable
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

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
            temperature=2.0,  # Higher temperature for more randomness
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
    
    def evolve(self, fitness_func: Callable, get_response_func: Callable = None, num_threads: int = 10, progress=None, task_id=None):
        self.evaluate_population(fitness_func, get_response_func, num_threads, progress, task_id)
        
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
    
    console = Console()
    console.print(Panel("[bold green]Starting evolutionary optimization with LLM-based mutations...[/bold green]"))
    console.print(f"[bold]Using {num_threads} threads for parallel evaluation (max 10 tokens per completion)[/bold]")
    
    # Create a progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False,
    ) as progress:
        # Add tasks for overall progress and current generation
        overall_task = progress.add_task("[cyan]Overall Progress", total=generations)
        gen_task = progress.add_task("[green]Generation 0", total=optimizer.population_size)
        
        # Track best fitness over generations
        best_fitness_history = []
        best_prompts = []
        best_responses = []
        
        for gen in range(1, generations + 1):
            progress.update(gen_task, description=f"[green]Generation {gen}", completed=0)
            
            # Evolve the population
            best = optimizer.evolve(evaluate_response, get_llm_response, num_threads, progress, gen_task)
            best_fitness_history.append(best.fitness)
            best_prompts.append(best.prompt)
            best_responses.append(best.response)
            
            # Update progress
            progress.update(overall_task, advance=1)
            
            # Display generation summary
            table = Table(title=f"Generation {gen} Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Best Fitness", str(best.fitness))
            table.add_row("Best System Prompt", best.prompt[:50] + "..." if len(best.prompt) > 50 else best.prompt)
            table.add_row("Response", best.response[:50] + "..." if len(best.response) > 50 else best.response)
            table.add_row("'a's in first 23 chars", str(best.response[:23].count('a')))
            
            console.print(table)
    
    # Final results
    console.print(Panel("[bold green]Optimization Complete![/bold green]"))
    
    result_table = Table(title="Final Results")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")
    
    best_individual = optimizer.best_individual
    result_table.add_row("Best System Prompt", best_individual.prompt)
    result_table.add_row("Fitness Score", str(best_individual.fitness))
    result_table.add_row("Response Length", str(len(best_individual.response)))
    result_table.add_row("'a's in first 23 chars", str(best_individual.response[:23].count('a')))
    result_table.add_row("Total 'a's", str(best_individual.response.count('a')))
    result_table.add_row("Full Response", best_individual.response)
    
    console.print(result_table)

def main():
    run_optimization(generations=5, num_threads=100)  # Using 100 threads by default

if __name__ == "__main__":
    main()
