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
    Returns a score based on bytes (+1 for 'a' bytes before 23, -1 for any byte after 23)
    """
    score = 0
    # Convert to bytes and check each byte position
    for i, byte in enumerate(response.encode('utf-8')):
        if i < 23 and byte == ord('a'):
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
        self.response = ""  # Initialize response attribute
    
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
        
    def evaluate_population(self, fitness_func: Callable, get_response_func: Callable = None, num_threads: int = 10, progress=None, task_id=None):
        # Evaluate individuals in parallel
        unevaluated = [ind for ind in self.population if ind.fitness is None]
        total = len(unevaluated)
        completed = 0
        
        console = Console()
        
        if unevaluated and get_response_func:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Create a list of futures
                futures = {executor.submit(ind.evaluate, fitness_func, get_response_func): ind for ind in unevaluated}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Get the result (or exception)
                        completed += 1
                        if progress and task_id:
                            progress.update(task_id, completed=completed, total=total)
                            ind = futures[future]
                            if hasattr(ind, 'response'):
                                progress.console.print(f"[cyan]Prompt:[/cyan] {ind.prompt[:30]}...")
                                progress.console.print(f"[green]Response:[/green] {ind.response}")
                                progress.console.print(f"[yellow]Fitness:[/yellow] {ind.fitness}")
                                progress.console.print("---")
                    except Exception as e:
                        console.print(f"[red]Error evaluating individual:[/red] {e}")
        else:
            # Fallback to sequential evaluation
            for individual in unevaluated:
                individual.evaluate(fitness_func, get_response_func)
                completed += 1
                if progress and task_id:
                    progress.update(task_id, completed=completed, total=total)
        
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
        
        # Add LLM responses as new prompts to the population
        for individual in self.population:
            if hasattr(individual, 'response') and individual.response:
                # Create a new individual using the LLM's response as the prompt
                new_individual = Individual(individual.response)
                new_population.append(new_individual)
                
                # Break if we've reached the population size
                if len(new_population) >= self.population_size:
                    break
        
        # If we still need more individuals, add random ones
        while len(new_population) < self.population_size:
            new_individual = Individual(length=self.prompt_length)
            new_population.append(new_individual)
        
        self.population = new_population
        self.generation += 1
        # No need to evaluate the population again here
        # The evaluation will happen at the beginning of the next generation
        
        return self.best_individual

def run_optimization(generations: int = 50, num_threads: int = 100):
    optimizer = EvolutionaryOptimizer(
        population_size=20,  # Reduced population size
        mutation_rate=0.0,   # No mutation needed as we use responses directly
        crossover_rate=0.0,  # No crossover needed
        elitism_count=2,     # Keep a few elites
        prompt_length=30     # System prompts can be a bit longer
    )
    
    console = Console()
    console.print(Panel("[bold green]Starting evolutionary optimization using LLM responses as new prompts...[/bold green]"))
    console.print(f"[bold]Using {num_threads} threads for parallel evaluation (max 10 tokens per completion)[/bold]")
    console.print("[bold yellow]Each LLM response is added directly to the population as a new prompt[/bold yellow]")
    
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
            table.add_row("'a's (Reward)", f"{best.response[:23].count('a')} (+{best.response[:23].count('a')})")
            
            console.print(table)
            
            # Population statistics
            pop_stats_table = Table(title=f"Population Statistics (Gen {gen})")
            pop_stats_table.add_column("Statistic", style="cyan")
            pop_stats_table.add_column("Value", style="green")
            
            # Calculate population statistics
            fitnesses = [ind.fitness for ind in optimizer.population if ind.fitness is not None]
            avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
            median_fitness = sorted(fitnesses)[len(fitnesses)//2] if fitnesses else 0
            a_counts = [ind.response[:23].count('a') for ind in optimizer.population if hasattr(ind, 'response')]
            avg_a_count = sum(a_counts) / len(a_counts) if a_counts else 0
            
            pop_stats_table.add_row("Average Fitness", f"{avg_fitness:.2f}")
            pop_stats_table.add_row("Median Fitness", f"{median_fitness:.2f}")
            pop_stats_table.add_row("Fitness Range", f"{min(fitnesses) if fitnesses else 0} to {max(fitnesses) if fitnesses else 0}")
            pop_stats_table.add_row("Average 'a's in first 23 chars", f"{avg_a_count:.2f}")
            pop_stats_table.add_row("Population Size", str(len(optimizer.population)))
            
            console.print(pop_stats_table)
            
            # Show top 3 individuals
            top_table = Table(title=f"Top 3 Individuals (Gen {gen})")
            top_table.add_column("Rank", style="cyan")
            top_table.add_column("Fitness", style="green")
            top_table.add_column("System Prompt", style="yellow")
            top_table.add_column("Response", style="magenta")
            top_table.add_column("'a's in first 23", style="blue")
            
            for i, ind in enumerate(optimizer.population[:3]):
                top_table.add_row(
                    str(i+1),
                    str(ind.fitness),
                    ind.prompt[:30] + "..." if len(ind.prompt) > 30 else ind.prompt,
                    ind.response[:30] + "..." if len(ind.response) > 30 else ind.response,
                    str(ind.response[:23].count('a'))
                )
            
            console.print(top_table)
            console.print("\n" + "="*80 + "\n")
    
    # Final results
    console.print(Panel("[bold green]Optimization Complete![/bold green]"))
    
    # Evolution summary
    evolution_table = Table(title="Evolution Summary")
    evolution_table.add_column("Generation", style="cyan")
    evolution_table.add_column("Best Fitness", style="green")
    evolution_table.add_column("'a's in first 23", style="yellow")
    evolution_table.add_column("Response Preview", style="magenta")
    
    for i, (fitness, response) in enumerate(zip(best_fitness_history, best_responses)):
        evolution_table.add_row(
            str(i+1),
            str(fitness),
            str(response[:23].count('a')),
            response[:20] + "..." if len(response) > 20 else response
        )
    
    console.print(evolution_table)
    
    # Detailed final results
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
    
    # Analysis of evolution
    analysis_table = Table(title="Evolution Analysis")
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="green")
    
    # Calculate improvement
    if len(best_fitness_history) > 1:
        first_gen = best_fitness_history[0]
        last_gen = best_fitness_history[-1]
        improvement = last_gen - first_gen
        percent_improvement = (improvement / abs(first_gen)) * 100 if first_gen != 0 else float('inf')
        
        analysis_table.add_row("Initial Best Fitness", str(first_gen))
        analysis_table.add_row("Final Best Fitness", str(last_gen))
        analysis_table.add_row("Absolute Improvement", str(improvement))
        analysis_table.add_row("Percent Improvement", f"{percent_improvement:.2f}%")
        
        # Find generation with biggest improvement
        improvements = [best_fitness_history[i] - best_fitness_history[i-1] for i in range(1, len(best_fitness_history))]
        if improvements:
            max_improvement_gen = improvements.index(max(improvements)) + 1
            max_improvement = max(improvements)
            analysis_table.add_row("Biggest Improvement", f"{max_improvement} (Gen {max_improvement_gen} to {max_improvement_gen+1})")
    
    console.print(analysis_table)
    
    # Print optimization parameters
    params_table = Table(title="Optimization Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")
    
    params_table.add_row("Population Size", str(optimizer.population_size))
    params_table.add_row("Mutation Rate", "N/A (Using responses as new prompts)")
    params_table.add_row("Crossover Rate", "N/A (No crossover used)")
    params_table.add_row("Elitism Count", str(optimizer.elitism_count))
    params_table.add_row("Prompt Length", str(optimizer.prompt_length))
    params_table.add_row("Number of Threads", str(num_threads))
    params_table.add_row("Generations", str(generations))
    
    console.print(params_table)

def main():
    run_optimization(generations=50, num_threads=100)  # Using 100 threads by default

if __name__ == "__main__":
    main()
