import pytest
from main import Individual, EvolutionaryOptimizer, evaluate_response

@pytest.fixture
def sample_individual():
    return Individual(prompt="test_prompt")

@pytest.fixture
def sample_optimizer():
    return EvolutionaryOptimizer(
        population_size=20,
        mutation_rate=0.0,
        crossover_rate=0.0,
        elitism_count=2,
        prompt_length=30
    )

def test_individual_initialization(sample_individual):
    assert sample_individual.prompt == "test_prompt"
    assert sample_individual.fitness is None
    assert sample_individual.response == ""

def test_individual_evaluate(sample_individual):
    # Test evaluation without response function
    sample_individual.evaluate(lambda x: 5)
    assert sample_individual.fitness == 5

def test_individual_mutate(sample_individual):
    original_prompt = sample_individual.prompt
    sample_individual.mutate(mutation_rate=1.0)  # 100% mutation rate
    assert sample_individual.prompt != original_prompt
    assert len(sample_individual.prompt) == len(original_prompt)

def test_individual_crossover(sample_individual):
    # Use same-length prompts for crossover test
    parent2 = Individual(prompt="prompt456")
    sample_individual.prompt = "prompt123"  # Set fixed length
    child1, child2 = sample_individual.crossover(parent2)
    assert len(child1.prompt) == len(sample_individual.prompt)
    assert len(child2.prompt) == len(sample_individual.prompt)

def test_evaluate_response():
    # Test basic scoring
    assert evaluate_response("a" * 23) == 23  # All 'a's in first 23 bytes
    assert evaluate_response("a" * 30) == 23 - 7  # 23 'a's - 7 extra bytes
    assert evaluate_response("") == 0  # Empty response
    
    # Test multi-byte characters (30 'å' chars = 60 bytes)
    assert evaluate_response("å" * 30) == -37  # 60 total bytes - 23 = 37 penalty

def test_optimizer_initialization(sample_optimizer):
    assert len(sample_optimizer.population) == 20
    assert all(isinstance(ind, Individual) for ind in sample_optimizer.population)

def test_optimizer_evolve(sample_optimizer):
    initial_population = sample_optimizer.population.copy()
    sample_optimizer.evolve(evaluate_response, lambda x: "new_prompt")
    
    # Check population size remains consistent
    assert len(sample_optimizer.population) == 20
    
    # Check elitism
    assert sample_optimizer.population[0].prompt in [ind.prompt for ind in initial_population]
    
    # Check new prompts from responses
    assert any(ind.prompt == "new_prompt" for ind in sample_optimizer.population)
