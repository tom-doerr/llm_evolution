import litellm
from typing import List, Tuple, Optional

def format_history(history: List[Tuple[str, int, int]]) -> str:
    """Format history into human-readable string"""
    if not history:
        return "No previous attempts"
        
    return "\n".join(
        f"Attempt {i+1}: {resp[:20]}... (Reward: {reward}, 'a's: {a_count})"
        for i, (resp, reward, a_count) in enumerate(history)
    )

def evaluate_response(response: str) -> Tuple[int, int]:
    """
    Calculate reward for response based on hidden criteria
    Returns tuple of (reward, a_count)
    """
    a_count = 0
    reward = 0
    # Convert to bytes and check each byte position
    for i, byte in enumerate(response.encode('utf-8')):
        if i < 23 and byte == ord('a'):
            a_count += 1
            reward += 1
        elif i >= 23:
            reward -= 1
    return reward, a_count

def reasoner_experiment(iterations: int = 10):
    """
    Run iterative prompting experiment with deepseek-reasoner
    """
    system_msg = """You are an AI that generates text. Your goal is to maximize your reward score.
    
    Rules:
    1. You will see your full history of attempts and rewards
    2. Each 'a' in first 23 characters gives +1 reward
    3. Any character after 23 gives -1 reward
    4. Responses limited to 10 tokens
    
    Analyze the patterns in your history and improve systematically.
    Respond ONLY with your new text attempt."""
    
    history = []
    current_prompt = "Start with random text"
    
    for i in range(iterations):
        try:
            # Get model completion
            response = litellm.completion(
                model="deepseek/deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Full History:\n{format_history(history)}\n\nNew attempt:"}
                ],
                temperature=0.7,
                max_tokens=10
            ).choices[0].message.content
            
            # Calculate reward
            reward, a_count = evaluate_response(response)
            history.append((response, reward, a_count))
            
            # Print results
            print(f"\n=== Iteration {i+1} ===")
            print(f"Response: {response}")
            print(f"Reward: {reward} (a's: {a_count})")
            
            # Update prompt for next iteration
            current_prompt = response
            
        except Exception as e:
            print(f"Error: {e}")
            return history
    
    return history

if __name__ == "__main__":
    reasoner_experiment()
