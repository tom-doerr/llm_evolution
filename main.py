import litellm

def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"}
    ]
    
    try:
        response = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=messages
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
