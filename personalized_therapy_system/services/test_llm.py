import asyncio
from openrouter_llm import OpenRouterLLM

async def main():
    # Replace with your valid OpenRouter API key
    api_key ="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"

    # Initialize your LLM
    llm = OpenRouterLLM(
        api_key=api_key,
        model_name="meta-llama/llama-3.2-3b-instruct:free",
        streaming=False  # Set to True if you want streaming
    )

    # Test a simple prompt
    prompt = "Hello! Can you explain the meaning of life in a sentence?"
    # Non-streaming call
    result = llm._generate(prompt)
    print("Response:", result.generations[0].message.content)

    # Streaming call
    print("\nStreaming response:")
    async for token in llm.stream(prompt):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
