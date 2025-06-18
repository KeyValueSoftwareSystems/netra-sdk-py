"""
Netra SDK LLM integration example

This example demonstrates how to use Netra SDK to trace LLM requests
with automatic instrumentation of popular LLM libraries.

Optional Dependencies:
- openai: For OpenAI API integration
- langchain: For LangChain framework integration

Environment Variables:
- NETRA_API_KEY: Your Netra API key (required)
- NETRA_ENDPOINT: Optional custom endpoint URL (defaults to console output)

Example usage:
    export NETRA_API_KEY='your-api-key'
    python llm_tracing.py
"""

import logging
import os

# Import the Netra SDK
from netra import Netra

# Set up logging
logging.basicConfig(level=logging.INFO)


def main() -> None:
    # Check for required environment variable
    if not os.getenv("NETRA_API_KEY"):
        print("Error: NETRA_API_KEY environment variable is not set")
        print("Please set your API key: export NETRA_API_KEY='your-api-key'")
        return

    # Initialize the Netra SDK
    Netra.init(
        app_name="llm-tracing-example",
        trace_content=True,  # Enable content tracing to see prompts and completions
        # Add any additional configuration here
    )

    print("Netra SDK initialized for LLM tracing")

    # Example with OpenAI (requires openai package)
    try:
        from openai import OpenAI

        # The SDK will automatically instrument OpenAI calls
        client = OpenAI()

        # This request will be automatically traced
        print("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"},
            ],
        )

        print(f"Response received: {response.choices[0].message.content}")
    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")
        print("This was just an example. The SDK works with multiple LLM providers.")

    # Example with LangChain (requires langchain package)
    try:
        from langchain.llms import OpenAI as LangChainOpenAI

        # The SDK will automatically instrument LangChain calls
        llm = LangChainOpenAI(temperature=0)

        # This request will be automatically traced
        print("\nSending request via LangChain...")
        result = llm.predict("What is the capital of Italy?")

        print(f"Response received: {result}")
    except ImportError:
        print("LangChain package not installed. Install with: pip install langchain")
        print("This was just an example. The SDK works with multiple LLM frameworks.")

    print("\nTracing data has been sent to the configured telemetry endpoint.")
    print("Check your monitoring dashboard to view the traces.")


if __name__ == "__main__":
    main()
