import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import cohere
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from netra import Netra
from netra.decorators import workflow
from netra.pii import get_default_detector

# --- Configuration ---

# Load environment variables from a .env file
load_dotenv()

# Initialize console for rich text output to the user
console = Console()

# Configure logging for internal status and error messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# --- SDK Initialization ---


def initialize_sdks() -> Any:
    """
    Initializes the Netra and Cohere SDKs.

    Returns:
        An initialized Cohere AsyncClient instance, or None if initialization fails.
    """
    # 1. Initialize Netra SDK
    try:
        netra_api_key = os.environ["NETRA_API_KEY"]
        Netra.init(
            app_name="cohere-cli-example",
            disable_batch=True,
            environment="dev",
            headers=f"x-api-key={netra_api_key}",
        )
        # Set context for the session for filtering in Netra's dashboard
        Netra.set_session_id("test-session")
        Netra.set_tenant_id("test-tenant")
        Netra.set_user_id("test-user")
        logging.info("Netra SDK initialized successfully.")
    except KeyError:
        logging.error("NETRA_API_KEY not found. Please check your .env file.")
        console.print("[bold red]Error: NETRA_API_KEY not found in environment variables.[/bold red]")
        return None

    # 2. Initialize Cohere Client
    try:
        cohere_api_key = os.environ["COHERE_API_KEY"]
        co = cohere.AsyncClient(api_key=cohere_api_key)  # type: ignore[attr-defined]
        logging.info("Cohere client initialized successfully.")
        return co
    except KeyError:
        logging.error("COHERE_API_KEY not found. Please check your .env file.")
        console.print("[bold red]Error: COHERE_API_KEY not found in environment variables.[/bold red]")
        return None


# --- Core Chat Logic ---


@workflow(name="cohere_chat_workflow")  # type: ignore[arg-type]
async def get_cohere_response_with_pii_protection(
    client: Any, messages: List[Dict[str, str]], model: str
) -> Optional[Any]:
    """
    Sends messages to the Cohere chat API with PII protection.

    This function is wrapped with Netra's `@workflow` decorator. It scans the
    latest user message for PII and masks it before sending the payload to Cohere.

    Args:
        client: The initialized Cohere AsyncClient.
        messages: A list of message objects for the chat history.
        model: The name of the Cohere model to use.

    Returns:
        The response object from Cohere, or None if an error occurs.
    """
    if not messages:
        logging.warning("Message list is empty. Nothing to send.")
        return None

    # Separate the latest message from the previous chat history
    latest_message_content = messages[-1].get("message", "")
    history_for_api = messages[:-1]

    logging.info("Scanning latest user message for PII.")

    # 1. PII Detection and Masking
    pii_detector = get_default_detector(action_type="MASK")
    pii_result = pii_detector.detect(latest_message_content)

    if pii_result.has_pii:
        logging.warning("PII detected. Using masked text for the API call.")
        message_to_send = pii_result.masked_text
        logging.info("Masked input: '%s'", message_to_send)
    else:
        logging.info("No PII detected in the latest message.")
        message_to_send = latest_message_content

    # 2. Call Cohere API
    try:
        logging.info(f"Sending request to Cohere model: {model}")
        response = await client.chat(
            model=model,
            message=message_to_send,
            chat_history=history_for_api,
        )
        logging.info("Successfully received response from Cohere.")
        return response
    except Exception as e:
        logging.error("An unexpected error occurred during chat: %s", e)
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return None


# --- Main Application Block ---


async def main() -> None:
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Simple Cohere CLI with Netra SDK integration")
    parser.add_argument("--model", default="command-a-03-2025", help="Model to use (default: command-a-03-2025)")
    args = parser.parse_args()

    # Initialize SDKs and exit if keys are missing
    cohere_client = initialize_sdks()
    if not cohere_client:
        sys.exit(1)

    # Start interactive chat session
    console.print(f"[bold blue]Cohere Chat CLI (Model: {args.model})[/bold blue]")
    console.print("Type your message and press Enter. Type 'exit' or 'quit' to end.")

    chat_history = []

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]")

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold blue]Ending chat session.[/bold blue]")
            break

        # Append user message to chat history
        chat_history.append({"role": "USER", "message": user_input})

        # Get response from Cohere
        with console.status("[bold cyan]Cohere is thinking...[/bold cyan]"):
            response = await get_cohere_response_with_pii_protection(
                client=cohere_client, messages=chat_history, model=args.model
            )  # type: ignore[misc]

        if response and response.text:
            # Display the response using rich Markdown
            console.print("[bold blue]Cohere:[/bold blue]")
            console.print(Markdown(response.text))
            # Add Cohere's response to the history for context in the next turn
            chat_history.append({"role": "CHATBOT", "message": response.text})
        else:
            # If the response failed, remove the last user message to allow a retry
            chat_history.pop()
            console.print("[bold yellow]Failed to get a response. Please try again.[/bold yellow]")


if __name__ == "__main__":
    # Example Usage from command line:
    # python your_script_name.py
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold blue]Program terminated by user.[/bold blue]")
        sys.exit(0)
