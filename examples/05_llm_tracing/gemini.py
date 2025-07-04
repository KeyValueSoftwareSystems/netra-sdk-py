import argparse
import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai as google_genai
from google.api_core import exceptions
from google.genai import types as google_types

from netra import Netra
from netra.decorators import workflow
from netra.pii import get_default_detector

# --- Configuration ---

# Load environment variables from a .env file for security
load_dotenv()

# Configure logging to provide clear output to the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# --- Netra SDK Initialization ---


def initialize_netra_sdk() -> None:
    """
    Initializes the Netra SDK with configuration details.

    This setup is crucial for enabling Netra's monitoring and PII protection
    features within the application.
    """
    try:
        netra_api_key = os.environ["NETRA_API_KEY"]
        Netra.init(
            app_name="gemini-translator-example",
            disable_batch=False,
            environment="dev",
            headers=f"x-api-key={netra_api_key}",
        )
        # Set context for the session, which can be used for filtering in Netra's dashboard
        Netra.set_session_id("test-session")
        Netra.set_tenant_id("test-tenant")
        Netra.set_user_id("test-user")
        logging.info("Netra SDK initialized successfully.")
    except KeyError:
        logging.error("NETRA_API_KEY not found in environment variables. Please check your .env file.")
        raise


# --- Core Translation Logic ---


@workflow(name="translator_workflow")  # type: ignore[arg-type]
async def translate_text_with_pii_protection(text_to_translate: str) -> Any:
    """
    Translates English text to French using Gemini, with PII protection.

    This function is wrapped with the Netra `@workflow` decorator, which allows
    Netra to monitor its execution. Before translation, it scans the input
    for PII. If PII is found, it's masked before being sent to the Gemini API.

    Args:
        text_to_translate: The string of English text to be translated.

    Returns:
        The translated French text as a string, or None if an error occurs.
    """
    if not text_to_translate:
        logging.warning("Input text is empty. Nothing to translate.")
        return None

    logging.info("Starting translation workflow for: '%s'", text_to_translate)

    # 1. PII Detection using Netra's default PII detector
    logging.info("Scanning for PII in the input text.")
    pii_detector = get_default_detector(action_type="MASK")
    pii_result = pii_detector.detect(text_to_translate)

    if pii_result.has_pii:
        logging.warning("PII detected. Using masked text for translation.")
        # Use the text with PII masked (e.g., "My name is [PERSON_0]")
        input_for_model = pii_result.masked_text
        logging.info("Masked input: '%s'", input_for_model)
    else:
        logging.info("No PII detected. Using original text.")
        input_for_model = text_to_translate

    # 2. Translation using Google Gemini API
    try:
        logging.info("Sending request to Gemini API.")
        google_api_key = os.environ["GOOGLE_API_KEY"]
        client = google_genai.Client(api_key=google_api_key)

        # Define the model's instructions and configuration
        system_instruction = [
            "You are a helpful language translator.",
            "Your mission is to translate text in English to French.",
        ]
        generation_config = google_types.GenerateContentConfig(system_instruction=system_instruction)

        # Asynchronously call the model
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=input_for_model,
            config=generation_config,
        )

        translated_text = response.text
        logging.info("Successfully received translation from Gemini.")
        return translated_text

    except KeyError:
        logging.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        return None
    except exceptions.GoogleAPICallError as e:
        logging.error("A Google API call error occurred: %s", e)
        return None
    except Exception as e:
        logging.error("An unexpected error occurred during translation: %s", e)
        return None


# --- Main Execution Block ---


async def main() -> None:
    """
    Main function to parse command-line arguments and run the translation.
    """
    # Setup command-line argument parsing to get input text from the user
    parser = argparse.ArgumentParser(
        description="Translate English text to French using Gemini with Netra PII protection."
    )
    parser.add_argument("message", type=str, help="The English text to translate. Please wrap in quotes.")
    args = parser.parse_args()

    # Initialize the Netra SDK before running the workflow
    initialize_netra_sdk()

    # Run the translation function and get the result
    translated_message = await translate_text_with_pii_protection(args.message)  # type: ignore[misc]

    if translated_message:
        print("\n--- Translation Result ---")
        print(f"Original: {args.message}")
        print(f"Translated: {translated_message}")
        print("--------------------------\n")
    else:
        print("\nTranslation failed. Please check the logs for more details.\n")


if __name__ == "__main__":
    # Example Usage from command line:
    # python your_script_name.py "Hello, my name is John and my email is john.doe@example.com"
    # python your_script_name.py "This is a test without any personal data."

    # Run the main asynchronous function
    asyncio.run(main())
