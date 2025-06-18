"""
Netra SDK PII Detection Example

This example demonstrates how to use the Netra SDK's PII detection capabilities
to identify, mask, or block sensitive information in text.

Features demonstrated:
- Default PII detector (Presidio-based) with FLAG mode
- Regex-based PII detector with MASK mode
- Blocking behavior with BLOCK mode
- Different types of PII detection (SSN, credit cards, emails, etc.)
- Error handling and result inspection
"""

import logging

# Import the Netra SDK
from netra import Netra
from netra.exceptions.pii import PIIBlockedException
from netra.pii import PIIDetector, RegexPIIDetector, get_default_detector

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

# Get logger for this module
logger = logging.getLogger(__name__)


def print_header(text: str, width: int = 80) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {text} ".center(width, "#"))
    print("=" * width)


def print_section(text: str, width: int = 40) -> None:
    """Print a section header."""
    print(f"\n{'-' * width}")
    print(f" {text}")
    print(f"{'=' * len(text)}")


def detect_pii_example(detector: PIIDetector, text: str, example_num: int) -> None:
    """
    Helper function to demonstrate PII detection with a given detector and text.
    """
    try:
        print_section(f"Example {example_num}")
        print(f"Detector: {detector.__class__.__name__}")
        print(f"Action Type: {getattr(detector, 'action_type', 'N/A')}")
        print(f"\nInput text:\n{text}")

        # Detect PII
        result = detector.detect(text)

        # Print results
        if hasattr(result, "has_pii") and result.has_pii:
            print("\n🚨 PII DETECTED!")
            print(f"Entity types found: {', '.join(result.pii_entities.keys())}")
            if hasattr(result, "masked_text") and result.masked_text:
                print("\nMasked text:")
                print(result.masked_text)
        else:
            print("\n✅ No PII detected")

    except PIIBlockedException as e:
        print("\n🚫 PII BLOCKED!")
        if hasattr(e, "pii_entities") and e.pii_entities:
            print(f"Blocked entity types: {', '.join(e.pii_entities.keys())}")
        if hasattr(e, "masked_text") and e.masked_text:
            print("\nMasked text:")
            print(e.masked_text)
        if hasattr(e, "pii_actions") and e.pii_actions:
            print(f"PII actions: {', '.join(e.pii_actions.keys())}")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """
    Main function demonstrating various PII detection scenarios.
    """
    try:
        # Initialize the Netra SDK with console span exporter disabled
        Netra.init(
            app_name="pii-detection-example",
            environment="local",
        )

        print_header("Netra SDK PII Detection Example")
        print("This example demonstrates different PII detection modes:")
        print("- FLAG: Identifies PII without modifying the text")
        print("- MASK: Replaces detected PII with entity type markers")
        print("- BLOCK: Raises an exception when PII is detected")
        print("=" * 80)

        # Example 1: Using the default detector (Presidio-based) in FLAG mode
        print_section("Default Detector (FLAG mode)")
        print("Detects PII without modifying the text.\n")

        detector = get_default_detector(action_type="FLAG")

        # Test with different types of PII
        texts = [
            "My name is Jane Doe and my email is jane.doe@example.com",
            "My SSN is 123-45-6789 and phone is (555) 123-4567",
            "Credit card: 4111 1111 1111 1111, expiry 12/25, CVV 123",
            "I live at 123 Main St, Anytown, CA 12345",
            "My passport number is A12345678 and my driver's license is D123-456-789-012",
        ]

        for i, text in enumerate(texts, 1):
            detect_pii_example(detector, text, i)

        # Example 2: Using the RegexPIIDetector in MASK mode
        print_section("RegexPIIDetector (MASK mode)")
        print("Masks detected PII in the text.\n")

        mask_detector = RegexPIIDetector(action_type="MASK")

        test_text = """
        Please process my order with the following details:
        Name: John Smith
        Email: john.smith@example.com
        Phone: (555) 987-6543
        Credit Card: 5555-1234-5678-9012
        Billing Address: 456 Payment St, Somewhere, NY 10001
        """

        detect_pii_example(mask_detector, test_text, 1)

        # Example 3: Using detector in BLOCK mode
        print_section("RegexPIIDetector (BLOCK mode)")
        print("Raises an exception when PII is detected.\n")

        block_detector = RegexPIIDetector(action_type="BLOCK")

        sensitive_texts = [
            "My SSN is 123-45-6789",
            "Credit card number is 4111-1111-1111-1111",
            "My password is S3cr3tP@ssw0rd!",
        ]

        for i, text in enumerate(sensitive_texts, 1):
            try:
                print(f"\nTesting text {i}: {text}")
                # In BLOCK mode, this line should never be reached if PII is detected
                block_detector.detect(text)
                print("✅ No PII detected (this shouldn't happen with blocking mode)")
            except PIIBlockedException as e:
                print("\n🚫 PII DETECTED AND BLOCKED!")
                print(f"- Entity types found: {e.pii_entities}")
                if hasattr(e, "masked_text") and e.masked_text:
                    print("\nMasked text:")
                    print(e.masked_text)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback

                traceback.print_exc()

        print_header("PII Detection Example Completed Successfully!")

    except Exception as e:
        logger.error(f"Error in PII detection example: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
