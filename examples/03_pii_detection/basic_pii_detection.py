import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

# Import the Netra SDK
from netra import Netra
from netra.decorators import workflow
from netra.pii import get_default_detector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@workflow(name="pii_detection_workflow_flag_mode")  # type: ignore[arg-type]
def demonstrate_flag_mode() -> Dict[str, Any]:
    """Demonstrate PII detection in FLAG mode."""
    print("\n🏁 Demonstrating FLAG Mode")
    print("-" * 40)

    test_texts = [
        "Contact me at john.doe@example.com for more information.",
        "My phone number is 555-123-4567 and email is test@domain.com",
        "This text has no PII information.",
        "SSN: 123-45-6789 should be detected",
    ]
    pii_detector = get_default_detector(action_type="FLAG")
    result = pii_detector.detect(test_texts)
    # Convert PIIDetectionResult to dictionary
    return {
        "mode": "FLAG",
        "original_text": result.original_text,
        "masked_text": result.masked_text,
        "has_pii": result.has_pii,
        "pii_entities": result.pii_entities,
        "is_blocked": result.is_blocked,
    }


@workflow(name="pii_detection_workflow_mask_mode")  # type: ignore[arg-type]
def demonstrate_mask_mode() -> Dict[str, Any]:
    """Demonstrate PII detection in MASK mode."""
    print("\n🎭 Demonstrating MASK Mode")
    print("-" * 40)

    test_texts = [
        "Please contact John at john.doe@company.com or call 555-987-6543",
        "Customer SSN is 987-65-4321 and phone is (555) 123-4567",
    ]

    pii_detector = get_default_detector(action_type="MASK")
    result = pii_detector.detect(test_texts)

    # Convert PIIDetectionResult to dictionary
    return {
        "mode": "MASK",
        "original_text": result.original_text,
        "masked_text": result.masked_text,
        "has_pii": result.has_pii,
        "pii_entities": result.pii_entities,
        "is_blocked": result.is_blocked,
    }


@workflow(name="pii_detection_workflow_block_mode")  # type: ignore[arg-type]
def demonstrate_block_mode() -> Dict[str, Any]:
    """Demonstrate PII detection in BLOCK mode."""
    print("\n🚫 Demonstrating BLOCK Mode")
    print("-" * 40)

    test_texts = [
        "This is safe text with no PII",
        "Contact admin@company.com for support",  # This will be blocked
        "Call us at 555-999-8888",  # This will be blocked
    ]

    pii_detector = get_default_detector(action_type="BLOCK")
    result = pii_detector.detect(test_texts)

    # Convert PIIDetectionResult to dictionary
    return {
        "mode": "BLOCK",
        "original_text": result.original_text,
        "masked_text": result.masked_text,
        "has_pii": result.has_pii,
        "pii_entities": result.pii_entities,
        "is_blocked": result.is_blocked,
    }


def main() -> None:
    """
    Main function demonstrating Netra SDK PII detection capabilities.
    """
    # Initialize Netra SDK
    try:
        Netra.init(
            app_name="basic-pii-detection-example",
            environment="development",
            trace_content=True,
            headers=f"x-api-key={os.getenv('NETRA_API_KEY')}",
        )
        logger.info("✅ Netra SDK initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Netra SDK: {e}")
        return

    # Set user context
    Netra.set_user_id("pii_detection_demo_user")
    Netra.set_session_id("pii_detection_demo_session")

    print("🎯 Netra SDK Basic PII Detection Example")
    print("=" * 50)

    # Demonstrate different PII detection modes and features
    demonstrate_flag_mode()  # type: ignore[misc]
    demonstrate_mask_mode()  # type: ignore[misc]
    demonstrate_block_mode()  # type: ignore[misc]

    print("\n" + "=" * 50)
    print("🎉 Basic PII detection example completed!")


if __name__ == "__main__":
    main()
