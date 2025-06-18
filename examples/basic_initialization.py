"""
Basic Netra SDK Initialization Example

This example demonstrates how to properly initialize and use the Netra SDK in your application.
It covers:
1. Basic SDK initialization with configuration
2. Error handling during initialization
3. Setting up user and session context
"""

import logging
import sys

# Import the Netra SDK
from netra import Netra
from netra.exceptions.injection import InjectionException
from netra.exceptions.pii import PIIBlockedException

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout
)


def main() -> None:
    """
    Main function demonstrating Netra SDK initialization and basic usage.
    """
    try:
        # Initialize the Netra SDK with your application name and configuration
        print("🚀 Initializing Netra SDK...")
        Netra.init(
            app_name="example-app",
            trace_content=True,  # Enable content tracing for debugging
            environment="development",  # Set the environment (development/staging/production)
        )
        print("✅ Netra SDK initialized successfully!")

        # Example: Set user ID for session tracking
        user_id = "user123"
        print(f"👤 Setting user context: {user_id}")
        Netra.set_user_id(user_id)

        # Example: Set session ID
        session_id = "session_123"
        print(f"🔗 Setting session ID: {session_id}")
        Netra.set_session_id(session_id)

        # Example: Set tenant ID (if applicable)
        tenant_id = "acme-corp"
        print(f"🏢 Setting tenant ID: {tenant_id}")
        Netra.set_tenant_id(tenant_id)

        # Your application code would go here
        print("\n🚀 Your application is now running with Netra monitoring...")
        print("   - All API calls and LLM interactions will be automatically tracked")
        print("   - Session and user context is being tracked")
        print("\nTry interacting with your application to see the monitoring in action!")

        # Simulate some application work
        input("\nPress Enter to exit...")

    except (InjectionException, PIIBlockedException) as e:
        print(f"❌ Security violation detected: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n🔌 Cleaning up...")
        # No explicit shutdown needed as Netra handles cleanup automatically

    print("\n✅ Application completed successfully.")


if __name__ == "__main__":
    main()
