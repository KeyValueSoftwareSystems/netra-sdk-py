import logging
import os
from typing import Optional

# Import the powerful Netra SDK for AI observability
from netra import Netra

# Configure comprehensive logging for initialization tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Initialize logger for this initialization module
logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """
    🔍 Validate that all required environment variables are properly configured.

    Returns:
        bool: True if environment is properly configured, False otherwise
    """
    logger.info("🔍 Validating environment configuration...")

    api_key = os.getenv("NETRA_API_KEY")
    if not api_key:
        logger.error("❌ NETRA_API_KEY environment variable is not set")
        print("🔧 Please configure your API key:")
        print("   export NETRA_API_KEY='your-api-key-here'")
        return False

    logger.info("✅ Environment validation completed successfully")
    return True


def initialize_netra_sdk() -> bool:
    """
    🚀 Initialize the Netra SDK with comprehensive configuration.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("🚀 Initializing Netra SDK with optimal settings...")

        # Initialize with production-ready configuration
        Netra.init(
            app_name="netra-basic-initialization-demo",
            environment="development",
            trace_content=True,  # Enable content tracing for comprehensive monitoring
            headers=f"x-api-key={os.getenv('NETRA_API_KEY')}",
        )

        logger.info("✅ Netra SDK initialized successfully with full observability")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Netra SDK: {str(e)}")
        print(f"🔧 Initialization error: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Verify your API key is correct")
        print("   - Check your internet connection")
        print("   - Ensure netra-sdk is properly installed")
        return False


def configure_user_context(user_id: str) -> None:
    """
    👤 Configure user context for request tracking and personalization.

    Args:
        user_id: Unique identifier for the user
    """
    try:
        logger.info(f"👤 Configuring user context for user: {user_id}")
        Netra.set_user_id(user_id)
        logger.info(f"✅ User context established successfully for: {user_id}")

    except Exception as e:
        logger.error(f"❌ Failed to set user context: {str(e)}")
        raise


def configure_session_context(session_id: str) -> None:
    """
    🔗 Configure session context for conversation flow tracking.

    Args:
        session_id: Unique identifier for the session
    """
    try:
        logger.info(f"🔗 Establishing session context: {session_id}")
        Netra.set_session_id(session_id)
        logger.info(f"✅ Session context configured successfully: {session_id}")

    except Exception as e:
        logger.error(f"❌ Failed to set session context: {str(e)}")
        raise


def configure_tenant_context(tenant_id: Optional[str] = None) -> None:
    """
    🏢 Configure tenant context for multi-tenant application support.

    Args:
        tenant_id: Optional unique identifier for the tenant
    """
    if not tenant_id:
        logger.info("🏢 Skipping tenant configuration (single-tenant mode)")
        return

    try:
        logger.info(f"🏢 Configuring tenant context: {tenant_id}")
        Netra.set_tenant_id(tenant_id)
        logger.info(f"✅ Tenant context established successfully: {tenant_id}")

    except Exception as e:
        logger.error(f"❌ Failed to set tenant context: {str(e)}")
        raise


def main() -> None:
    """
    🎯 Main function demonstrating comprehensive Netra SDK initialization.

    This function orchestrates the complete initialization process with
    proper error handling and user feedback.
    """
    print("🚀 Netra SDK - Basic Initialization Guide")
    print("=" * 55)
    print("Welcome to your first step in AI observability!")
    print()

    # Step 1: Environment validation
    print("📋 Step 1: Environment Validation")
    if not validate_environment():
        print("❌ Environment validation failed. Please fix the issues above.")
        return
    print("✅ Environment validation completed successfully!")
    print()

    # Step 2: SDK initialization
    print("🚀 Step 2: Netra SDK Initialization")
    if not initialize_netra_sdk():
        print("❌ SDK initialization failed. Please check the error messages above.")
        return
    print("✅ Netra SDK initialized and ready for monitoring!")
    print()

    # Step 3: User context configuration
    print("👤 Step 3: User Context Configuration")
    demo_user_id = "demo_user_12345"
    configure_user_context(demo_user_id)
    print(f"✅ User context configured: {demo_user_id}")
    print()

    # Step 4: Session context configuration
    print("🔗 Step 4: Session Context Configuration")
    demo_session_id = "demo_session_67890"
    configure_session_context(demo_session_id)
    print(f"✅ Session context configured: {demo_session_id}")
    print()

    # Step 5: Tenant context configuration (optional)
    print("🏢 Step 5: Tenant Context Configuration")
    demo_tenant_id = "demo_tenant_org"
    configure_tenant_context(demo_tenant_id)
    print(f"✅ Tenant context configured: {demo_tenant_id}")
    print()


if __name__ == "__main__":
    main()
