# Combat SDK

Combat SDK is a Python library for AI application observability that provides OpenTelemetry-based monitoring, tracing, and PII protection for LLM and vector database applications. It enables easy instrumentation, session tracking, and privacy-focused data collection for AI systems in production environments.

## Installation

You can install the Combat SDK using pip:

```bash
pip install git+https://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@github.com/KeyValueSoftwareSystems/promptops-sdk-py.git@beta
```

Or, using Poetry:

```bash
poetry add combat-sdk @ git+https://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@github.com/KeyValueSoftwareSystems/promptops-sdk-py.git@beta
```

## Usage

### Basic Setup

Initialize the Combat SDK at the start of your application:

```python
from combat import Combat

# Initialize with default settings
Combat.init(app_name="Your application name")

# Or with custom configuration
api_key = "Your API key"
headers = f"x-api-key={api_key}"
Combat.init(
    app_name="Your application name",
    otlp_endpoint="https://api.dev.getcombat.ai/telemetry",
    headers=headers,
    trace_content=True,
    environment="Your Application environment"
)
```

### Session Management

Track user sessions to correlate telemetry data:

```python
# Set session identification
Combat.set_session_id("unique-session-id")
Combat.set_user_id("user-123")
Combat.set_user_account_id("account-456")

# Add custom context attributes
Combat.set_custom_attributes("customer_tier", "premium")
Combat.set_custom_attributes("region", "us-east")

# Record custom events
Combat.set_custom_event("user_feedback", {"rating": 5, "comment": "Great response!"})
```

### Thread Safety

The Combat SDK is designed to be thread-safe. The initialization process is protected with a lock to prevent race conditions when multiple threads attempt to initialize the SDK simultaneously.

## Development Setup

To set up your development environment for the Combat SDK, run the provided setup script:

```bash
./setup_dev.sh
```

This script will:

1. Install all Python dependencies in development mode
2. Set up pre-commit hooks for code quality
3. Configure commit message formatting

### Manual Setup

If you prefer to set up manually:

```bash
# Install dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

- **`type`** must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
- **`scope`** is optional: a label indicating a section of the code (e.g. `api`, `cli`, `auth`).
- **Short summary** must be written in the imperative (“add feature”, not “added feature”).
- **Body** (if present) should explain _why_ and _what_, not _how_.
- **Footer** can be used for “BREAKING CHANGE:” or issue references.

---
