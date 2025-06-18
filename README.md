# Netra SDK

Netra SDK is a Python library for AI application observability that provides OpenTelemetry-based monitoring, tracing, and PII protection for LLM and vector database applications. It enables easy instrumentation, session tracking, and privacy-focused data collection for AI systems in production environments.

## Installation

You can install the Netra SDK using pip:

```bash
pip install git+https://<GITHUB_TOKEN>@github.com/KeyValueSoftwareSystems/promptops-sdk-py.git@beta
```

Or, using Poetry:

```bash
poetry add netra-sdk @ git+https://<GITHUB_TOKEN>@github.com/KeyValueSoftwareSystems/promptops-sdk-py.git@beta
```

## Usage

### Basic Setup

Initialize the Netra SDK at the start of your application:

```python
from netra import Netra

# Initialize with default settings
Netra.init(app_name="Your application name")

# Or with custom configuration
api_key = "Your API key"
headers = f"x-api-key={api_key}"
Netra.init(
    app_name="Your application name",
    otlp_endpoint="https://api.dev.getnetra.ai/telemetry",
    headers=headers,
    trace_content=True,
    environment="Your Application environment"
)
```

### Session Management

Track user sessions to correlate telemetry data:

```python
# Set session identification
Netra.set_session_id("unique-session-id")
Netra.set_user_id("user-123")
Netra.set_user_account_id("account-456")

# Add custom context attributes
Netra.set_custom_attributes("customer_tier", "premium")
Netra.set_custom_attributes("region", "us-east")

# Record custom events
Netra.set_custom_event("user_feedback", {"rating": 5, "comment": "Great response!"})
```

### Thread Safety

The Netra SDK is designed to be thread-safe. The initialization process is protected with a lock to prevent race conditions when multiple threads attempt to initialize the SDK simultaneously.

## Development Setup

To set up your development environment for the Netra SDK, run the provided setup script:

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
