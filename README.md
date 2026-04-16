# Netra SDK

🚀 **Netra SDK** is a comprehensive Python library for AI application observability that provides OpenTelemetry-based monitoring, and tracing for LLM applications. It enables easy instrumentation, session tracking, and privacy-focused data analysis for AI systems.

## ✨ Key Features

- 🔍 **Comprehensive AI Observability**: Monitor LLM calls, vector database operations, and HTTP requests
- 📊 **OpenTelemetry Integration**: Industry-standard tracing and metrics
- 🎯 **Decorator Support**: Easy instrumentation with `@workflow`, `@agent`, `@task` and `@span` decorators
- 🔧 **Multi-Provider Support**: Works with OpenAI, Cohere, Google GenAI, Mistral, and more
- 📈 **Session Management**: Track user sessions and custom attributes
- 🌐 **HTTP Client Instrumentation**: Automatic tracing for aiohttp and httpx
- 💾 **Vector Database Support**: Weaviate, Qdrant, and other vector DB instrumentation

## 📦 Installation

You can install the Netra SDK using pip:

```bash
pip install netra-sdk
```

Or, using Poetry:

```bash
poetry add netra-sdk
```


## 🚀 Quick Start

### Basic Setup

Initialize the Netra SDK at the start of your application:

```python
from netra import Netra
from netra.instrumentation.instruments import InstrumentSet

# Initialize with default settings
Netra.init(app_name="Your application name", instruments={InstrumentSet.OPENAI, InstrumentSet.ANTHROPIC})

# Or with custom configuration
api_key = "Your API key"
headers = f"x-api-key={api_key}"
Netra.init(
    app_name="Your application name",
    headers=headers,
    trace_content=True,
    environment="Your Application environment",
    instruments={InstrumentSet.OPENAI, InstrumentSet.ANTHROPIC},
)
```

### 🎯 Decorators for Easy Instrumentation

Use decorators to automatically trace your functions and classes:

```python
from netra.decorators import workflow, agent, task, span

@workflow
def data_processing_workflow(data):
    """Main workflow for processing data"""
    cleaned_data = clean_data(data)
    return analyze_data(cleaned_data)

@agent
def ai_assistant(query):
    """AI agent that processes user queries"""
    return generate_response(query)

@task
def data_validation_task(data):
    """Task for validating input data"""
    return validate_schema(data)

@span
def data_processing_span(data):
    """Span for processing data"""
    return process_data(data)

# Works with async functions too
@workflow(name="Async Data Pipeline")
async def async_workflow(data):
    result = await process_data_async(data)
    return result

# Apply to classes to instrument all methods
@agent
class CustomerSupportAgent:
    def handle_query(self, query):
        return self.process_query(query)

    def escalate_issue(self, issue):
        return self.forward_to_human(issue)

@task
async def async_task(data):
    """Task for processing data"""
    return await process_data_async(data)

@span
async def async_span(data):
    """Span for processing data"""
    return await process_data_async(data)
```

## 🔍 Supported Instrumentations

### 🤖 LLM Providers

- **OpenAI** - GPT models and completions API
- **Anthropic Claude** - Claude 3 models and messaging API
- **Cohere** - Command models and generation API
- **Google GenAI (Gemini)** - Gemini Pro and other Google AI models
- **Mistral AI** - Mistral models and chat completions
- **Aleph Alpha** - Advanced European AI models
- **AWS Bedrock** - Amazon's managed AI service
- **Groq** - High-performance AI inference
- **Ollama** - Local LLM deployment and management
- **Replicate** - Cloud-based model hosting platform
- **Together AI** - Collaborative AI platform
- **Transformers** - Hugging Face transformers library
- **Vertex AI** - Google Cloud AI platform
- **Watson X** - IBM's enterprise AI platform

### 💾 Vector Databases

- **Weaviate** - Open-source vector database with GraphQL
- **Qdrant** - High-performance vector similarity search
- **Pinecone** - Managed vector database service
- **Chroma** - Open-source embedding database
- **LanceDB** - Fast vector database for AI applications
- **Marqo** - Tensor-based search engine
- **Milvus** - Open-source vector database at scale
- **Redis** - Vector search with Redis Stack

### 🌐 HTTP Clients & Web Frameworks

- **HTTPX** - Modern async HTTP client
- **AIOHTTP** - Asynchronous HTTP client/server
- **FastAPI** - Modern web framework for APIs
- **Requests** - Popular HTTP library for Python
- **Django** - High-level Python web framework
- **Flask** - Lightweight WSGI web application framework
- **Falcon** - High-performance Python web framework
- **Starlette** - Lightweight ASGI framework/toolkit
- **Tornado** - Asynchronous networking library and web framework
- **gRPC** - High-performance, open-source universal RPC framework
- **Urllib** - Standard Python HTTP client library
- **Urllib3** - Powerful, user-friendly HTTP client for Python

### 🗄️ Database Clients

- **PyMySQL** - Pure Python MySQL client
- **Redis** - In-memory data structure store
- **SQLAlchemy** - SQL toolkit and Object-Relational Mapper
- **Psycopg** - Modern PostgreSQL database adapter for Python
- **Pymongo** - Python driver for MongoDB
- **Elasticsearch** - Distributed, RESTful search and analytics engine
- **Cassandra** - Distributed NoSQL database
- **PyMSSQL** - Simple Microsoft SQL Server client
- **MySQL Connector** - Official MySQL driver
- **Sqlite3** - Built-in SQL database engine
- **Aiopg** - Asynchronous PostgreSQL client
- **Asyncpg** - Fast asynchronous PostgreSQL client
- **Pymemcache** - Comprehensive Memcached client
- **Tortoise ORM** - Easy-to-use asyncio ORM

### 📨 Messaging & Task Queues

- **Celery** - Distributed task queue
- **Pika** - Pure-Python implementation of the AMQP 0-9-1 protocol
- **AIO Pika** - Asynchronous AMQP client
- **Kafka-Python** - Python client for Apache Kafka
- **AIOKafka** - Asynchronous Python client for Kafka
- **Confluent-Kafka** - Confluent's Python client for Apache Kafka
- **Boto3 SQS** - Amazon SQS client via Boto3

### 🔧 AI Frameworks & Orchestration

- **LangChain** - Framework for developing LLM applications
- **LangGraph** - Modern framework for LLM applications
- **LlamaIndex** - Data framework for LLM applications
- **Haystack** - End-to-end NLP framework
- **CrewAI** - Multi-agent AI systems
- **Pydantic AI** - AI model communication standard
- **MCP (Model Context Protocol)** - AI model communication standard
- **LiteLLM** - LLM provider agnostic client


## 📊 Context and Event Logging

Track user sessions and add custom context:

```python
from netra import Netra
from netra.instrumentation.instruments import InstrumentSet

# Initialize SDK
Netra.init(app_name="My App", instruments={InstrumentSet.OPENAI})

# Set session identification
Netra.set_session_id("unique-session-id")
Netra.set_user_id("user-123")
Netra.set_tenant_id("tenant-456")

# Add custom context attributes
Netra.set_custom_attributes(key="customer_tier", value="premium")
Netra.set_custom_attributes(key="region", value="us-east")

# Record custom events
Netra.set_custom_event(event_name="user_feedback", attributes={
    "rating": 5,
    "comment": "Great response!",
    "timestamp": "2024-01-15T10:30:00Z"
})

# Custom events for business metrics
Netra.set_custom_event(event_name="conversion", attributes={
    "type": "subscription",
    "plan": "premium",
    "value": 99.99
})
```
## 🔄 Custom Span Tracking

Use the custom span tracking utility to track external API calls with detailed observability:

```python
from netra import Netra, UsageModel

# Start a new span
with Netra.start_span("image_generation") as span:
    # Set span attributes
    span.set_prompt("A beautiful sunset over mountains")
    span.set_negative_prompt("blurry, low quality")
    span.set_model("dall-e-3")
    span.set_llm_system("openai")

    # Set usage data with UsageModel
    usage_data = [
        UsageModel(
            model="dall-e-3",
            usage_type="image_generation",
            units_used=1,
            cost_in_usd=0.02
        )
    ]
    span.set_usage(usage_data)

    # Your API calls here
    # ...

    # Set custom attributes
    span.set_attribute("custom_key", "custom_value")

    # Add events
    span.add_event("generation_started", {"step": "1", "status": "processing"})
    span.add_event("processing_completed", {"step": "rendering"})

    # Get the current active open telemetry span
    current_span = span.get_current_span()

    # Track database operations and other actions
    action = ActionModel(
        start_time="1753857049844249088",  # timestamp in nanoseconds
        action="DB",
        action_type="INSERT",
        affected_records=[
            {"record_id": "user_123", "record_type": "user"},
            {"record_id": "profile_456", "record_type": "profile"}
        ],
        metadata={
            "table": "users",
            "operation_id": "tx_789",
            "duration_ms": "45"
        },
        success=True
    )
    span.set_action([action])

    # Record API calls
    api_action = ActionModel(
        start_time="1753857049844249088",  # timestamp in nanoseconds
        action="API",
        action_type="CALL",
        metadata={
            "endpoint": "/api/v1/process",
            "method": "POST",
            "status_code": 200,
            "duration_ms": "120"
        },
        success=True
    )
    span.set_action([api_action])
```

### Action Tracking Schema

Action tracking follows this schema:

```python
[
    {
        "start_time": str,            # Start time of the action in nanoseconds
        "action": str,                # Type of action (e.g., "DB", "API", "CACHE")
        "action_type": str,           # Action subtype (e.g., "INSERT", "SELECT", "CALL")
        "affected_records": [         # Optional: List of records affected
            {
                "record_id": str,     # ID of the affected record
                "record_type": str    # Type of the record
            }
        ],
        "metadata": Dict[str, str],   # Additional metadata as key-value pairs
        "success": bool              # Whether the action succeeded
    }
]
```

## 🔧 Advanced Configuration

### Environment Variables

Netra SDK can be configured using the following environment variables:

#### Netra-specific Variables

| Variable Name | Description | Default |
|---------------|-------------|---------|
| `NETRA_APP_NAME` | Logical name for your service | Falls back to `OTEL_SERVICE_NAME` or `llm_tracing_service` |
| `NETRA_OTLP_ENDPOINT` | URL for OTLP collector | Falls back to `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `NETRA_API_KEY` | API key for authentication | `None` |
| `NETRA_HEADERS` | Additional headers in W3C Correlation-Context format | `None` |
| `NETRA_DISABLE_BATCH` | Disable batch span processor (`true`/`false`) | `false` |
| `NETRA_TRACE_CONTENT` | Whether to capture prompt/completion content (`true`/`false`) | `true` |
| `NETRA_ENV` | Deployment environment (e.g., `prod`, `staging`, `dev`) | `local` |
| `NETRA_RESOURCE_ATTRS` | JSON string of custom resource attributes | `{}` |

#### Standard OpenTelemetry Variables

| Variable Name | Description | Used When |
|---------------|-------------|-----------|
| `OTEL_SERVICE_NAME` | Logical name for your service | When `NETRA_APP_NAME` is not set |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL for OTLP collector | When `NETRA_OTLP_ENDPOINT` is not set |
| `OTEL_EXPORTER_OTLP_HEADERS` | Additional headers for OTLP exporter | When `NETRA_HEADERS` is not set |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes | When `NETRA_RESOURCE_ATTRS` is not set |

### Configuration Precedence

Configuration values are resolved in the following order (highest to lowest precedence):

1. **Code Parameters**: Values passed directly to `Netra.init()`
2. **Netra Environment Variables**: `NETRA_*` variables
3. **OpenTelemetry Environment Variables**: Standard `OTEL_*` variables
4. **Default Values**: Fallback values defined in the SDK

This allows you to:
- **🔄 Vendor Agnostic**: Switch between observability platforms without code changes
- **📊 Standard Format**: Consistent telemetry data across all tools
- **🔧 Flexible Integration**: Works with existing observability infrastructure
- **🚀 Future Proof**: Built on industry-standard protocols
- **📈 Rich Ecosystem**: Leverage the entire OpenTelemetry ecosystem


## 🧪 Tests

Our test suite is built on `pytest` and is designed to ensure the reliability and stability of the Netra SDK. We follow comprehensive testing standards, including unit, integration, and thread-safety tests.

### Running Tests

To run the complete test suite, use the following command from the root of the project:

```bash
poetry run pytest
```


### Run Specific Test File
To run a specific test file, use the following command from the root of the project:
```bash
poetry run pytest tests/test_netra_init.py
```

### Test Coverage

To generate a test coverage report, you can run:

```bash
poetry run pytest --cov=netra --cov-report=html
```

This will create an `htmlcov` directory with a detailed report.

### Running Specific Test Categories

Tests are organized using `pytest` markers. You can run specific categories of tests as follows:

```bash
# Run only unit tests (default)
poetry run pytest -m unit

# Run only integration tests
poetry run pytest -m integration

# Run only thread-safety tests
poetry run pytest -m thread_safety
```

For more detailed information on our testing strategy, fixtures, and best practices, please refer to the `README.md` file in the `tests` directory.



## 🛠️ Development Setup

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information on how to contribute to the project, including development setup, testing, and our commit message format.

---
