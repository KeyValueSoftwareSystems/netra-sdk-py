# Netra SDK Examples

This directory contains example scripts demonstrating how to use the Netra SDK for various use cases. Each example is designed to showcase specific features and capabilities of the SDK.

## Available Examples

1. **Basic Initialization** (`basic_initialization.py`)
   Shows how to initialize the SDK with various configuration options.

2. **LLM Tracing** (`llm_tracing.py`)
   Demonstrates automatic instrumentation and tracing of LLM interactions using popular libraries like OpenAI and LangChain.
   - Automatically traces LLM API calls
   - Captures prompts, responses, and metadata
   - Works with both direct API calls and LangChain integrations
   - Example output includes token usage, model information, and response content

3. **PII Detection** (`pii_detection.py`)
   Showcases the SDK's PII detection functionality, including how to configure detectors for flagging, masking, or blocking PII.

4. **Session Management** (`session_management.py`)
   Illustrates how to use session tracking features to associate user interactions and maintain context.

5. **Injection Scanning** (`injection_scanning.py`)
   Shows how to use prompt injection detection to identify potentially malicious inputs.

6. **Integrated Example** (`integrated_example.py`)
   Comprehensive example that combines multiple SDK features in a simulated chatbot application.

## Running the Examples

### Prerequisites

1. Install the Netra SDK and core dependencies:

   ```bash
   pip install netra-sdk
   ```

### Running LLM Tracing Example

1. Install required dependencies:

   ```bash
   pip install openai langchain
   ```

2. Set up environment variables:

   ```bash
   export NETRA_API_KEY='your-netra-api-key'
   export OPENAI_API_KEY='your-openai-api-key'
   ```

3. Run the LLM tracing example:

   ```bash
   python examples/llm_tracing.py
   ```

### Running PII Detection Example

1. Run the PII detection example:

   ```bash
   python examples/pii_detection.py
   ```

## Configuration

### Environment Variables

- `NETRA_API_KEY`: Required for all examples
- `OPENAI_API_KEY`: Required for LLM examples using OpenAI
- `NETRA_ENDPOINT`: Optional custom endpoint URL (defaults to console output)
- `NETRA_APP_NAME`: Optional application name for tracing

### Example Output

When running the LLM tracing example, you'll see detailed tracing output including:

- Request/response content
- Token usage
- Model information
- Performance metrics
- Any detected issues or anomalies

## Notes

- The SDK will automatically detect and instrument supported LLM libraries
- For production use, configure a proper OTLP endpoint instead of console output
- All sensitive data handling follows security best practices
- The SDK is designed to be non-intrusive and have minimal performance impact

## Troubleshooting

- If you see warnings about missing OTLP endpoints, configure a proper endpoint or ignore if console output is sufficient
- Make sure all required environment variables are set
- Check that you have the necessary permissions for the API keys being used

## Further Reading

For more detailed information on the Netra SDK, refer to the main [README.md](../README.md) file in the repository root.
