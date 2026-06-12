"""
Integration tests for token-to-span attribute assignment.

Verifies that each LLM provider's usage extraction function correctly maps
response fields to standardised OpenTelemetry span attribute keys, and that
SpanIOProcessor aliasing rewrites input_tokens/output_tokens to the canonical
prompt_tokens/completion_tokens keys before export.

Flow under test:
    Provider utils  →  SpanIOProcessor-patched span.set_attribute  →  final attributes
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.cerebras.utils import set_response_attributes as cerebras_set_usage
from netra.instrumentation.dspy.utils import extract_usage_info as dspy_extract_usage
from netra.instrumentation.google_genai.utils import set_response_attributes as google_genai_set_usage
from netra.instrumentation.groq.utils import _set_usage_attributes as groq_set_usage
from netra.instrumentation.litellm.utils import _set_usage_attributes as litellm_set_usage
from netra.instrumentation.openai.utils import _set_usage_attributes as openai_set_usage
from netra.instrumentation.pydantic_ai.utils import _set_call_tools_node_attributes
from netra.instrumentation.pydantic_ai.utils import set_pydantic_response_attributes as pydantic_ai_set_usage
from netra.processors.span_io_processor import SpanIOProcessor


@pytest.fixture
def patched_span():
    """
    A MagicMock span pre-patched by SpanIOProcessor.on_start().

    Attributes written via set_attribute are stored in span.attributes so
    tests can assert on the final resolved keys.
    """
    span = MagicMock()
    span.attributes = {}
    span._is_recording = True

    def set_attr(key, value):
        span.attributes[key] = value

    span.set_attribute.side_effect = set_attr
    span.is_recording.return_value = True

    mock_context = Mock()
    mock_context.is_valid = True
    span.get_span_context.return_value = mock_context

    processor = SpanIOProcessor()
    processor.on_start(span)
    return span


class TestTokenUsageAttributes:
    """Integration tests for token-to-span attribute assignment."""

    class TestSpanIO:
        """Tests for SpanIOProcessor aliasing."""

        def test_aliases_input_tokens_to_prompt_tokens(self, patched_span):
            """Test that SpanIOProcessor rewrites input_tokens to prompt_tokens."""
            # Act
            patched_span.set_attribute("gen_ai.usage.input_tokens", 100)
            patched_span.set_attribute("gen_ai.usage.output_tokens", 50)

            # Assert — canonical keys present, raw alias keys absent
            assert patched_span.attributes["gen_ai.usage.prompt_tokens"] == 100
            assert patched_span.attributes["gen_ai.usage.completion_tokens"] == 50
            assert "gen_ai.usage.input_tokens" not in patched_span.attributes
            assert "gen_ai.usage.output_tokens" not in patched_span.attributes

    class TestOpenAI:
        """Tests for OpenAI token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test OpenAI token usage extraction and mapping."""
            # Arrange
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 7},
            }

            # Act
            openai_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 10
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 20
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 30
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 5
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 7

        def test_token_usage_alternative_keys(self, patched_span):
            """Test OpenAI token usage with alternative keys (input/output)."""
            # Arrange
            usage = {
                "input_tokens": 15,
                "output_tokens": 25,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 4},
            }

            # Act
            openai_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 15
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 25
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 4

        def test_missing_usage_fields_writes_nothing(self, patched_span):
            """Test OpenAI with empty usage dict does not write any token attributes."""
            # Arrange
            usage = {}

            # Act
            openai_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in patched_span.attributes

        def test_zero_token_values_not_written(self, patched_span):
            """Test OpenAI skips zero token values due to falsy `or` guard in implementation."""
            # Arrange — prompt_tokens=0 is falsy so the `or` falls through to input_tokens
            # which is also absent, resulting in None; attribute is not written
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            # Act
            openai_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test OpenAI with only completion_tokens present writes only that attribute."""
            # Arrange
            usage = {"completion_tokens": 42}

            # Act
            openai_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 42
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes

        class TestStreaming:
            """Tests for OpenAI streaming usage accumulation."""

            def _make_streaming_wrapper(self, patched_span):
                """Set up a StreamingWrapper with two chunks and all timing patches applied."""
                from netra.instrumentation.openai.wrappers import StreamingWrapper

                class DummyStream:
                    def __iter__(self):
                        return self

                    def __next__(self):
                        raise StopIteration

                wrapper = StreamingWrapper(span=patched_span, response=DummyStream(), request_kwargs={})
                chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
                chunk2 = {
                    "choices": [{"delta": {"content": " world"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

                with (
                    patch("netra.instrumentation.openai.wrappers.time.time", return_value=123.456),
                    patch("netra.instrumentation.openai.wrappers.record_span_timing"),
                    patch("netra.instrumentation.openai.wrappers.model_as_dict", side_effect=lambda x: x),
                ):
                    wrapper._process_chunk(chunk1)
                    wrapper._process_chunk(chunk2)

                return wrapper

            def test_chunk_accumulation(self, patched_span):
                """Test OpenAI streaming usage is accumulated from chunks into _complete_response."""
                # Arrange + Act
                wrapper = self._make_streaming_wrapper(patched_span)

                # Assert — usage captured from chunk2 into _complete_response
                assert wrapper._complete_response["usage"]["prompt_tokens"] == 10
                assert wrapper._complete_response["usage"]["completion_tokens"] == 5
                assert wrapper._complete_response["usage"]["total_tokens"] == 15

            def test_finalize_writes_span_attributes(self, patched_span):
                """Test OpenAI streaming finalise passes accumulated usage to set_response_attributes."""
                # Arrange
                from netra.instrumentation.openai.utils import set_response_attributes

                wrapper = self._make_streaming_wrapper(patched_span)

                # Act
                with patch("netra.instrumentation.openai.wrappers.set_response_attributes") as mock_set_attr:
                    wrapper._finalize_span()
                    mock_set_attr.assert_called_once()
                    call_args = mock_set_attr.call_args[0]
                    set_response_attributes(patched_span, call_args[1])

                # Assert — final span attributes written correctly
                assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 10
                assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 5

    class TestGroq:
        """Tests for Groq token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test Groq token usage extraction."""
            # Arrange
            usage = {
                "prompt_tokens": 12,
                "completion_tokens": 22,
                "total_tokens": 34,
                "prompt_tokens_details": {"cached_tokens": 6},
            }

            # Act
            groq_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 12
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 22
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 34
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 6

        def test_token_usage_alternative_keys(self, patched_span):
            """Test Groq token usage falls back to input_tokens/output_tokens when primary keys absent."""
            # Arrange
            usage = {
                "input_tokens": 18,
                "output_tokens": 9,
                "input_tokens_details": {"cached_tokens": 4},
            }

            # Act
            groq_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 18
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 9
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4

        def test_zero_token_values_not_written(self, patched_span):
            """Test Groq skips zero token values despite `is not None` guard."""
            # Arrange
            usage = {"prompt_tokens": 0, "completion_tokens": 0}

            # Act
            groq_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_missing_usage_fields_writes_nothing(self, patched_span):
            """Test Groq with empty usage dict does not write any token attributes."""
            # Arrange
            usage = {}

            # Act
            groq_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test Groq with only completion_tokens present writes only that attribute."""
            # Arrange
            usage = {"completion_tokens": 33}

            # Act
            groq_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 33
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes

    class TestCerebras:
        """Tests for Cerebras token usage extraction."""

        def test_token_usage_dict(self, patched_span):
            """Test Cerebras token usage with dictionary-based prompt_tokens_details."""
            # Arrange
            response = Mock()

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 60,
                        "total_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 10},
                    }
                }
                cerebras_set_usage(patched_span, response)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 40
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 60
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 100
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 10

        def test_token_usage_object(self, patched_span):
            """Test Cerebras token usage with object-based prompt_tokens_details."""
            # Arrange
            response = Mock()
            details = Mock()
            details.cached_tokens = 15

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 60,
                        "total_tokens": 100,
                        "prompt_tokens_details": details,
                    }
                }
                cerebras_set_usage(patched_span, response)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 15

        def test_token_usage_object_no_cached_tokens_attr(self, patched_span):
            """Test Cerebras silently skips cached tokens when details object lacks cached_tokens attr."""
            # Arrange
            response = Mock()
            details = Mock(spec=[])  # spec=[] means no attributes defined, hasattr returns False

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 60,
                        "prompt_tokens_details": details,
                    }
                }
                cerebras_set_usage(patched_span, response)

            # Assert — main tokens written, but cache key silently absent
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 40
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 60
            assert SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS not in patched_span.attributes

        def test_missing_usage_key_writes_nothing(self, patched_span):
            """Test Cerebras with no usage key in response dict writes nothing."""
            # Arrange
            response = Mock()

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {}
                cerebras_set_usage(patched_span, response)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_zero_token_values_not_written(self, patched_span):
            """Test Cerebras skips zero token values due to falsy if guard in implementation."""
            # Arrange
            response = Mock()

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
                cerebras_set_usage(patched_span, response)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test Cerebras with only prompt_tokens present writes only that attribute."""
            # Arrange
            response = Mock()

            # Act
            with patch("netra.instrumentation.cerebras.utils.model_as_dict") as mock_model_as_dict:
                mock_model_as_dict.return_value = {"usage": {"prompt_tokens": 25}}
                cerebras_set_usage(patched_span, response)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 25
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

    class TestLiteLLM:
        """Tests for LiteLLM token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test LiteLLM token usage extraction."""
            # Arrange
            usage = {
                "prompt_tokens": 50,
                "completion_tokens": 70,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 10},
            }

            # Act
            litellm_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 50
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 70
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 120
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 20
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 10

        def test_token_usage_alternative_keys(self, patched_span):
            """Test LiteLLM token usage falls back to input_tokens/output_tokens when primary keys absent."""
            # Arrange
            usage = {
                "input_tokens": 90,
                "output_tokens": 45,
                "input_tokens_details": {"cached_tokens": 15},
                "output_tokens_details": {"reasoning_tokens": 8},
            }

            # Act
            litellm_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 90
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 45
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 15
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 8

        def test_missing_usage_fields_writes_nothing(self, patched_span):
            """Test LiteLLM with empty usage dict does not write any token attributes."""
            # Arrange
            usage = {}

            # Act
            litellm_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in patched_span.attributes

        def test_zero_token_values_not_written(self, patched_span):
            """Test LiteLLM skips zero token values due to falsy `or` guard in implementation."""
            # Arrange
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            # Act
            litellm_set_usage(patched_span, usage)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test LiteLLM with only prompt_tokens present writes only that attribute."""
            # Arrange
            usage = {"prompt_tokens": 55}

            # Act
            litellm_set_usage(patched_span, usage)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 55
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

    class TestDSPy:
        """Tests for DSPy token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test DSPy token usage generator yields correct key-value pairs."""
            # Arrange
            response = {"usage": {"prompt_tokens": 80, "completion_tokens": 90, "total_tokens": 170}}

            # Act
            with patch("netra.instrumentation.dspy.utils.convert_to_dict", return_value=response):
                for key, value in dspy_extract_usage(response):
                    patched_span.set_attribute(key, value)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 80
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 90
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 170

        def test_token_usage_alternative_keys(self, patched_span):
            """Test DSPy generator falls back to input_tokens/output_tokens when primary keys absent."""
            # Arrange
            response = {"usage": {"input_tokens": 55, "output_tokens": 22, "total_tokens": 77}}

            # Act
            with patch("netra.instrumentation.dspy.utils.convert_to_dict", return_value=response):
                for key, value in dspy_extract_usage(response):
                    patched_span.set_attribute(key, value)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 55
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 22
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 77

        def test_missing_usage_yields_nothing(self, patched_span):
            """Test DSPy generator yields nothing when usage key is absent."""
            # Arrange
            response = {}

            # Act
            with patch("netra.instrumentation.dspy.utils.convert_to_dict", return_value=response):
                result = dict(dspy_extract_usage(response))

            # Assert
            assert result == {}
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes

        def test_zero_token_values_not_yielded(self, patched_span):
            """Test DSPy skips zero token values despite `is not None` guard."""
            # Arrange
            response = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}

            # Act
            with patch("netra.instrumentation.dspy.utils.convert_to_dict", return_value=response):
                for key, value in dspy_extract_usage(response):
                    patched_span.set_attribute(key, value)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test DSPy with only total_tokens present yields only that key."""
            # Arrange
            response = {"usage": {"total_tokens": 60}}

            # Act
            with patch("netra.instrumentation.dspy.utils.convert_to_dict", return_value=response):
                for key, value in dspy_extract_usage(response):
                    patched_span.set_attribute(key, value)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 60
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

    class TestPydanticAI:
        """Tests for Pydantic AI token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test Pydantic AI maps request_tokens/response_tokens to standard keys."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.request_tokens = 200
            usage.response_tokens = 150
            usage.total_tokens = 350
            usage.requests = 1
            usage.details = {"some_detail": 5}
            response.usage.return_value = usage

            # Act
            with patch("netra.instrumentation.pydantic_ai.utils._safe_get_attribute") as mock_get:

                def side_effect(obj, attr):
                    if obj == usage:
                        return getattr(usage, attr) if hasattr(usage, attr) else usage.details.get(attr)
                    return getattr(obj, attr, None)

                mock_get.side_effect = side_effect
                pydantic_ai_set_usage(patched_span, response)

            # Assert — _safe_set_attribute stringifies values in Pydantic AI utils
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == "200"
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == "150"
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == "350"

        def test_none_tokens_not_written(self, patched_span):
            """Test Pydantic AI writes nothing when all token fields are None."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.request_tokens = None
            usage.response_tokens = None
            usage.total_tokens = None
            usage.requests = None
            usage.details = None
            response.usage.return_value = usage

            # Act
            with patch("netra.instrumentation.pydantic_ai.utils._safe_get_attribute") as mock_get:

                def side_effect(obj, attr):
                    return getattr(obj, attr, None)

                mock_get.side_effect = side_effect
                pydantic_ai_set_usage(patched_span, response)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in patched_span.attributes

        def test_partial_usage_only_present_fields_written(self, patched_span):
            """Test Pydantic AI with only request_tokens present writes only prompt attribute."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.request_tokens = 100
            usage.response_tokens = None
            usage.total_tokens = None
            usage.requests = None
            usage.details = None
            response.usage.return_value = usage

            # Act
            with patch("netra.instrumentation.pydantic_ai.utils._safe_get_attribute") as mock_get:

                def side_effect(obj, attr):
                    return getattr(obj, attr, None)

                mock_get.side_effect = side_effect
                pydantic_ai_set_usage(patched_span, response)

            # Assert
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == "100"
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_dual_attribute_write(self, patched_span):
            """Test Pydantic AI CallToolsNode writes both gen_ai.usage.* and pydantic_ai.usage.* keys."""
            # Arrange

            usage = Mock()
            usage.request_tokens = 75
            usage.response_tokens = 50
            usage.total_tokens = 125
            usage.requests = 1
            usage.details = None

            model_response = Mock()
            model_response.usage = usage
            model_response.parts = []
            model_response.model_name = None
            model_response.timestamp = None

            node = Mock()
            node.model_response = model_response
            node.tool_results = None

            # Act
            _set_call_tools_node_attributes(patched_span, node)

            # Assert — standard OTel keys
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == "75"
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == "50"
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == "125"
            # Assert — pydantic_ai-specific dual-write keys written alongside OTel keys
            assert patched_span.attributes["pydantic_ai.usage.request_tokens"] == "75"
            assert patched_span.attributes["pydantic_ai.usage.response_tokens"] == "50"
            assert patched_span.attributes["pydantic_ai.usage.total_tokens"] == "125"

    class TestGoogleGenAI:
        """Tests for Google GenAI token usage extraction."""

        def test_token_usage(self, patched_span):
            """Test Google GenAI sums candidates and thoughts for completion tokens."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.total_token_count = 500
            usage.candidates_token_count = 100
            usage.thoughts_token_count = 50
            usage.cached_content_token_count = 200
            usage.prompt_token_count = 350

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=usage):
                google_genai_set_usage(patched_span, response)

            # Assert — completion = candidates (100) + thoughts (50) = 150
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 500
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 150
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 200
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 350

        def test_only_candidates_no_thoughts(self, patched_span):
            """Test Google GenAI completion tokens equals candidates alone when thoughts absent."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.total_token_count = 200
            usage.candidates_token_count = 80
            usage.thoughts_token_count = None
            usage.cached_content_token_count = None
            usage.prompt_token_count = 120

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=usage):
                google_genai_set_usage(patched_span, response)

            # Assert — completion = 80 + 0 = 80
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 80

        def test_no_candidates_no_thoughts_skips_completion(self, patched_span):
            """Test Google GenAI skips completion tokens when both candidates and thoughts are absent."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.total_token_count = None
            usage.candidates_token_count = None
            usage.thoughts_token_count = None
            usage.cached_content_token_count = None
            usage.prompt_token_count = None

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=usage):
                google_genai_set_usage(patched_span, response)

            # Assert — output sum is 0, so completion key not written
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_missing_usage_metadata_writes_nothing(self, patched_span):
            """Test Google GenAI writes nothing when _extract_usage_metadata returns None."""
            # Arrange
            response = Mock()

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=None):
                google_genai_set_usage(patched_span, response)

            # Assert
            assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes
            assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in patched_span.attributes

        def test_zero_completion_tokens_not_written(self, patched_span):
            """Test Google GenAI skips completion tokens when candidates and thoughts both equal zero."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.total_token_count = 0
            usage.candidates_token_count = 0
            usage.thoughts_token_count = 0
            usage.cached_content_token_count = None
            usage.prompt_token_count = 0

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=usage):
                google_genai_set_usage(patched_span, response)

            # Assert — output sum is 0 + 0 = 0, `if output > 0` guard skips write
            assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in patched_span.attributes

        def test_cached_tokens_not_written_when_absent(self, patched_span):
            """Test Google GenAI skips cache key when cached_content_token_count is None."""
            # Arrange
            response = Mock()
            usage = Mock()
            usage.total_token_count = 100
            usage.candidates_token_count = 60
            usage.thoughts_token_count = None
            usage.cached_content_token_count = None
            usage.prompt_token_count = 40

            # Act
            with patch("netra.instrumentation.google_genai.utils._extract_usage_metadata", return_value=usage):
                google_genai_set_usage(patched_span, response)

            # Assert — main tokens written, cache key absent
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 40
            assert patched_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 60
            assert SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS not in patched_span.attributes
