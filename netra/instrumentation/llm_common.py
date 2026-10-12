"""Shared helpers for the LLM provider instrumentations.

Two concerns live here:

1. **Request values that must never break the user's call.** Stainless-generated
   SDKs (OpenAI, Groq, Cerebras, ...) pass sentinel objects such as ``NOT_GIVEN``
   and ``omit`` for unset parameters, and some parameters are pydantic models.
   :func:`is_unset_sentinel` and :func:`to_jsonable` turn those into values that
   are safe to put on a span.

2. **Stream lifecycle.** :class:`LLMStream` and :class:`AsyncLLMStream` wrap a
   provider stream so that the span is ended exactly once, whichever way the
   stream finishes: full consumption, early ``break`` followed by ``close()`` or
   garbage collection, a ``with`` block, or an error raised mid-stream (which
   marks the span as ERROR). Providers subclass them and implement
   ``_process_chunk`` and ``_apply_response_attributes``.

   Telemetry code never raises into the user's iteration: chunk processing and
   finalisation errors are logged at DEBUG and swallowed. Only exceptions from
   the underlying stream propagate.
"""

import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

from netra.instrumentation.span_utils import record_span_timing

logger = logging.getLogger(__name__)

TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"

# Class names of the "parameter not provided" sentinels used by Stainless-generated SDKs
# (openai.NOT_GIVEN / openai.omit, and the groq / cerebras equivalents).
_UNSET_SENTINEL_CLASS_NAMES = frozenset({"NotGiven", "Omit"})


def is_unset_sentinel(value: Any) -> bool:
    """Return True if ``value`` is an SDK sentinel meaning "parameter not provided"."""
    return type(value).__name__ in _UNSET_SENTINEL_CLASS_NAMES


def to_jsonable(value: Any) -> Any:
    """Convert a request parameter into a JSON-serialisable value.

    Returns None for None and for unset sentinels, so callers can skip the
    attribute. Pydantic models are dumped without their None fields. Values that
    are already JSON types pass through, and anything else falls back to ``str``.
    """
    if value is None or is_unset_sentinel(value):
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except Exception:
            logger.debug("model_dump failed for %s", type(value).__name__, exc_info=True)
    if isinstance(value, (dict, list, tuple, str, int, float, bool)):
        return value
    return str(value)


def _safe_index(value: Any) -> int:
    """Parse a choice index from a chunk, treating missing or malformed values as 0."""
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


class BaseLLMStream(ObjectProxy):  # type: ignore[misc]
    """State and finalisation shared by the sync and async stream wrappers.

    All per-wrapper state uses wrapt's ``_self_`` prefix so it is stored on the
    proxy rather than written onto the wrapped SDK stream object.
    """

    _netra_stream_wrapper = True

    def __init__(self, span: Span, response: Any, request_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(response)
        self._self_span = span
        self._self_request_kwargs = request_kwargs if isinstance(request_kwargs, dict) else {}
        self._self_complete_response: Dict[str, Any] = {"choices": [], "model": ""}
        self._self_first_content_recorded = False
        self._self_span_ended = False
        self._self_netra_output: Optional[str] = None

    # ``stream_utils`` reads ``_netra_output`` to capture the root output. It must
    # stay None until the stream is finalised.
    @property
    def _netra_output(self) -> Optional[str]:
        return self._self_netra_output

    @_netra_output.setter
    def _netra_output(self, value: Optional[str]) -> None:
        self._self_netra_output = value

    @property
    def _span(self) -> Span:
        return self._self_span

    @property
    def _span_ended(self) -> bool:
        return self._self_span_ended

    @property
    def _complete_response(self) -> Dict[str, Any]:
        return self._self_complete_response

    # Provider hooks
    def _process_chunk(self, chunk: Any) -> None:
        """Accumulate one chunk. Providers override this."""

    def _apply_response_attributes(self) -> None:
        """Set the provider's response attributes from the accumulated state."""

    def _extract_output_text(self) -> str:
        """Return the plain-text output used for root-span output capture."""
        parts = []
        for choice in self._self_complete_response.get("choices", []):
            message = choice.get("message") or {}
            if content := message.get("content"):
                parts.append(content)
        return "".join(parts)

    # Shared chunk helpers
    def _record_first_token(self) -> None:
        if self._self_first_content_recorded:
            return
        self._self_first_content_recorded = True
        first_token_time = time.time()
        record_span_timing(self._self_span, TIME_TO_FIRST_TOKEN, first_token_time, record_event_timestamp=True)
        record_span_timing(self._self_span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)

    def _ensure_choice(self, index: int) -> Dict[str, Any]:
        choices = self._self_complete_response["choices"]
        while len(choices) <= index:
            if "messages" in self._self_request_kwargs:
                choices.append({"message": {"role": "assistant", "content": ""}})
            else:
                choices.append({"text": ""})
        choice: Dict[str, Any] = choices[index]
        return choice

    def _accumulate_chat_chunk(self, chunk_dict: Dict[str, Any]) -> None:
        """Accumulate a Chat Completions / Completions style chunk."""
        if chunk_dict.get("model"):
            self._self_complete_response["model"] = chunk_dict["model"]

        choices = chunk_dict.get("choices") or []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                accumulated = self._ensure_choice(_safe_index(choice.get("index")))
                delta = choice.get("delta") or {}
                content_piece = None
                if isinstance(delta, dict) and delta.get("content"):
                    content_piece = str(delta["content"])
                elif choice.get("text"):
                    content_piece = str(choice["text"])

                if content_piece:
                    self._record_first_token()
                    message = accumulated.setdefault("message", {"role": "assistant", "content": ""})
                    message["content"] = (message.get("content") or "") + content_piece

                if isinstance(delta, dict) and delta.get("tool_calls"):
                    message = accumulated.setdefault("message", {"role": "assistant", "content": ""})
                    tool_calls = message.setdefault("tool_calls", {})
                    for tc_delta in delta["tool_calls"]:
                        if not isinstance(tc_delta, dict):
                            continue
                        tc_index = _safe_index(tc_delta.get("index"))
                        if tc_index not in tool_calls:
                            tool_calls[tc_index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.get("id"):
                            tool_calls[tc_index]["id"] = tc_delta["id"]
                        if func := tc_delta.get("function"):
                            tool_calls[tc_index]["function"]["name"] += func.get("name") or ""
                            tool_calls[tc_index]["function"]["arguments"] += func.get("arguments") or ""

                if choice.get("finish_reason"):
                    accumulated["finish_reason"] = choice["finish_reason"]

        if isinstance(chunk_dict.get("usage"), dict) and chunk_dict["usage"]:
            self._self_complete_response["usage"] = chunk_dict["usage"]

    def _accumulate_responses_event(self, chunk_dict: Dict[str, Any]) -> None:
        """Accumulate a Responses API stream event.

        Text deltas only mark time-to-first-token; the full output is taken from the
        final ``response.completed`` / ``response.incomplete`` event. Text from every
        message item is concatenated, function calls are kept as output items, and
        reasoning items are ignored.
        """
        if chunk_dict.get("delta"):
            self._record_first_token()

        response = chunk_dict.get("response")
        if not isinstance(response, dict) or response.get("status") not in ("completed", "incomplete"):
            return

        if response.get("model"):
            self._self_complete_response["model"] = response["model"]

        texts = []
        function_calls = []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "function_call":
                function_calls.append(item)
            elif item_type in (None, "message"):
                for part in item.get("content") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])

        if function_calls:
            self._self_complete_response["output"] = function_calls
        if texts:
            self._self_complete_response["choices"] = [{"message": {"role": "assistant", "content": "".join(texts)}}]
        if usage := response.get("usage"):
            self._self_complete_response["usage"] = usage

    # Lifecycle
    def _handle_chunk(self, chunk: Any) -> None:
        try:
            self._process_chunk(chunk)
            self._self_span.add_event("llm.content.completion.chunk")
        except Exception:
            logger.debug("Failed to process stream chunk", exc_info=True)

    def _finalize_span(self, error: Optional[BaseException] = None) -> None:
        """End the span once, recording accumulated attributes and the outcome.

        Safe to call any number of times; only the first call has an effect.
        """
        if self._self_span_ended:
            return
        self._self_span_ended = True
        span = self._self_span
        try:
            try:
                for choice in self._self_complete_response.get("choices", []):
                    message = choice.get("message") or {}
                    if isinstance(message.get("tool_calls"), dict):
                        message["tool_calls"] = [message["tool_calls"][i] for i in sorted(message["tool_calls"])]
                record_span_timing(span, LLM_RESPONSE_DURATION)
                self._apply_response_attributes()
            except Exception:
                logger.debug("Failed to set stream response attributes", exc_info=True)
            try:
                self._self_netra_output = self._extract_output_text()
            except Exception:
                self._self_netra_output = ""
            if error is not None:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                span.set_status(Status(StatusCode.OK))
        except Exception:
            logger.debug("Failed to finalize stream span", exc_info=True)
        finally:
            span.end()

    def _finalize(self) -> None:
        """Idempotent finaliser looked up by ``stream_utils``."""
        self._finalize_span()

    def __del__(self) -> None:
        try:
            self._finalize_span()
        except Exception:
            pass

    @staticmethod
    def _error_from_exit(exc_val: Any) -> Optional[BaseException]:
        # Only real errors mark the span as failed; GeneratorExit / cancellation do not.
        return exc_val if isinstance(exc_val, Exception) else None


class LLMStream(BaseLLMStream):
    """Sync stream wrapper. Subclasses implement ``_process_chunk`` and ``_apply_response_attributes``."""

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finalize_span()
            raise
        except Exception as e:
            self._finalize_span(error=e)
            raise
        self._handle_chunk(chunk)
        return chunk

    def __enter__(self) -> "LLMStream":
        enter = getattr(self.__wrapped__, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            exit_ = getattr(self.__wrapped__, "__exit__", None)
            if callable(exit_):
                return exit_(exc_type, exc_val, exc_tb)
            return None
        finally:
            self._finalize_span(error=self._error_from_exit(exc_val))

    def close(self) -> None:
        """Close the stream and end the span."""
        self._finalize_span()
        close = getattr(self.__wrapped__, "close", None)
        if callable(close):
            close()


class AsyncLLMStream(BaseLLMStream):
    """Async stream wrapper. Subclasses implement ``_process_chunk`` and ``_apply_response_attributes``."""

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finalize_span()
            raise
        except Exception as e:
            self._finalize_span(error=e)
            raise
        self._handle_chunk(chunk)
        return chunk

    async def __aenter__(self) -> "AsyncLLMStream":
        enter = getattr(self.__wrapped__, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            exit_ = getattr(self.__wrapped__, "__aexit__", None)
            if callable(exit_):
                return await exit_(exc_type, exc_val, exc_tb)
            return None
        finally:
            self._finalize_span(error=self._error_from_exit(exc_val))

    async def aclose(self) -> None:
        """Close the stream and end the span."""
        self._finalize_span()
        aclose = getattr(self.__wrapped__, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        close = getattr(self.__wrapped__, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def close(self) -> None:
        """Alias for :meth:`aclose`, matching ``openai.AsyncStream.close``."""
        await self.aclose()
