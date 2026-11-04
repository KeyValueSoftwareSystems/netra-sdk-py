"""``trace_content=False`` must keep prompt/completion content off every Netra span (audit H-4).

Each case drives one instrumentor's attribute-setting code with a sentinel string in
every prompt, completion, tool argument, transcript and system prompt, then checks
the exported span:

- content off: the sentinel appears in no attribute and no event, but the metadata
  the case lists (model, usage, ...) is still there;
- content on: the sentinel *is* there, which proves the case reaches the code path.

Fakes and in-memory exporters only — no provider SDK is needed.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from netra import config as netra_config
from netra.instrumentation.span_utils import CONTENT_TRACING_OVERRIDE_KEY, should_capture_content

SENTINEL = "SECRET-SENTINEL-123"


# Fixtures
@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


@pytest.fixture
def tracer(provider: TracerProvider) -> Any:
    return provider.get_tracer("test.trace_content")


def _set_trace_content(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Register an active config the way ``Netra.init`` does, restored after the test."""
    fake_config = SimpleNamespace(trace_content=enabled, attribute_max_len=50_000)
    monkeypatch.setattr(netra_config, "_active_config", fake_config)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true" if enabled else "false")


@pytest.fixture(params=[False, True], ids=["content_off", "content_on"])
def capture(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    _set_trace_content(monkeypatch, request.param)
    return bool(request.param)


def _span_text(span: ReadableSpan) -> str:
    """Every attribute and event attribute on *span*, as one searchable string."""
    parts: List[Any] = [dict(span.attributes or {})]
    for event in span.events:
        parts.append({"event": event.name, "attributes": dict(event.attributes or {})})
    return json.dumps(parts, default=str)


def _all_spans_text(exporter: InMemorySpanExporter) -> str:
    return "\n".join(_span_text(span) for span in exporter.get_finished_spans())


def _all_attributes(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for span in exporter.get_finished_spans():
        merged.update(dict(span.attributes or {}))
    return merged


def _assert_gate(exporter: InMemorySpanExporter, capture: bool, metadata: Tuple[str, ...]) -> None:
    assert exporter.get_finished_spans(), "the case exported no span"
    text = _all_spans_text(exporter)
    if capture:
        assert SENTINEL in text, "the case never reached a content attribute"
    else:
        assert SENTINEL not in text, f"content leaked with trace_content=False: {text}"
    attributes = _all_attributes(exporter)
    for key in metadata:
        if key.startswith("*"):  # suffix match: the index of e.g. a finish reason follows the messages
            assert any(k.endswith(key[1:]) for k in attributes), f"metadata attribute {key!r} must not be gated"
        else:
            assert key in attributes, f"metadata attribute {key!r} must not be gated"


# Cases: each drives one instrumentor with SENTINEL as content and returns the
# metadata keys that must survive with content off.
def _case_openai(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.openai.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("openai.chat") as span:
        set_request_attributes(
            span,
            {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": SENTINEL},
                    {"role": "user", "content": SENTINEL},
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": SENTINEL}}],
                    },
                ],
            },
            "chat",
        )
        set_response_attributes(
            span,
            {
                "model": "gpt-test",
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": SENTINEL,
                            "tool_calls": [{"id": "c2", "function": {"name": "f", "arguments": SENTINEL}}],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    with tracer.start_as_current_span("openai.response") as span:
        set_request_attributes(
            span,
            {
                "model": "gpt-test",
                "instructions": SENTINEL,
                "input": [
                    {"role": "user", "content": SENTINEL},
                    {"type": "function_call", "name": "f", "arguments": SENTINEL},
                    {"type": "function_call_output", "output": SENTINEL},
                ],
            },
            "response",
        )
        set_response_attributes(
            span,
            {
                "model": "gpt-test",
                "output_text": SENTINEL,
                "output": [
                    {"type": "function_call", "name": "f", "arguments": SENTINEL},
                    {"type": "message", "content": [{"type": "output_text", "text": SENTINEL}]},
                ],
            },
        )
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens", "*.finish_reason")


def _case_litellm(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.litellm.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("litellm.chat") as span:
        set_request_attributes(span, {"model": "m", "messages": [{"role": "user", "content": SENTINEL}]}, "chat")
        set_response_attributes(
            span,
            {
                "model": "m",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "choices": [{"message": {"role": "assistant", "content": SENTINEL}, "finish_reason": "stop"}],
            },
        )
    with tracer.start_as_current_span("litellm.response") as span:
        set_request_attributes(span, {"model": "m", "instructions": SENTINEL, "input": SENTINEL}, "response")
        set_response_attributes(
            span,
            {
                "model": "m",
                "output_text": SENTINEL,
                "output": [{"type": "function_call", "name": "f", "arguments": SENTINEL}],
                "choices": [{"delta": {"role": "assistant", "content": SENTINEL}}],
            },
        )
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens", "*.finish_reason")


def _case_groq(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.groq.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("groq.chat") as span:
        set_request_attributes(
            span, {"model": "m", "messages": [{"role": "user", "content": SENTINEL}], "prompt": SENTINEL}, "chat"
        )
        set_response_attributes(
            span,
            {
                "model": "m",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "choices": [
                    {"message": {"role": "assistant", "content": SENTINEL}, "finish_reason": "stop"},
                    {"delta": {"role": "assistant", "content": SENTINEL}},
                ],
            },
        )
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens", "*.finish_reason")


def _case_cerebras(tracer: Any) -> Tuple[str, ...]:
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    from netra.instrumentation.libraries.cerebras.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("cerebras.chat") as span:
        set_request_attributes(
            span, LLMRequestTypeValues.CHAT, {"model": "m", "messages": [{"role": "user", "content": SENTINEL}]}
        )
        set_response_attributes(
            span,
            {
                "id": "r1",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                "choices": [{"message": {"role": "assistant", "content": SENTINEL}, "finish_reason": "stop"}],
            },
        )
    with tracer.start_as_current_span("cerebras.completion") as span:
        set_request_attributes(span, LLMRequestTypeValues.COMPLETION, {"model": "m", "prompt": SENTINEL})
        set_response_attributes(span, {"choices": [{"text": SENTINEL, "finish_reason": "stop"}]})
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens", "*.finish_reason")


def _case_google_genai(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.google_genai.utils import set_request_attributes, set_response_attributes

    usage = SimpleNamespace(
        total_token_count=3,
        candidates_token_count=2,
        thoughts_token_count=None,
        cached_content_token_count=None,
        prompt_token_count=1,
    )
    response = SimpleNamespace(
        usage_metadata=usage,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=SENTINEL)]))],
    )
    with tracer.start_as_current_span("gemini.generate") as span:
        set_request_attributes(span, (), {"model": "gemini-test", "contents": SENTINEL})
        set_response_attributes(span, response)
    with tracer.start_as_current_span("gemini.generate.parts") as span:
        content = SimpleNamespace(role="user", parts=[SimpleNamespace(text=SENTINEL)])
        set_request_attributes(span, (), {"model": "gemini-test", "contents": [content]})
        set_response_attributes(span, SENTINEL)
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens")


def _case_claude_agent_sdk(tracer: Any) -> Tuple[str, ...]:
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions

    from netra.instrumentation.libraries.claude_agent_sdk import utils

    with tracer.start_as_current_span("claude-agent") as span:
        utils.set_request_attributes(span, SENTINEL, ClaudeAgentOptions(model="claude-test", system_prompt=SENTINEL))
        result = SimpleNamespace(
            result=SENTINEL,
            structured_output={"answer": SENTINEL},
            num_turns=1,
            session_id="s1",
            stop_reason="end_turn",
            is_error=False,
            permission_denials=[{"tool_input": SENTINEL}],
            errors=None,
            model_usage=None,
            usage=None,
        )
        utils.set_result_message_attributes(tracer, span, result)
    return ("gen_ai.request.model", "gen_ai.claude_code.num_turns")


def _case_pydantic_ai(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.pydantic_ai import utils

    with tracer.start_as_current_span("pydantic_ai.run") as span:
        usage = SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3, details=None)
        utils.set_pydantic_response_attributes(span, SimpleNamespace(model_name="m", usage=usage, output=SENTINEL))
        utils._set_assistant_response_content(span, SimpleNamespace(output=SENTINEL))
        utils._set_user_prompt_node_attributes(
            span, SimpleNamespace(user_prompt=SENTINEL, instructions=SENTINEL, system_prompts=[SENTINEL])
        )
        part = SimpleNamespace(content=SENTINEL, role="user", args=SENTINEL, tool_name="t", tool_call_id="c")
        utils._set_model_request_node_attributes(span, SimpleNamespace(request=SimpleNamespace(parts=[part])))
        utils._set_call_tools_node_attributes(
            span,
            SimpleNamespace(
                model_response=SimpleNamespace(parts=[part], usage=None, model_name="m", timestamp=None),
                tool_results=[SENTINEL],
            ),
        )
        utils._set_end_node_attributes(span, SimpleNamespace(data=SimpleNamespace(output=SENTINEL)))
        utils._set_generic_node_attributes(span, SimpleNamespace(content=SENTINEL, message=SENTINEL))
    return ("gen_ai.response.model", "gen_ai.usage.prompt_tokens", "pydantic_ai.request.parts_count")


class _FakeLM:
    model = "openai/gpt-test"
    kwargs: Dict[str, Any] = {"temperature": 0.0}

    def __call__(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        return [SENTINEL]


def _case_dspy(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.dspy.wrappers import LMCallWrapper, ToolCallWrapper

    lm = _FakeLM()
    LMCallWrapper(tracer)(
        lm.__call__, lm, (), {"prompt": SENTINEL, "messages": [{"role": "user", "content": SENTINEL}]}
    )
    tool = SimpleNamespace(name="lookup", desc="looks things up")
    ToolCallWrapper(tracer)(lambda **kw: SENTINEL, tool, (), {"query": SENTINEL})
    return ("gen_ai.request.model", "dspy.tool.name")


def _case_agno(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.agno import utils

    agent = SimpleNamespace(name="agent", instructions=SENTINEL, model=SimpleNamespace(id="m"))
    message = SimpleNamespace(role="user", content=SENTINEL)
    with tracer.start_as_current_span("agno.agent") as span:
        utils.set_request_attributes(span, agent, (SENTINEL,), {}, "agent")
        utils.set_response_attributes(span, SimpleNamespace(content=SENTINEL, run_id="r1"))
        utils.set_llm_prompt_attributes(span, [message])
        utils.set_llm_completion_attributes(span, json.dumps([{"role": "assistant", "content": SENTINEL}]))
        utils.update_active_span_with_system_prompt([SimpleNamespace(role="system", content=SENTINEL), message])
        span.set_attributes(utils.extract_memory_attributes(SimpleNamespace(), (SENTINEL,), "add_user_memory"))
        utils.set_agentos_request_input(span, {"method": "POST", "path": "/runs", "headers": []}, SENTINEL.encode())
    return ("gen_ai.agent.name", "gen_ai.agent.model")


def _case_google_adk(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.google_adk import utils

    request = {
        "model": "gemini-test",
        "config": {"temperature": 0.1, "system_instruction": SENTINEL},
        "contents": [
            {"role": "user", "parts": [{"text": SENTINEL}]},
            {"role": "model", "parts": [{"function_call": {"name": "f", "args": {"q": SENTINEL}}}]},
            {"role": "user", "parts": [{"function_response": {"name": "f", "response": {"r": SENTINEL}}}]},
        ],
    }
    func_call = SimpleNamespace(id="c1", name="f", args={"q": SENTINEL})
    last = SimpleNamespace(
        content=SimpleNamespace(
            parts=[SimpleNamespace(function_call=func_call), SimpleNamespace(function_call=None, text=SENTINEL)]
        ),
        model_version="gemini-test",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=2,
            thoughts_token_count=None,
            total_token_count=3,
            cached_content_token_count=None,
        ),
        actions=SimpleNamespace(transfer_to_agent=None, state_delta={"k": SENTINEL}, escalate=None),
    )
    agent = SimpleNamespace(
        name="a", description="d", model="gemini-test", instruction=SENTINEL, tools=[], sub_agents=[]
    )
    with tracer.start_as_current_span("adk.llm") as span:
        span.set_attributes(utils.extract_llm_request_attributes(request))
        span.set_attributes(utils.extract_llm_response_attributes(last, []))
        span.set_attributes(utils.extract_agent_attributes(agent))
    return ("gen_ai.request.model", "gen_ai.usage.prompt_tokens", "gen_ai.completions.0.tool_calls.0.name")


def _case_honcho(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.honcho import utils

    with tracer.start_as_current_span("honcho.chat") as span:
        utils._safe_set(span, "gen_ai.provider.name", "honcho")
        utils._set_input(span, {"query": SENTINEL})
        utils._set_output(span, SENTINEL)
        utils._set_content(span, "gen_ai.memory.query.text", SENTINEL)
    return ("gen_ai.provider.name",)


def _case_livekit(tracer: Any, provider: TracerProvider) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.livekit.trace_processor import SpanMappingProcessor
    from netra.instrumentation.libraries.livekit.utils import LIVEKIT_SCOPE_NAME

    provider.add_span_processor(SpanMappingProcessor())
    livekit_tracer = provider.get_tracer(LIVEKIT_SCOPE_NAME)
    with livekit_tracer.start_as_current_span("agent_turn") as span:
        span.set_attribute("lk.instructions", SENTINEL)
        span.set_attribute("lk.user_input", SENTINEL)
        span.set_attribute("lk.response.text", SENTINEL)
        span.set_attribute("lk.response.ttft", 0.25)
        span.set_attributes({"lk.user_transcript": SENTINEL, "lk.chat_ctx": json.dumps({"items": []})})
        span.add_event("gen_ai.user.message", {"content": SENTINEL})
        span.add_event("gen_ai.choice", {"content": SENTINEL})
    with livekit_tracer.start_as_current_span("function_tool") as span:
        span.set_attribute("lk.function_tool.name", "lookup")
        span.set_attribute("lk.function_tool.arguments", SENTINEL)
        span.set_attribute("lk.function_tool.output", SENTINEL)
    return ("netra.latency.ttft", "lk.function_tool.name")


def _case_elevenlabs(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.elevenlabs.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("elevenlabs.tts") as span:
        set_request_attributes(
            span, {"text": SENTINEL, "model_id": "eleven-test", "previous_text": SENTINEL, "next_text": SENTINEL}
        )
        set_response_attributes(span, SimpleNamespace(text=SENTINEL, transcription_id="t1"))
    return ("gen_ai.request.model", "gen_ai.usage.prompt.character_count")


def _case_cartesia(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.cartesia.utils import set_request_attributes, set_response_attributes

    with tracer.start_as_current_span("cartesia.tts") as span:
        set_request_attributes(span, {"model_id": "sonic-test", "transcript": SENTINEL})
        set_response_attributes(span, SimpleNamespace(text=SENTINEL, duration=60))
    return ("gen_ai.request.model", "gen_ai.usage.prompt.character_count")


def _case_deepgram(tracer: Any) -> Tuple[str, ...]:
    from netra.instrumentation.libraries.deepgram.utils import set_request_attributes, set_response_attributes

    alternative = SimpleNamespace(transcript=SENTINEL)
    results = SimpleNamespace(
        channels=[SimpleNamespace(alternatives=[alternative])],
        summary=SimpleNamespace(result="success", text=SENTINEL, short=SENTINEL),
        topics=None,
        intents=None,
        sentiments=None,
    )
    response = SimpleNamespace(type=None, results=results, metadata=SimpleNamespace(request_id="req-1", duration=60))
    with tracer.start_as_current_span("deepgram.listen") as span:
        set_request_attributes(span, {"text": SENTINEL, "model": "nova-test"})
        set_response_attributes(span, response)
    with tracer.start_as_current_span("deepgram.ws") as span:
        set_response_attributes(span, SimpleNamespace(type="TurnInfo", transcript=SENTINEL, request_id="req-2"))
        set_response_attributes(span, SimpleNamespace(type="ConversationText", role="user", content=SENTINEL))
        set_response_attributes(span, SimpleNamespace(type="History", role="user", content=SENTINEL))
    return ("gen_ai.request.model", "gen_ai.response.request_id")


CASES: Dict[str, Callable[..., Tuple[str, ...]]] = {
    "openai": _case_openai,
    "litellm": _case_litellm,
    "groq": _case_groq,
    "cerebras": _case_cerebras,
    "google_genai": _case_google_genai,
    "claude_agent_sdk": _case_claude_agent_sdk,
    "pydantic_ai": _case_pydantic_ai,
    "dspy": _case_dspy,
    "agno": _case_agno,
    "google_adk": _case_google_adk,
    "honcho": _case_honcho,
    "elevenlabs": _case_elevenlabs,
    "cartesia": _case_cartesia,
    "deepgram": _case_deepgram,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_content_is_gated(name: str, capture: bool, tracer: Any, exporter: InMemorySpanExporter) -> None:
    metadata = CASES[name](tracer)
    _assert_gate(exporter, capture, metadata)


def test_livekit_content_is_gated(
    capture: bool, provider: TracerProvider, tracer: Any, exporter: InMemorySpanExporter
) -> None:
    metadata = _case_livekit(tracer, provider)
    _assert_gate(exporter, capture, metadata)


def test_livekit_transcript_buffer_ignores_content_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from netra.instrumentation.libraries.livekit import trace_processor

    _set_trace_content(monkeypatch, False)

    class _TurnSpan:
        pass

    turn_span = _TurnSpan()
    monkeypatch.setitem(trace_processor._user_turn_spans, 4242, turn_span)
    monkeypatch.setattr(trace_processor, "_user_transcripts", {})
    trace_processor.record_user_transcript(4242, SENTINEL, is_final=True)
    assert trace_processor._user_transcripts == {}


# The three instrumentors that already honoured the flag now use the shared gate.
@pytest.mark.parametrize(
    "module",
    [
        "netra.instrumentation.libraries.cohere",
        "netra.instrumentation.libraries.mistralai",
        "netra.instrumentation.libraries.hermes_agent.utils",
    ],
)
def test_legacy_gates_use_the_shared_helper(module: str) -> None:
    imported = pytest.importorskip(module)
    assert imported.should_send_prompts is should_capture_content


# should_capture_content semantics
def test_reads_the_active_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_trace_content(monkeypatch, False)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")  # the active config wins
    assert should_capture_content() is False
    _set_trace_content(monkeypatch, True)
    assert should_capture_content() is True


def test_falls_back_to_the_env_var_before_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(netra_config, "_active_config", None)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    assert should_capture_content() is False
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT")
    assert should_capture_content() is True


def test_context_override_force_enables_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_trace_content(monkeypatch, False)
    token = context_api.attach(context_api.set_value(CONTENT_TRACING_OVERRIDE_KEY, True))
    try:
        assert should_capture_content() is True
    finally:
        context_api.detach(token)
    assert should_capture_content() is False


# DSPy wrappers used to swallow the user's exception and return None.
class _BoomError(Exception):
    pass


class _RaisingLM(_FakeLM):
    def __call__(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        raise _BoomError("provider failed")

    async def acall(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        raise _BoomError("provider failed")


def test_dspy_lm_call_reraises_the_provider_error(tracer: Any, exporter: InMemorySpanExporter) -> None:
    from netra.instrumentation.libraries.dspy.wrappers import LMCallWrapper

    lm = _RaisingLM()
    with pytest.raises(_BoomError):
        LMCallWrapper(tracer)(lm.__call__, lm, (), {"prompt": "hi"})
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_dspy_async_lm_call_reraises_the_provider_error(tracer: Any, exporter: InMemorySpanExporter) -> None:
    from netra.instrumentation.libraries.dspy.wrappers import LMAsyncCallWrapper

    lm = _RaisingLM()
    with pytest.raises(_BoomError):
        asyncio.run(LMAsyncCallWrapper(tracer)(lm.acall, lm, (), {"prompt": "hi"}))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("wrapper_name", ["ModuleForwardWrapper", "ModuleForwardSyncWrapper", "ToolCallWrapper"])
def test_dspy_module_and_tool_wrappers_reraise(wrapper_name: str, tracer: Any) -> None:
    from netra.instrumentation.libraries.dspy import wrappers

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise _BoomError("user code failed")

    instance = SimpleNamespace(name="tool", desc="")
    wrapper = getattr(wrappers, wrapper_name)(tracer)
    with pytest.raises(_BoomError):
        wrapper(failing, instance, (), {"x": 1})


def test_dspy_telemetry_failure_does_not_break_the_call(
    tracer: Any, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from netra.instrumentation.libraries.dspy import wrappers

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("telemetry bug")

    monkeypatch.setattr(wrappers, "get_llm_invocation_parameters", broken)
    monkeypatch.setattr(wrappers, "extract_usage_info", broken)
    lm = _FakeLM()
    assert wrappers.LMCallWrapper(tracer)(lm.__call__, lm, (), {"prompt": "hi"}) == [SENTINEL]
    assert len(exporter.get_finished_spans()) == 1
