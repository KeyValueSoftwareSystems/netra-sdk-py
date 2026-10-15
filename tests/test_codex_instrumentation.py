"""
Unit tests for NetraCodexInstrumentor.

The real `openai-codex` package is not a test dependency: minimal stand-in
modules for `openai_codex.client` and `openai_codex.api` are registered in
sys.modules before instrumenting, mirroring the SDK's call signatures. The
notification sequences are modelled on a recorded run of openai-codex 0.162.1.
"""

import asyncio
import json
import sys
import types
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterator, List, Optional

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from netra.instrumentation.libraries.codex import NetraCodexInstrumentor
from netra.instrumentation.libraries.codex import utils as codex_utils

THREAD_ID = "thread-1"
MODEL = "gpt-test"


# --------------------------------------------------------------------------------------
# Notification builders (shaped like the SDK's pydantic models: snake_case fields,
# ThreadItem as a RootModel with the variant in `.root`)
# --------------------------------------------------------------------------------------


def _note(method: str, **payload: Any) -> NS:
    return NS(method=method, payload=NS(**payload))


def _item(**fields: Any) -> NS:
    return NS(root=NS(**fields))


def turn_started(turn_id: str) -> NS:
    return _note("turn/started", thread_id=THREAD_ID, turn=NS(id=turn_id, status="inProgress"))


def item_started(turn_id: str, at_ms: int, **fields: Any) -> NS:
    return _note("item/started", thread_id=THREAD_ID, turn_id=turn_id, started_at_ms=at_ms, item=_item(**fields))


def item_completed(turn_id: str, at_ms: int, **fields: Any) -> NS:
    return _note("item/completed", thread_id=THREAD_ID, turn_id=turn_id, completed_at_ms=at_ms, item=_item(**fields))


def usage(turn_id: str, last_total: int, input_tokens: int, output_tokens: int, cumulative: int) -> NS:
    last = NS(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=last_total,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_output_tokens=0,
    )
    total = NS(**{**vars(last), "total_tokens": cumulative})
    return _note(
        "thread/tokenUsage/updated",
        thread_id=THREAD_ID,
        turn_id=turn_id,
        token_usage=NS(last=last, total=total, model_context_window=475000),
    )


def turn_completed(turn_id: str, status: str = "completed", error: Optional[str] = None) -> NS:
    return _note(
        "turn/completed",
        thread_id=THREAD_ID,
        turn=NS(id=turn_id, status=status, duration_ms=5457, error=NS(message=error) if error else None),
    )


def shell_turn(turn_id: str, exit_code: int = 0, cumulative_before: int = 0) -> List[NS]:
    """A turn that runs one shell command, like the recorded `ls -la` turn."""
    cmd = dict(type="commandExecution", id=f"exec-{turn_id}", command="/bin/bash -lc 'ls -la'", cwd="/work")
    return [
        turn_started(turn_id),
        item_started(turn_id, 1000, type="userMessage", id="u1", content=[NS(type="text", text="list files")]),
        item_completed(turn_id, 1000, type="userMessage", id="u1", content=[NS(type="text", text="list files")]),
        item_started(turn_id, 2000, type="agentMessage", id="m1", phase="commentary", text=""),
        item_completed(turn_id, 2400, type="agentMessage", id="m1", phase="commentary", text="Running ls"),
        item_started(turn_id, 3000, status="inProgress", **cmd),
        item_completed(
            turn_id,
            3500,
            status="completed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            aggregated_output="total 8\nhello.txt\n",
            duration_ms=500,
            source="unifiedExecStartup",
            **cmd,
        ),
        usage(turn_id, 13810, 13754, 56, cumulative_before + 13810),
        item_started(turn_id, 4000, type="agentMessage", id="m2", phase="final_answer", text=""),
        item_completed(turn_id, 4200, type="agentMessage", id="m2", phase="final_answer", text="6 entries"),
        usage(turn_id, 14039, 14017, 22, cumulative_before + 27849),
        turn_completed(turn_id),
    ]


# --------------------------------------------------------------------------------------
# Stub openai_codex modules
# --------------------------------------------------------------------------------------


class FakeSubscription:
    def __init__(self, events: List[NS]):
        self._events = list(events)
        self.closed = False

    def next(self) -> NS:
        return self._events.pop(0)

    def close(self) -> None:
        self.closed = True


class CodexClient:
    def __init__(self) -> None:
        self.scripts: Dict[str, List[NS]] = {}
        self.written: List[dict] = []
        self.subscriptions: List[FakeSubscription] = []
        self.next_turn_ids: List[str] = []
        self.start_error: Optional[BaseException] = None
        self._approval_handler = self._default_approval_handler

    def _default_approval_handler(self, method: str, params: Any) -> dict:
        return {"decision": "accept"}

    def _write_message(self, payload: dict) -> None:
        self.written.append(payload)

    def thread_start(self, params: Any = None) -> NS:
        return NS(
            thread=NS(id=THREAD_ID),
            model=MODEL,
            model_provider="openai",
            reasoning_effort=None,
            approval_policy="on-request",
            sandbox=NS(type="workspaceWrite"),
            cwd="/work",
        )

    def thread_resume(self, thread_id: str, params: Any = None) -> NS:
        return self.thread_start(params)

    def thread_fork(self, thread_id: str, params: Any = None) -> NS:
        return self.thread_start(params)

    def _start_turn(self, thread_id: str, input_items: Any, params: Any, for_handle: bool) -> tuple:
        self._write_message({"id": "req-1", "method": "turn/start", "params": {"threadId": thread_id}})
        if self.start_error is not None:
            raise self.start_error
        turn_id = self.next_turn_ids.pop(0)
        subscription = FakeSubscription(self.scripts.get(turn_id, []))
        self.subscriptions.append(subscription)
        return NS(turn=NS(id=turn_id)), (subscription if for_handle else None)

    def close(self) -> None:
        pass


class TurnHandle:
    def __init__(self, client: CodexClient, thread_id: str, id: str, subscription: FakeSubscription):
        self._client, self.thread_id, self.id, self._subscription = client, thread_id, id, subscription

    def stream(self) -> Iterator[NS]:
        try:
            while True:
                event = self._subscription.next()
                yield event
                if event.method == "turn/completed" and event.payload.turn.id == self.id:
                    break
        finally:
            self._subscription.close()

    def run(self) -> str:
        stream = self.stream()
        try:
            final = None
            for event in stream:
                if event.method == "turn/completed" and event.payload.turn.status == "failed":
                    raise RuntimeError(event.payload.turn.error.message)
                if (
                    event.method == "item/completed"
                    and getattr(event.payload.item.root, "phase", None) == "final_answer"
                ):
                    final = event.payload.item.root.text
            return final
        finally:
            stream.close()


class AsyncTurnHandle(TurnHandle):
    async def stream(self):  # type: ignore[override]
        try:
            while True:
                event = await asyncio.to_thread(self._subscription.next)
                yield event
                if event.method == "turn/completed" and event.payload.turn.id == self.id:
                    break
        finally:
            self._subscription.close()


class Thread:
    def __init__(self, client: CodexClient, id: str):
        self._client, self.id = client, id

    def turn(self, input: Any, **params: Any) -> TurnHandle:
        started, subscription = self._client._start_turn(self.id, input, params or None, for_handle=True)
        return TurnHandle(self._client, self.id, started.turn.id, subscription)

    def run(self, input: Any, **params: Any) -> str:
        return self.turn(input, **params).run()

    async def aturn(self, input: Any) -> AsyncTurnHandle:
        started, subscription = await asyncio.to_thread(self._client._start_turn, self.id, input, None, True)
        return AsyncTurnHandle(self._client, self.id, started.turn.id, subscription)


def _install_fake_modules() -> Dict[str, Optional[types.ModuleType]]:
    package = types.ModuleType("openai_codex")
    package.__path__ = []  # type: ignore[attr-defined]
    client_module = types.ModuleType("openai_codex.client")
    client_module.CodexClient = CodexClient  # type: ignore[attr-defined]
    api_module = types.ModuleType("openai_codex.api")
    api_module.TurnHandle = TurnHandle  # type: ignore[attr-defined]
    api_module.AsyncTurnHandle = AsyncTurnHandle  # type: ignore[attr-defined]
    package.client = client_module  # type: ignore[attr-defined]
    package.api = api_module  # type: ignore[attr-defined]
    modules = {"openai_codex": package, "openai_codex.client": client_module, "openai_codex.api": api_module}
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return previous


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    previous = _install_fake_modules()
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    instrumentor = NetraCodexInstrumentor()
    instrumentor._instrument(tracer_provider=provider)
    tracer = provider.get_tracer("test")
    span_exporter.tracer = tracer  # type: ignore[attr-defined]
    try:
        yield span_exporter
    finally:
        instrumentor._uninstrument()
        codex_utils._turns.clear()
        codex_utils._thread_info.clear()
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def client() -> CodexClient:
    return CodexClient()


def _start_thread(client: CodexClient) -> Thread:
    response = client.thread_start()
    return Thread(client, response.thread.id)


def _spans(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _generations(exporter: InMemorySpanExporter, turn_span: Any) -> list:
    return sorted(
        (
            s
            for s in exporter.get_finished_spans()
            if s.name == "codex.generation" and s.parent and s.parent.span_id == turn_span.context.span_id
        ),
        key=lambda s: s.attributes["gen_ai.codex.request_index"],
    )


def _usage_total(exporter: InMemorySpanExporter, turn_span: Any) -> int:
    return sum(s.attributes["llm.usage.total_tokens"] for s in _generations(exporter, turn_span))


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_run_produces_one_turn_span_with_tool_and_usage_children(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    assert thread.run("list files") == "6 entries"

    turns = _spans(exporter, "codex.turn")
    assert len(turns) == 1
    turn = turns[0]
    attrs = turn.attributes
    assert attrs["netra.span.type"] == "AGENT"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.codex.thread_id"] == THREAD_ID
    assert attrs["gen_ai.codex.turn_id"] == "t1"
    assert attrs["gen_ai.codex.turn.status"] == "completed"
    assert attrs["gen_ai.codex.sandbox"] == "workspaceWrite"
    assert attrs["gen_ai.codex.auto_approve"] is True
    assert '"list files"' in attrs["input"]
    assert '"6 entries"' in attrs["output"]
    assert "gen_ai.codex.turn.stream_closed_early" not in attrs
    assert turn.status.status_code == StatusCode.UNSET

    (shell,) = _spans(exporter, "shell")
    assert shell.parent.span_id == turn.context.span_id
    assert shell.attributes["netra.span.type"] == "TOOL"
    assert shell.attributes["gen_ai.codex.command.exit_code"] == 0
    assert shell.attributes["output"] == "total 8\nhello.txt\n"
    assert shell.start_time == 3000 * 1_000_000
    assert shell.end_time == 3500 * 1_000_000

    assert _usage_total(exporter, turn) == 13810 + 14039


def test_each_model_request_is_a_generation_span(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    thread.run("list files")

    (turn,) = _spans(exporter, "codex.turn")
    first, second = _generations(exporter, turn)
    assert [s.name for s in exporter.get_finished_spans() if s.name.startswith("codex.usage")] == []
    for generation in (first, second):
        assert generation.attributes["netra.span.type"] == "GENERATION"
        assert generation.attributes["gen_ai.request.model"] == MODEL
        assert generation.attributes["llm.request.type"] == "response"

    # Request 1: user input -> commentary + the shell tool call.
    assert json.loads(first.attributes["input"]) == [{"role": "user", "content": "list files"}]
    output = json.loads(first.attributes["output"])
    assert output[0] == {"role": "assistant", "content": "Running ls"}
    assert output[1]["tool_call_id"] == "exec-t1"
    assert json.loads(output[1]["content"]) == {"name": "shell", "arguments": {"command": "/bin/bash -lc 'ls -la'"}}
    assert first.attributes["gen_ai.usage.prompt_tokens"] == 13754
    assert first.attributes["gen_ai.usage.completion_tokens"] == 56
    assert first.attributes["llm.usage.total_tokens"] == 13810
    assert first.start_time <= first.end_time == 3000 * 1_000_000

    # Request 2: transcript including the tool result -> final answer.
    second_input = json.loads(second.attributes["input"])
    assert second_input[-1] == {"role": "tool", "content": "total 8\nhello.txt\n", "tool_call_id": "exec-t1"}
    assert json.loads(second.attributes["output"]) == [{"role": "assistant", "content": "6 entries"}]
    assert second.attributes["llm.usage.total_tokens"] == 14039
    assert second.start_time == 3500 * 1_000_000
    assert second.end_time == 4200 * 1_000_000


def test_second_turn_usage_excludes_first_turn(exporter, client) -> None:
    client.next_turn_ids = ["t1", "t2"]
    client.scripts["t1"] = shell_turn("t1")
    client.scripts["t2"] = shell_turn("t2", cumulative_before=27849)
    thread = _start_thread(client)

    thread.run("one")
    thread.run("two")

    first, second = sorted(_spans(exporter, "codex.turn"), key=lambda s: s.start_time)
    assert _usage_total(exporter, first) == 27849
    assert _usage_total(exporter, second) == 27849


def test_turn_stream_directly_matches_run(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    events = list(thread.turn("list files").stream())

    assert events[-1].method == "turn/completed"
    assert len(_spans(exporter, "codex.turn")) == 1
    assert len(_spans(exporter, "shell")) == 1


def test_failed_turn_sets_error_and_exception_propagates(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = [turn_started("t1"), turn_completed("t1", status="failed", error="model overloaded")]
    thread = _start_thread(client)

    with pytest.raises(RuntimeError, match="model overloaded"):
        thread.run("hi")

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.status.status_code == StatusCode.ERROR
    assert turn.status.description == "model overloaded"
    assert turn.attributes["gen_ai.codex.turn.status"] == "failed"


def test_interrupted_turn_is_not_an_error(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = [turn_started("t1"), turn_completed("t1", status="interrupted")]
    thread = _start_thread(client)

    thread.run("hi")

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.status.status_code == StatusCode.UNSET
    assert turn.attributes["gen_ai.codex.turn.status"] == "interrupted"


def test_failed_command_marks_tool_span_error(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1", exit_code=2)
    thread = _start_thread(client)

    thread.run("list files")

    (shell,) = _spans(exporter, "shell")
    assert shell.status.status_code == StatusCode.ERROR
    assert shell.attributes["gen_ai.codex.command.exit_code"] == 2


def test_early_break_ends_turn_span_and_closes_subscription(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    stream = thread.turn("list files").stream()
    for event in stream:
        if event.method == "item/started" and event.payload.item.root.type == "commandExecution":
            break
    stream.close()

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.attributes["gen_ai.codex.turn.stream_closed_early"] is True
    (shell,) = _spans(exporter, "shell")
    assert shell.attributes["gen_ai.codex.tool.status"] == "incomplete"
    assert client.subscriptions[0].closed
    assert codex_utils._turns == {}


def test_turn_start_carries_trace_context_of_turn_span(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    with exporter.tracer.start_as_current_span("app.root") as root:
        thread.run("list files")

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.parent.span_id == root.get_span_context().span_id
    (request,) = [m for m in client.written if m["method"] == "turn/start"]
    traceparent = request["trace"]["traceparent"]
    assert traceparent.split("-")[1] == format(turn.context.trace_id, "032x")
    assert traceparent.split("-")[2] == format(turn.context.span_id, "016x")


def test_trace_context_not_injected_into_other_requests(exporter, client) -> None:
    with exporter.tracer.start_as_current_span("app.root"):
        client._write_message({"id": "r", "method": "thread/start", "params": {}})
    assert "trace" not in client.written[0]


def test_content_not_recorded_when_trace_content_disabled(exporter, client, monkeypatch) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    thread.run("list files")

    (turn,) = _spans(exporter, "codex.turn")
    (shell,) = _spans(exporter, "shell")
    for key in ("input", "output", "gen_ai.prompt.0.content", "gen_ai.completion.0.content"):
        assert key not in turn.attributes
    for key in ("input", "output", "gen_ai.codex.command"):
        assert key not in shell.attributes
    assert shell.attributes["gen_ai.codex.command.exit_code"] == 0
    for generation in _generations(exporter, turn):
        assert not any(key.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for key in generation.attributes)
        assert "input" not in generation.attributes and "output" not in generation.attributes
        assert generation.attributes["llm.usage.total_tokens"] > 0


def test_async_stream_parity(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    async def consume() -> list:
        handle = await thread.aturn("list files")
        return [event async for event in handle.stream()]

    events = asyncio.run(consume())

    assert events[-1].method == "turn/completed"
    (turn,) = _spans(exporter, "codex.turn")
    assert turn.attributes["gen_ai.codex.turn.status"] == "completed"
    assert '"6 entries"' in turn.attributes["output"]
    (shell,) = _spans(exporter, "shell")
    assert shell.parent.span_id == turn.context.span_id
    assert _usage_total(exporter, turn) == 13810 + 14039


def test_input_joining_active_turn_does_not_create_second_turn_span(exporter, client) -> None:
    client.next_turn_ids = ["t1", "t1"]
    client.scripts["t1"] = shell_turn("t1")
    thread = _start_thread(client)

    handle = thread.turn("list files")
    thread.turn("also this")  # joins the active turn t1
    handle.run()

    assert len(_spans(exporter, "codex.turn")) == 1
    (join,) = _spans(exporter, "codex.turn.join")
    assert join.attributes["gen_ai.codex.turn_id"] == "t1"
    assert '"also this"' in join.attributes["input"]


def test_never_streamed_turn_is_ended_on_client_close(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    thread = _start_thread(client)

    thread.turn("list files")
    assert _spans(exporter, "codex.turn") == []

    client.close()

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.attributes["gen_ai.codex.turn.stream_closed_early"] is True


def test_turn_start_failure_ends_span_with_error(exporter, client) -> None:
    client.start_error = ConnectionError("app-server gone")
    thread = _start_thread(client)

    with pytest.raises(ConnectionError):
        thread.turn("hi")

    (turn,) = _spans(exporter, "codex.turn")
    assert turn.status.status_code == StatusCode.ERROR
    assert codex_utils._turns == {}


def test_raw_turn_start_without_handle_creates_no_span(exporter, client) -> None:
    client.next_turn_ids = ["t1"]

    client._start_turn(THREAD_ID, "hi", None, False)

    assert _spans(exporter, "codex.turn") == []


def test_mcp_and_file_change_items_become_tool_spans(exporter, client) -> None:
    client.next_turn_ids = ["t1"]
    mcp = dict(type="mcpToolCall", id="mcp-1", server="github", tool="search", arguments={"q": "x"})
    patch = dict(type="fileChange", id="fc-1", changes=[NS(path="a.py", kind=NS(type="update"), diff="@@ secret @@")])
    client.scripts["t1"] = [
        turn_started("t1"),
        item_started("t1", 1000, status="inProgress", **mcp),
        item_completed("t1", 1100, status="completed", result={"content": "ok"}, error=None, **mcp),
        item_started("t1", 1200, status="inProgress", **patch),
        item_completed("t1", 1300, status="completed", **patch),
        item_completed("t1", 1400, type="reasoning", id="r1", summary=["thought"], content=[]),
        turn_completed("t1"),
    ]
    thread = _start_thread(client)

    thread.run("go")

    (mcp_span,) = _spans(exporter, "github.search")
    assert mcp_span.attributes["gen_ai.codex.mcp.server"] == "github"
    assert '"q"' in mcp_span.attributes["input"]
    (patch_span,) = _spans(exporter, "apply_patch")
    assert "a.py" in patch_span.attributes["gen_ai.codex.file_changes"]
    assert "secret" not in patch_span.attributes["gen_ai.codex.file_changes"]
    (turn,) = _spans(exporter, "codex.turn")
    assert [e.name for e in turn.events if e.name.startswith("codex.")] == ["codex.reasoning"]


def test_uninstrument_restores_original_methods(exporter) -> None:
    import wrapt

    api = sys.modules["openai_codex.api"]
    client_module = sys.modules["openai_codex.client"]
    assert isinstance(vars(api.TurnHandle)["stream"], wrapt.FunctionWrapper)

    NetraCodexInstrumentor()._uninstrument()

    assert not isinstance(vars(api.TurnHandle)["stream"], wrapt.FunctionWrapper)
    assert not isinstance(vars(api.AsyncTurnHandle)["stream"], wrapt.FunctionWrapper)
    for method in ("_start_turn", "_write_message", "thread_start", "close"):
        assert not isinstance(vars(client_module.CodexClient)[method], wrapt.FunctionWrapper)
