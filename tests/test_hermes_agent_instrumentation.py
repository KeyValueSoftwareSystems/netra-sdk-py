"""
Unit tests for NetraHermesAgentInstrumentor.

The real `hermes-agent` package is not a test dependency: minimal stand-in
modules for `agent.conversation_loop` and `model_tools` are registered in
sys.modules before instrumenting, mirroring Hermes's actual call signatures.
"""

import sys
import types
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.instrumentation.hermes_agent import NetraHermesAgentInstrumentor
from netra.processors.session_span_processor import SessionSpanProcessor

TURN_SPAN_NAME = "hermes-agent.turn"
SUBAGENT_TURN_SPAN_NAME = "hermes-agent.subagent.turn"


class FakeAgent:
    def __init__(
        self,
        platform: str = "cli",
        session_id: str = "sess-1",
        model: str = "test-model",
        call_tool: bool = False,
        raise_error: bool = False,
        fail_turn: bool = False,
        interrupt_turn: bool = False,
        delegate_child: Optional["FakeAgent"] = None,
    ):
        self.platform = platform
        self.session_id = session_id
        self.model = model
        self.call_tool = call_tool
        self.raise_error = raise_error
        self.fail_turn = fail_turn
        self.interrupt_turn = interrupt_turn
        self.delegate_child = delegate_child


def _build_fake_hermes_modules(model_tools_file: str = "/fake/hermes/model_tools.py") -> Dict[str, types.ModuleType]:
    """Build stub agent/agent.conversation_loop/model_tools modules matching Hermes signatures."""
    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []  # type: ignore[attr-defined]
    agent_pkg.__file__ = "/fake/hermes/agent/__init__.py"

    conversation_loop = types.ModuleType("agent.conversation_loop")
    conversation_loop.__file__ = "/fake/hermes/agent/conversation_loop.py"

    model_tools = types.ModuleType("model_tools")
    model_tools.__file__ = model_tools_file

    tool_executor = types.ModuleType("agent.tool_executor")
    tool_executor.__file__ = "/fake/hermes/agent/tool_executor.py"

    skill_commands = types.ModuleType("agent.skill_commands")
    skill_commands.__file__ = "/fake/hermes/agent/skill_commands.py"

    skill_bundles = types.ModuleType("agent.skill_bundles")
    skill_bundles.__file__ = "/fake/hermes/agent/skill_bundles.py"

    tools_pkg = types.ModuleType("tools")
    tools_pkg.__path__ = []  # type: ignore[attr-defined]
    tools_pkg.__file__ = "/fake/hermes/tools/__init__.py"

    approval = types.ModuleType("tools.approval")
    approval.__file__ = "/fake/hermes/tools/approval.py"

    def build_skill_invocation_message(
        cmd_key: str,
        user_instruction: str = "",
        task_id: Optional[str] = None,
        runtime_note: str = "",
    ) -> Optional[str]:
        if cmd_key == "/missing":
            return None
        return f'[IMPORTANT: The user has invoked the "{cmd_key}" skill.] <body> {user_instruction}'

    def build_stacked_skill_invocation_message(
        cmd_keys: List[str],
        user_instruction: str = "",
        task_id: Optional[str] = None,
    ) -> Optional[tuple]:
        loaded = [k.lstrip("/") for k in cmd_keys if k and k != "/missing"]
        missing = [k.lstrip("/") for k in cmd_keys if k == "/missing"]
        if not loaded:
            return None
        return ("<stacked body>", loaded, missing)

    def build_bundle_invocation_message(
        cmd_key: str,
        user_instruction: str = "",
        task_id: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Optional[tuple]:
        if cmd_key == "/missing-bundle":
            return None
        return ("<bundle body>", ["skill-a", "skill-b"], [])

    def _run_approval_gate(
        *,
        pattern_key: str,
        description: str,
        display_target: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if display_target == "explode":
            raise RuntimeError("approval gate blew up")
        approved = display_target != "rm -rf /"
        return {"approved": approved, "message": None if approved else "denied"}

    def _run_agent_tool_execution_middleware(
        agent: Any,
        *,
        function_name: str,
        function_args: Dict[str, Any],
        effective_task_id: str,
        tool_call_id: str,
        execute: Any,
    ) -> tuple:
        return execute(function_args), function_args

    def handle_function_call(
        function_name: str,
        function_args: Dict[str, Any],
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        api_request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if function_name == "explode":
            raise RuntimeError("tool blew up")
        if function_name == "tool_call":
            # Mirrors the tool_search bridge: re-enter with the SAME tool_call_id.
            return model_tools.handle_function_call(  # type: ignore[attr-defined]
                function_args["underlying_name"],
                function_args.get("underlying_args", {}),
                task_id=task_id,
                tool_call_id=tool_call_id,
                session_id=session_id,
            )
        return '{"result": "ok"}'

    def run_conversation(
        agent: Any,
        user_message: Any,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_id: Optional[str] = None,
        stream_callback: Any = None,
        persist_user_message: Any = None,
        persist_user_timestamp: Optional[float] = None,
        moa_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if agent.raise_error:
            raise RuntimeError("turn failed")
        if agent.fail_turn:
            # Mirrors conversation_loop's retry-exhaustion return shape.
            return {
                "final_response": "Invalid API response after 10 retries: boom",
                "completed": False,
                "failed": True,
                "error": "Invalid API response after 10 retries: boom",
            }
        if agent.interrupt_turn:
            # Mirrors the user-interrupt return shape: not completed, not failed.
            return {"final_response": "[interrupted]", "completed": False}
        if agent.call_tool:
            model_tools.handle_function_call(  # type: ignore[attr-defined]
                "my_tool",
                {"x": 1},
                task_id=task_id,
                tool_call_id="call-1",
                session_id=agent.session_id,
            )
        if agent.delegate_child is not None:
            child = agent.delegate_child
            tool_executor._run_agent_tool_execution_middleware(  # type: ignore[attr-defined]
                agent,
                function_name="delegate_task",
                function_args={"goal": "sub goal"},
                effective_task_id=task_id or "",
                tool_call_id="call-d",
                execute=lambda next_args: conversation_loop.run_conversation(  # type: ignore[attr-defined]
                    child, "sub goal"
                ),
            )
        return {"final_response": "hello from hermes", "completed": True}

    conversation_loop.run_conversation = run_conversation  # type: ignore[attr-defined]
    model_tools.handle_function_call = handle_function_call  # type: ignore[attr-defined]
    tool_executor._run_agent_tool_execution_middleware = _run_agent_tool_execution_middleware  # type: ignore[attr-defined]
    skill_commands.build_skill_invocation_message = build_skill_invocation_message  # type: ignore[attr-defined]
    skill_commands.build_stacked_skill_invocation_message = build_stacked_skill_invocation_message  # type: ignore[attr-defined] # noqa: E501
    skill_bundles.build_bundle_invocation_message = build_bundle_invocation_message  # type: ignore[attr-defined]
    approval._run_approval_gate = _run_approval_gate  # type: ignore[attr-defined]
    agent_pkg.conversation_loop = conversation_loop  # type: ignore[attr-defined]
    agent_pkg.tool_executor = tool_executor  # type: ignore[attr-defined]
    agent_pkg.skill_commands = skill_commands  # type: ignore[attr-defined]
    agent_pkg.skill_bundles = skill_bundles  # type: ignore[attr-defined]
    tools_pkg.approval = approval  # type: ignore[attr-defined]
    return {
        "agent": agent_pkg,
        "agent.conversation_loop": conversation_loop,
        "agent.tool_executor": tool_executor,
        "agent.skill_commands": skill_commands,
        "agent.skill_bundles": skill_bundles,
        "model_tools": model_tools,
        "tools": tools_pkg,
        "tools.approval": approval,
    }


@pytest.fixture
def hermes_env(monkeypatch):
    modules = _build_fake_hermes_modules()
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Stamps baggage session_id onto spans as netra.session_id, like the real SDK pipeline.
    provider.add_span_processor(SessionSpanProcessor())

    instrumentor = NetraHermesAgentInstrumentor()
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    yield modules, exporter
    instrumentor.uninstrument()


def _run_conversation(modules: Dict[str, types.ModuleType], agent: FakeAgent, **kwargs: Any) -> Dict[str, Any]:
    return modules["agent.conversation_loop"].run_conversation(agent, "hi there", **kwargs)


def _spans_by_name(exporter: InMemorySpanExporter, name: str) -> List[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


@pytest.mark.unit
class TestTurnSpan:
    def test_turn_span_name_type_and_attributes(self, hermes_env):
        modules, exporter = hermes_env
        agent = FakeAgent()

        result = _run_conversation(modules, agent, task_id="task-42")

        assert result == {"final_response": "hello from hermes", "completed": True}
        (span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert span.attributes["netra.span.type"] == "AGENT"
        assert span.attributes["gen_ai.session.id"] == "sess-1"
        assert span.attributes["gen_ai.hermes_agent.task_id"] == "task-42"
        assert span.attributes["gen_ai.request.model"] == "test-model"
        assert '"role": "user"' in span.attributes["input"]
        assert "hi there" in span.attributes["input"]
        assert "hello from hermes" in span.attributes["output"]

    def test_subagent_turn_span_name_and_identity(self, hermes_env):
        modules, exporter = hermes_env
        agent = FakeAgent(platform="subagent", session_id="child-sess", model="child-model")
        agent._subagent_id = "sa-1-abcd1234"
        agent._parent_session_id = "parent-sess"

        _run_conversation(modules, agent)

        assert not _spans_by_name(exporter, TURN_SPAN_NAME)
        (span,) = _spans_by_name(exporter, SUBAGENT_TURN_SPAN_NAME)
        assert span.attributes["netra.span.type"] == "AGENT"
        assert span.attributes["gen_ai.agent.id"] == "sa-1-abcd1234"
        assert span.attributes["gen_ai.agent.parent_session.id"] == "parent-sess"
        assert span.attributes["gen_ai.request.model"] == "child-model"

    def test_turn_exception_sets_error_and_reraises(self, hermes_env):
        modules, exporter = hermes_env
        agent = FakeAgent(raise_error=True)

        with pytest.raises(RuntimeError, match="turn failed"):
            _run_conversation(modules, agent)

        (span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert not span.status.is_ok
        assert any(e.name == "exception" for e in span.events)


@pytest.mark.unit
class TestTurnOutcome:
    def test_successful_turn_records_completed_true(self, hermes_env):
        modules, exporter = hermes_env

        _run_conversation(modules, FakeAgent())

        (span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert span.status.is_ok
        assert span.attributes["gen_ai.hermes_agent.completed"] is True

    def test_failed_turn_sets_error_status_without_exception(self, hermes_env):
        modules, exporter = hermes_env

        result = _run_conversation(modules, FakeAgent(fail_turn=True))

        assert result["failed"] is True
        (span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert not span.status.is_ok
        assert span.attributes["gen_ai.hermes_agent.completed"] is False
        assert "Invalid API response" in span.attributes["gen_ai.hermes_agent.error"]
        assert "Invalid API response" in span.attributes["output"]

    def test_interrupted_turn_stays_ok(self, hermes_env):
        modules, exporter = hermes_env

        _run_conversation(modules, FakeAgent(interrupt_turn=True))

        (span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert span.status.is_ok
        assert span.attributes["gen_ai.hermes_agent.completed"] is False
        assert "gen_ai.hermes_agent.error" not in span.attributes


@pytest.mark.unit
class TestInlineToolSpan:
    def test_inline_tool_span_attributes(self, hermes_env):
        modules, exporter = hermes_env
        agent = FakeAgent(session_id="sess-i")
        agent._current_turn_id = "turn-3"
        agent._current_api_request_id = "req-3"

        result = modules["agent.tool_executor"]._run_agent_tool_execution_middleware(
            agent,
            function_name="todo",
            function_args={"todos": ["a"]},
            effective_task_id="task-7",
            tool_call_id="call-11",
            execute=lambda next_args: '{"done": true}',
        )

        assert result == ('{"done": true}', {"todos": ["a"]})
        (span,) = _spans_by_name(exporter, "todo")
        assert span.attributes["netra.span.type"] == "TOOL"
        assert span.attributes["gen_ai.tool.name"] == "todo"
        assert span.attributes["gen_ai.tool.call.id"] == "call-11"
        assert span.attributes["gen_ai.session.id"] == "sess-i"
        assert span.attributes["gen_ai.hermes_agent.turn_id"] == "turn-3"
        assert span.attributes["gen_ai.hermes_agent.api_request_id"] == "req-3"
        assert "todos" in span.attributes["input"]
        assert "done" in span.attributes["output"]

    def test_subagent_turn_nests_under_delegate_task_span(self, hermes_env):
        modules, exporter = hermes_env
        child = FakeAgent(platform="subagent", session_id="child-sess")
        child._subagent_id = "sa-0-abcd"
        child._parent_session_id = "parent-sess"
        parent = FakeAgent(session_id="parent-sess", delegate_child=child)

        _run_conversation(modules, parent)

        (turn_span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        (delegate_span,) = _spans_by_name(exporter, "delegate_task")
        (subagent_span,) = _spans_by_name(exporter, SUBAGENT_TURN_SPAN_NAME)
        assert delegate_span.parent.span_id == turn_span.context.span_id
        assert subagent_span.parent.span_id == delegate_span.context.span_id
        assert subagent_span.context.trace_id == turn_span.context.trace_id

    def test_inline_tool_exception_sets_error_and_reraises(self, hermes_env):
        modules, exporter = hermes_env

        def _explode(next_args: Dict[str, Any]) -> Any:
            raise RuntimeError("inline tool blew up")

        with pytest.raises(RuntimeError, match="inline tool blew up"):
            modules["agent.tool_executor"]._run_agent_tool_execution_middleware(
                FakeAgent(),
                function_name="memory",
                function_args={},
                effective_task_id="",
                tool_call_id="call-3",
                execute=_explode,
            )

        (span,) = _spans_by_name(exporter, "memory")
        assert not span.status.is_ok
        assert any(e.name == "exception" for e in span.events)


@pytest.mark.unit
class TestSessionBaggage:
    def test_netra_session_id_stamped_on_turn_and_children(self, hermes_env):
        modules, exporter = hermes_env

        _run_conversation(modules, FakeAgent(call_tool=True))

        (turn_span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        (tool_span,) = _spans_by_name(exporter, "my_tool")
        assert turn_span.attributes["netra.session_id"] == "sess-1"
        assert tool_span.attributes["netra.session_id"] == "sess-1"

    def test_baggage_detached_after_turn(self, hermes_env):
        modules, _ = hermes_env

        _run_conversation(modules, FakeAgent())

        assert baggage.get_baggage("session_id", otel_context.get_current()) is None

    def test_existing_session_baggage_wins(self, hermes_env):
        modules, exporter = hermes_env
        token = otel_context.attach(baggage.set_baggage("session_id", "app-set-session"))
        try:
            _run_conversation(modules, FakeAgent())
        finally:
            otel_context.detach(token)

        (turn_span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        assert turn_span.attributes["netra.session_id"] == "app-set-session"
        # gen_ai.session.id still records the agent's own session.
        assert turn_span.attributes["gen_ai.session.id"] == "sess-1"

    def test_delegation_tree_groups_under_parent_session(self, hermes_env):
        modules, exporter = hermes_env
        child = FakeAgent(platform="subagent", session_id="child-sess")
        child._subagent_id = "sa-0-abcd"
        child._parent_session_id = "parent-sess"
        parent = FakeAgent(session_id="parent-sess", delegate_child=child)

        _run_conversation(modules, parent)

        (subagent_span,) = _spans_by_name(exporter, SUBAGENT_TURN_SPAN_NAME)
        assert subagent_span.attributes["netra.session_id"] == "parent-sess"
        assert subagent_span.attributes["gen_ai.session.id"] == "child-sess"


@pytest.mark.unit
class TestToolSpan:
    def test_tool_span_attributes(self, hermes_env):
        modules, exporter = hermes_env
        modules["model_tools"].handle_function_call(
            "my_tool",
            {"x": 1},
            task_id="task-1",
            tool_call_id="call-9",
            session_id="sess-9",
            turn_id="turn-9",
            api_request_id="req-9",
        )

        (span,) = _spans_by_name(exporter, "my_tool")
        assert span.attributes["netra.span.type"] == "TOOL"
        assert span.attributes["gen_ai.tool.name"] == "my_tool"
        assert span.attributes["gen_ai.tool.call.id"] == "call-9"
        assert span.attributes["gen_ai.session.id"] == "sess-9"
        assert span.attributes["gen_ai.hermes_agent.turn_id"] == "turn-9"
        assert span.attributes["gen_ai.hermes_agent.api_request_id"] == "req-9"
        assert '\\"x\\": 1' in span.attributes["input"]
        assert "ok" in span.attributes["output"]

    def test_tool_span_nests_under_turn_span(self, hermes_env):
        modules, exporter = hermes_env
        agent = FakeAgent(call_tool=True)

        _run_conversation(modules, agent)

        (turn_span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        (tool_span,) = _spans_by_name(exporter, "my_tool")
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == turn_span.context.span_id
        assert tool_span.context.trace_id == turn_span.context.trace_id

    def test_bridge_recursion_creates_single_span(self, hermes_env):
        modules, exporter = hermes_env
        modules["model_tools"].handle_function_call(
            "tool_call",
            {"underlying_name": "real_tool", "underlying_args": {"y": 2}},
            tool_call_id="call-7",
        )

        tool_spans = [s for s in exporter.get_finished_spans() if s.attributes.get("netra.span.type") == "TOOL"]
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "tool_call"
        assert tool_spans[0].attributes["gen_ai.tool.call.id"] == "call-7"

    def test_recursion_without_tool_call_id_is_not_deduped(self, hermes_env):
        modules, exporter = hermes_env
        modules["model_tools"].handle_function_call(
            "tool_call",
            {"underlying_name": "real_tool"},
        )

        tool_spans = [s for s in exporter.get_finished_spans() if s.attributes.get("netra.span.type") == "TOOL"]
        assert sorted(s.name for s in tool_spans) == ["real_tool", "tool_call"]

    def test_tool_exception_sets_error_and_reraises(self, hermes_env):
        modules, exporter = hermes_env

        with pytest.raises(RuntimeError, match="tool blew up"):
            modules["model_tools"].handle_function_call("explode", {}, tool_call_id="call-2")

        (span,) = _spans_by_name(exporter, "explode")
        assert not span.status.is_ok
        assert any(e.name == "exception" for e in span.events)


@pytest.mark.unit
class TestContentTracing:
    def test_content_not_captured_when_trace_content_disabled(self, hermes_env, monkeypatch):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        modules, exporter = hermes_env

        _run_conversation(modules, FakeAgent(call_tool=True))

        (turn_span,) = _spans_by_name(exporter, TURN_SPAN_NAME)
        (tool_span,) = _spans_by_name(exporter, "my_tool")
        assert "input" not in turn_span.attributes
        assert "output" not in turn_span.attributes
        assert "input" not in tool_span.attributes
        assert "output" not in tool_span.attributes
        # Non-content metadata is still captured.
        assert turn_span.attributes["gen_ai.session.id"] == "sess-1"
        assert tool_span.attributes["gen_ai.tool.name"] == "my_tool"


SKILL_SPAN_NAME = "hermes-agent.skill.invoke"
APPROVAL_SPAN_NAME = "hermes-agent.approval"


@pytest.mark.unit
class TestSkillInvocationSpan:
    def test_single_skill_invocation_records_instruction(self, hermes_env):
        modules, exporter = hermes_env

        msg = modules["agent.skill_commands"].build_skill_invocation_message("/gif-search", "funny cats")

        assert msg is not None
        (span,) = _spans_by_name(exporter, SKILL_SPAN_NAME)
        assert span.attributes["netra.span.type"] == "SPAN"
        assert span.attributes["gen_ai.hermes_agent.skill.invocation_kind"] == "single"
        assert span.attributes["gen_ai.hermes_agent.skill.name"] == "gif-search"
        assert span.attributes["gen_ai.hermes_agent.skill.loaded"] is True
        # The span captures the user's actual instruction, not the skill body.
        assert "funny cats" in span.attributes["input"]
        assert "<body>" not in span.attributes["input"]

    def test_stacked_skill_invocation_records_loaded_names_and_count(self, hermes_env):
        modules, exporter = hermes_env

        result = modules["agent.skill_commands"].build_stacked_skill_invocation_message(
            ["/skill-a", "/skill-b", "/missing"], "do the thing"
        )

        assert result is not None
        (span,) = _spans_by_name(exporter, SKILL_SPAN_NAME)
        assert span.attributes["gen_ai.hermes_agent.skill.invocation_kind"] == "stacked"
        # Response attributes reflect the authoritative loaded_names from the result.
        assert span.attributes["gen_ai.hermes_agent.skill.name"] == "skill-a, skill-b"
        assert span.attributes["gen_ai.hermes_agent.skill.count"] == 2
        assert "do the thing" in span.attributes["input"]

    def test_bundle_invocation_span(self, hermes_env):
        modules, exporter = hermes_env

        result = modules["agent.skill_bundles"].build_bundle_invocation_message("/backend-dev", "refactor auth")

        assert result is not None
        (span,) = _spans_by_name(exporter, SKILL_SPAN_NAME)
        assert span.attributes["gen_ai.hermes_agent.skill.invocation_kind"] == "bundle"
        assert span.attributes["gen_ai.hermes_agent.skill.name"] == "skill-a, skill-b"
        assert span.attributes["gen_ai.hermes_agent.skill.count"] == 2

    def test_skill_not_found_marks_loaded_false(self, hermes_env):
        modules, exporter = hermes_env

        assert modules["agent.skill_commands"].build_skill_invocation_message("/missing") is None

        (span,) = _spans_by_name(exporter, SKILL_SPAN_NAME)
        assert span.attributes["gen_ai.hermes_agent.skill.loaded"] is False
        assert span.status.is_ok

    def test_skill_instruction_not_captured_when_content_disabled(self, hermes_env, monkeypatch):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        modules, exporter = hermes_env

        modules["agent.skill_commands"].build_skill_invocation_message("/gif-search", "funny cats")

        (span,) = _spans_by_name(exporter, SKILL_SPAN_NAME)
        assert "input" not in span.attributes
        assert span.attributes["gen_ai.hermes_agent.skill.name"] == "gif-search"


@pytest.mark.unit
class TestApprovalSpan:
    def test_approved_gate_records_approved_true(self, hermes_env):
        modules, exporter = hermes_env

        result = modules["tools.approval"]._run_approval_gate(
            pattern_key="rm-rf",
            description="Delete a directory",
            display_target="rm -rf ./build",
        )

        assert result["approved"] is True
        (span,) = _spans_by_name(exporter, APPROVAL_SPAN_NAME)
        assert span.attributes["netra.span.type"] == "SPAN"
        assert span.attributes["gen_ai.hermes_agent.approval.pattern_key"] == "rm-rf"
        assert span.attributes["gen_ai.hermes_agent.approval.description"] == "Delete a directory"
        assert span.attributes["gen_ai.hermes_agent.approval.approved"] is True
        assert "rm -rf ./build" in span.attributes["input"]
        assert span.status.is_ok

    def test_denied_gate_stays_ok_with_approved_false(self, hermes_env):
        modules, exporter = hermes_env

        result = modules["tools.approval"]._run_approval_gate(
            pattern_key="rm-rf-root",
            description="Delete root",
            display_target="rm -rf /",
        )

        assert result["approved"] is False
        (span,) = _spans_by_name(exporter, APPROVAL_SPAN_NAME)
        # A denial is a valid outcome, not an error.
        assert span.status.is_ok
        assert span.attributes["gen_ai.hermes_agent.approval.approved"] is False

    def test_approval_span_nests_under_tool_span(self, hermes_env):
        modules, exporter = hermes_env

        def _run_with_approval(next_args):
            modules["tools.approval"]._run_approval_gate(
                pattern_key="rm-rf",
                description="Delete a directory",
                display_target="rm -rf ./build",
            )
            return '{"output": "done"}'

        modules["agent.tool_executor"]._run_agent_tool_execution_middleware(
            FakeAgent(),
            function_name="terminal",
            function_args={"command": "rm -rf ./build"},
            effective_task_id="",
            tool_call_id="call-a",
            execute=_run_with_approval,
        )

        (tool_span,) = _spans_by_name(exporter, "terminal")
        (approval_span,) = _spans_by_name(exporter, APPROVAL_SPAN_NAME)
        assert approval_span.parent is not None
        assert approval_span.parent.span_id == tool_span.context.span_id
        assert approval_span.context.trace_id == tool_span.context.trace_id

    def test_approval_gate_exception_sets_error_and_reraises(self, hermes_env):
        modules, exporter = hermes_env

        with pytest.raises(RuntimeError, match="approval gate blew up"):
            modules["tools.approval"]._run_approval_gate(
                pattern_key="boom",
                description="boom",
                display_target="explode",
            )

        (span,) = _spans_by_name(exporter, APPROVAL_SPAN_NAME)
        assert not span.status.is_ok
        assert any(e.name == "exception" for e in span.events)


@pytest.mark.unit
class TestInstrumentorLifecycle:
    def test_uninstrument_restores_original_functions(self, hermes_env):
        modules, exporter = hermes_env
        instrumentor = NetraHermesAgentInstrumentor()
        instrumentor.uninstrument()
        exporter.clear()

        try:
            _run_conversation(modules, FakeAgent())
            modules["model_tools"].handle_function_call("my_tool", {})
            assert exporter.get_finished_spans() == ()
        finally:
            # Re-instrument so the fixture's teardown uninstrument stays balanced.
            provider = TracerProvider()
            instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)

    def test_instrument_skipped_for_non_hermes_modules(self, monkeypatch):
        # model_tools lives in a different root than agent/ -> not a Hermes install.
        modules = _build_fake_hermes_modules(model_tools_file="/elsewhere/model_tools.py")
        for name, mod in modules.items():
            monkeypatch.setitem(sys.modules, name, mod)

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        instrumentor = NetraHermesAgentInstrumentor()
        instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
        try:
            modules["agent.conversation_loop"].run_conversation(FakeAgent(), "hi")
            modules["model_tools"].handle_function_call("my_tool", {})
            assert exporter.get_finished_spans() == ()
        finally:
            instrumentor.uninstrument()
