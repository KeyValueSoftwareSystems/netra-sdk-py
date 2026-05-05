import inspect
import json
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span

from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

NETRA_SPAN_TYPE = "netra.span.type"
LLM_SYSTEM_AGNO = "agno"

ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_AGENT_ID = "gen_ai.agent.id"
ATTR_AGENT_DESCRIPTION = "gen_ai.agent.description"
ATTR_AGENT_MODEL = "gen_ai.agent.model"
ATTR_AGENT_INSTRUCTIONS = "gen_ai.agent.instructions"
ATTR_AGENT_TOOLS = "gen_ai.agent.tools"
ATTR_AGENT_CONVERSATION_ID = "gen_ai.agent.conversation_id"
ATTR_AGENT_USER_ID = "gen_ai.agent.user_id"

ATTR_TEAM_NAME = "gen_ai.agno.team.name"
ATTR_TEAM_DESCRIPTION = "gen_ai.agno.team.description"
ATTR_TEAM_AGENTS = "gen_ai.agno.team.agents"
ATTR_TEAM_CONVERSATION_ID = "gen_ai.agno.team.conversation_id"

ATTR_WORKFLOW_NAME = "gen_ai.agno.workflow.name"
ATTR_WORKFLOW_DESCRIPTION = "gen_ai.agno.workflow.description"
ATTR_WORKFLOW_CONVERSATION_ID = "gen_ai.agno.workflow.conversation_id"

ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_DESCRIPTION = "gen_ai.tool.description"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call_id"
ATTR_TOOL_TYPE = "gen_ai.tool.type"

ATTR_MEMORY_OPERATION = "gen_ai.agno.memory.operation"
ATTR_MEMORY_DB_TYPE = "gen_ai.agno.memory.db_type"
ATTR_MEMORY_INPUT = "gen_ai.agno.memory.input"
ATTR_MEMORY_USER_ID = "gen_ai.agno.memory.user_id"

ATTR_KNOWLEDGE_DATASOURCE_ID = "gen_ai.data_source.id"
ATTR_VECTORDB_DATASOURCE_ID = "gen_ai.data_source.id"
ATTR_VECTORDB_OPERATION = "gen_ai.agno.vectordb.operation"

ATTR_AGENTOS_ID = "gen_ai.agno.agentos.id"
ATTR_AGENTOS_NAME = "gen_ai.agno.agentos.name"
ATTR_AGENTOS_DESCRIPTION = "gen_ai.agno.agentos.description"
ATTR_AGENTOS_VERSION = "gen_ai.agno.agentos.version"
ATTR_AGENTOS_AGENTS = "gen_ai.agno.agentos.agents"
ATTR_AGENTOS_TEAMS = "gen_ai.agno.agentos.teams"
ATTR_AGENTOS_WORKFLOWS = "gen_ai.agno.agentos.workflows"
ATTR_AGENTOS_ENTITY_TYPE = "gen_ai.agno.agentos.entity_type"
ATTR_AGENTOS_ENTITY_ID = "gen_ai.agno.agentos.entity_id"

ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_OUTPUT_TYPE = "gen_ai.output.type"

ATTR_ENTITY = "gen_ai.entity"

ATTR_HTTP_METHOD = "http.method"
ATTR_HTTP_URL = "http.url"
ATTR_HTTP_TARGET = "http.target"
ATTR_HTTP_SCHEME = "http.scheme"
ATTR_HTTP_FLAVOR = "http.flavor"
ATTR_HTTP_USER_AGENT = "http.user_agent"
ATTR_HTTP_STATUS_CODE = "http.status_code"
ATTR_NET_PEER_IP = "net.peer.ip"
ATTR_NET_PEER_PORT = "net.peer.port"
ATTR_NET_HOST_PORT = "net.host.port"
ATTR_AGENTOS_STREAM = "gen_ai.agno.agentos.stream"

_SENSITIVE_HEADERS: FrozenSet[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)

_ENTITY_SPAN_TYPE_MAP: Dict[str, SpanType] = {
    "agent": SpanType.AGENT,
    "team": SpanType.AGENT,
    "workflow": SpanType.SPAN,
    "tool": SpanType.TOOL,
    "vectordb": SpanType.SPAN,
    "memory": SpanType.SPAN,
    "knowledge": SpanType.SPAN,
    "llm": SpanType.GENERATION,
    "agentos": SpanType.SPAN,
}

# Maps entity type to an attribute extractor with unified signature (instance, kwargs).
# Extractors that don't use kwargs simply ignore the second argument.
_ENTITY_EXTRACTOR_MAP: Dict[str, str] = {
    "agent": "extract_agent_attributes",
    "team": "extract_team_attributes",
    "workflow": "extract_workflow_attributes",
    "tool": "extract_tool_attributes",
}


def should_suppress_instrumentation() -> bool:
    """Return True if OTel instrumentation suppression is active in this context.

    Returns:
        True if the suppress-instrumentation context key is set, False otherwise.
    """
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Return ``getattr(obj, attr, default)``, swallowing any unexpected exception.

    Args:
        obj: The object to read from.
        attr: Attribute name.
        default: Value returned when the attribute is missing or access raises.

    Returns:
        The attribute value, or ``default`` on failure.
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    """Convert ``value`` to str, falling back to ``repr`` if ``str()`` raises.

    Args:
        value: Any Python object.

    Returns:
        A string representation of ``value``.
    """
    try:
        return str(value)
    except Exception:
        return repr(value)


def _is_lazy_iterable(value: Any) -> bool:
    """Return True if ``value`` is a coroutine, generator, or async generator.

    These types cannot be safely serialized: consuming a generator or awaiting
    a coroutine would mutate state (or execute side effects) in the instrumented
    code. Callers should treat them as opaque rather than attempting traversal.

    Args:
        value: Any Python object to inspect.

    Returns:
        True if ``value`` is a coroutine, generator, or async generator; False otherwise.
    """
    return inspect.isasyncgen(value) or inspect.iscoroutine(value) or inspect.isgenerator(value)


def _normalize(value: Any, *, clean: bool) -> Any:
    """Convert ``value`` into a JSON-serializable structure.

    Handles Pydantic models (via ``model_dump``), dictionaries, lists,
    and generic Python objects exposing ``__dict__``. Optionally removes
    ``None`` values during traversal.

    Args:
        value: The input value to normalize.
        clean: If ``True``, recursively removes keys/items whose value is ``None``.

    Returns:
        A JSON-serializable Python object. Structure mirrors the input,
        with optional removal of ``None`` values.
    """
    if value is None:
        return None

    if _is_lazy_iterable(value):
        return f"<{type(value).__name__}>"

    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump(), clean=clean)

    if isinstance(value, dict):
        return {
            k: v for k, v in ((k, _normalize(v, clean=clean)) for k, v in value.items()) if not (clean and v is None)
        }

    if isinstance(value, list):
        return [v for v in (_normalize(item, clean=clean) for item in value) if not (clean and v is None)]

    if hasattr(value, "__dict__"):
        return {
            k: v
            for k, v in ((k, _normalize(v, clean=clean)) for k, v in vars(value).items() if not k.startswith("_"))
            if not (clean and v is None)
        }

    if hasattr(value, "__slots__"):
        slot_attrs = {}
        for slot in getattr(value, "__slots__", ()):
            if not slot.startswith("_"):
                try:
                    slot_attrs[slot] = _normalize(getattr(value, slot), clean=clean)
                except AttributeError as e:
                    logger.debug(
                        "netra.instrumentation.agno: skipping unset slot %r on %s: %s", slot, type(value).__name__, e
                    )
        result = {k: v for k, v in slot_attrs.items() if not (clean and v is None)}
        if result:
            return result
    return value


def serialize_value(data: Any, clean: bool = False) -> Optional[str]:
    """Serialize ``data`` into a JSON string.

    Converts supported Python objects into a JSON-serializable structure
    and encodes it using ``json.dumps``. Falls back to ``_safe_str`` if
    serialization fails.

    Args:
        data: The value to serialize.
        clean: If ``True``, removes ``None`` values from the structure
            before serialization.

    Returns:
        A JSON-encoded string representation of ``data``. Returns the
        original string if ``data`` is already a string. Returns ``None``
        if ``data`` is ``None``. Falls back to ``_safe_str(data)`` if
        serialization fails.
    """
    if data is None:
        return None

    try:
        result = _normalize(data, clean=clean)

        if result is None:
            return None

        if isinstance(result, str):
            return result

        return json.dumps(result)

    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to serialize value: %s", e)
        return _safe_str(data)


def is_run_content(event: Any) -> bool:
    """Return True if the event represents a ``run_content`` event.

    Falls back to a string comparison with ``"RunContent"`` if the agno
    import fails.

    Args:
        event: The event object to evaluate.

    Returns:
        True if the event corresponds to ``RunEvent.run_content``, False otherwise.
    """
    if event is None or (event_type := getattr(event, "event", None)) is None:
        return False

    try:
        from agno.agent import RunEvent

        return bool(event_type == RunEvent.run_content.value)
    except Exception:
        return bool(event_type == "RunContent")


def is_assistant_response(event: Any) -> bool:
    """Return True if the event represents an ``assistant_response`` event.

    Falls back to a string comparison with ``"AssistantResponse"`` if the
    agno import fails.

    Args:
        event: The event object to evaluate.

    Returns:
        True if the event corresponds to ``ModelResponseEvent.assistant_response``, False otherwise.
    """
    if event is None or (event_type := getattr(event, "event", None)) is None:
        return False

    try:
        from agno.models.response import ModelResponseEvent

        return bool(event_type == ModelResponseEvent.assistant_response.value)
    except Exception:
        return bool(event_type == "AssistantResponse")


def parse_input_message_item(input_content: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single input message dict into a ``{role, content}`` pair.

    Args:
        input_content: A dict expected to contain ``role`` and ``content`` keys.

    Returns:
        ``{"role": role, "content": content}`` when both keys are present;
        otherwise ``{"role": "user", "content": input_content}``.
    """
    role = input_content.get("role")
    content = input_content.get("content")
    if role and content:
        return {"role": role, "content": content}
    else:
        return {"role": "user", "content": input_content}


def build_agent_input(input_content: Any) -> str:
    """
    Normalize agent input into a message list.

    The system prompt is intentionally omitted here — it is captured from the
    actual assembled messages passed to ``Model.response()`` by
    ``update_active_span_with_system_prompt`` and written back onto the span
    at that point, replacing this initial value.

    Args:
        input_content: Input payload from the ``run()`` call.

    Returns:
        - JSON-serialized list of user messages
    """
    if not input_content:
        return json.dumps([{"role": "user", "content": ""}])
    if isinstance(input_content, str):
        try:
            input_content = json.loads(input_content)
        except (json.JSONDecodeError, ValueError):
            pass  # Plain string input (e.g. from AgentOS HTTP requests) — not an error
    if isinstance(input_content, dict):
        messages = [parse_input_message_item(input_content)]
    elif isinstance(input_content, list):
        messages = []
        for item in input_content:
            if isinstance(item, dict):
                messages.append(parse_input_message_item(item))
            elif item is not None:
                messages.append({"role": "user", "content": str(item)})
    else:
        messages = [{"role": "user", "content": str(input_content)}]

    try:
        return json.dumps(messages)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to convert input messages to JSON string: %s", e)
        return str(messages)


def update_active_span_with_system_prompt(messages: Any) -> None:
    """Extract the actual system prompt from the model's assembled message list.

    Called from the ``Model.response`` / ``Model.aresponse`` wrappers where
    ``messages`` is the fully-assembled list that agno passes to the underlying
    LLM adapter.  The system message at position 0 is the real prompt — built
    from ``description``, ``role``, ``instructions``, ``additional_information``,
    tool schemas, memories, etc. — so we use it directly instead of guessing.

    The function updates the ``input`` attribute on the currently active OTel
    span (expected to be the enclosing agent/team span).
    """
    try:
        from opentelemetry.trace import get_current_span

        span = get_current_span()
        if not span.is_recording():
            return

        if not messages:
            return

        system_content: Optional[str] = None
        first_user_content: Optional[str] = None

        for msg in messages:
            role = getattr(msg, "role", None)
            if role == "system" and system_content is None:
                raw = getattr(msg, "content", None)
                if raw is not None:
                    system_content = raw if isinstance(raw, str) else json.dumps(raw)
            elif role == "user" and first_user_content is None:
                raw = getattr(msg, "content", None)
                if raw is not None:
                    first_user_content = raw if isinstance(raw, str) else json.dumps(raw)

        if system_content is None:
            return

        msg_list: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        if first_user_content is not None:
            msg_list.append({"role": "user", "content": first_user_content})

        span.set_attribute("input", json.dumps(msg_list))
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to update span with system prompt: %s", e)


def extract_agent_attributes(instance: Any, run_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract span attributes from an Agno Agent instance.

    Args:
        instance: An Agno Agent object.
        run_kwargs: Keyword arguments from the ``run()`` call (e.g. ``session_id``).

    Returns:
        Dict of span attribute key-value pairs for the agent.
    """
    attributes: Dict[str, Any] = {}

    name = _safe_getattr(instance, "name")
    if name:
        attributes[ATTR_AGENT_NAME] = name

    agent_id = _safe_getattr(instance, "agent_id") or _safe_getattr(instance, "id")
    if agent_id:
        attributes[ATTR_AGENT_ID] = _safe_str(agent_id)

    description = _safe_getattr(instance, "description")
    if description:
        attributes[ATTR_AGENT_DESCRIPTION] = description

    model = _safe_getattr(instance, "model")
    if model:
        model_id = _safe_getattr(model, "id") or _safe_getattr(model, "name")
        if model_id:
            attributes[ATTR_AGENT_MODEL] = _safe_str(model_id)

    instructions = _safe_getattr(instance, "instructions")
    if instructions:
        if isinstance(instructions, list):
            attributes[ATTR_AGENT_INSTRUCTIONS] = "\n".join(str(i) for i in instructions)
        else:
            attributes[ATTR_AGENT_INSTRUCTIONS] = _safe_str(instructions)

    tools = _safe_getattr(instance, "tools")
    if tools:
        tool_defs: List[Dict[str, str]] = []
        for tool in tools:
            tool_def: Dict[str, str] = {}
            t_name = _safe_getattr(tool, "name")
            if t_name:
                tool_def["name"] = t_name
            t_desc = _safe_getattr(tool, "description")
            if t_desc:
                tool_def["description"] = t_desc
            if tool_def:
                tool_defs.append(tool_def)
        if tool_defs:
            attributes[ATTR_AGENT_TOOLS] = serialize_value(tool_defs)

    # Prefer session_id from run() kwargs (set per-run); fall back to instance attribute
    conversation_id = (
        (run_kwargs.get("session_id") or run_kwargs.get("conversation_id") if run_kwargs else None)
        or _safe_getattr(instance, "session_id")
        or _safe_getattr(instance, "conversation_id")
    )
    if conversation_id:
        attributes[ATTR_AGENT_CONVERSATION_ID] = _safe_str(conversation_id)

    user_id = _safe_getattr(instance, "user_id")
    if user_id:
        attributes[ATTR_AGENT_USER_ID] = _safe_str(user_id)

    return attributes


def extract_team_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Team instance.

    Args:
        instance: An Agno Team object.

    Returns:
        Dict of span attribute key-value pairs for the team.
    """
    attributes: Dict[str, Any] = {}

    name = _safe_getattr(instance, "name")
    if name:
        attributes[ATTR_TEAM_NAME] = name

    description = _safe_getattr(instance, "description")
    if description:
        attributes[ATTR_TEAM_DESCRIPTION] = description

    members = _safe_getattr(instance, "members") or _safe_getattr(instance, "agents")
    if members:
        member_names: List[str] = []
        for member in members:
            m_name = _safe_getattr(member, "name")
            if m_name:
                member_names.append(m_name)
        if member_names:
            attributes[ATTR_TEAM_AGENTS] = serialize_value(member_names)

    conversation_id = _safe_getattr(instance, "session_id") or _safe_getattr(instance, "conversation_id")
    if conversation_id:
        attributes[ATTR_TEAM_CONVERSATION_ID] = _safe_str(conversation_id)

    return attributes


def extract_workflow_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Workflow instance.

    Args:
        instance: An Agno Workflow object.

    Returns:
        Dict of span attribute key-value pairs for the workflow.
    """
    attributes: Dict[str, Any] = {}

    name = _safe_getattr(instance, "name")
    if name:
        attributes[ATTR_WORKFLOW_NAME] = name

    description = _safe_getattr(instance, "description")
    if description:
        attributes[ATTR_WORKFLOW_DESCRIPTION] = description

    conversation_id = _safe_getattr(instance, "session_id") or _safe_getattr(instance, "conversation_id")
    if conversation_id:
        attributes[ATTR_WORKFLOW_CONVERSATION_ID] = _safe_str(conversation_id)

    return attributes


def extract_tool_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno FunctionCall instance.

    Args:
        instance: An Agno FunctionCall object.

    Returns:
        Dict of span attribute key-value pairs for the tool call.
    """
    attributes: Dict[str, Any] = {}

    func = _safe_getattr(instance, "function")
    if func:
        func_name = _safe_getattr(func, "name")
        if func_name:
            attributes[ATTR_TOOL_NAME] = func_name
        func_desc = _safe_getattr(func, "description")
        if func_desc:
            attributes[ATTR_TOOL_DESCRIPTION] = func_desc
    else:
        func_name = _safe_getattr(instance, "name")
        if func_name:
            attributes[ATTR_TOOL_NAME] = func_name

    call_id = _safe_getattr(instance, "call_id")
    if call_id:
        attributes[ATTR_TOOL_CALL_ID] = _safe_str(call_id)

    tool_type = _safe_getattr(instance, "type")
    if tool_type:
        attributes[ATTR_TOOL_TYPE] = _safe_str(tool_type)

    return attributes


def extract_memory_attributes(instance: Any, args: Tuple[Any, ...], operation: str) -> Dict[str, Any]:
    """Extract span attributes from an Agno Memory operation.

    Args:
        instance: The Memory or MemoryManager object.
        args: Positional arguments passed to the memory method.
        operation: The memory operation name (e.g. ``"add_user_memory"``).

    Returns:
        Dict of span attribute key-value pairs for the memory operation.
    """
    attributes: Dict[str, Any] = {ATTR_MEMORY_OPERATION: operation}

    db = _safe_getattr(instance, "db")
    if db is not None:
        attributes[ATTR_MEMORY_DB_TYPE] = type(db).__name__

    user_id = _safe_getattr(instance, "user_id")
    if user_id:
        attributes[ATTR_MEMORY_USER_ID] = _safe_str(user_id)

    if args:
        try:
            attributes[ATTR_MEMORY_INPUT] = _safe_str(args[0])
        except Exception as e:
            logger.debug("netra.instrumentation.agno: failed to extract memory input: %s", e)

    return attributes


def extract_knowledge_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Knowledge instance.

    Args:
        instance: An Agno AgentKnowledge or Knowledge object.

    Returns:
        Dict of span attribute key-value pairs for the knowledge source.
    """
    attributes: Dict[str, Any] = {}

    data_source_id = _safe_getattr(instance, "id") or _safe_getattr(instance, "name")
    if data_source_id:
        attributes[ATTR_KNOWLEDGE_DATASOURCE_ID] = _safe_str(data_source_id)

    return attributes


def extract_vectordb_attributes(instance: Any, operation: str) -> Dict[str, Any]:
    """Extract span attributes from an Agno VectorDb operation.

    Args:
        instance: An Agno VectorDb object.
        operation: The operation name (e.g. ``"search"``, ``"upsert"``).

    Returns:
        Dict of span attribute key-value pairs for the vectordb operation.
    """
    attributes: Dict[str, Any] = {ATTR_VECTORDB_OPERATION: operation}

    data_source_id = _safe_getattr(instance, "name") or _safe_getattr(instance, "id")
    if data_source_id:
        attributes[ATTR_VECTORDB_DATASOURCE_ID] = _safe_str(data_source_id)

    return attributes


def extract_token_usage(response: Any) -> Dict[str, Any]:
    """Extract token usage metrics from an Agno response object.

    Args:
        response: An Agno RunResponse, RunOutput, or TeamRunOutput object.

    Returns:
        Dict with input/output token count attributes, or empty dict if unavailable.
    """
    attributes: Dict[str, Any] = {}

    metrics = _safe_getattr(response, "metrics")
    if metrics is None:
        return attributes

    if isinstance(metrics, dict):
        input_tokens = metrics.get("input_tokens", 0)
        output_tokens = metrics.get("output_tokens", 0)
        reasoning_tokens = metrics.get("reasoning_tokens", 0)
        total_tokens = metrics.get("total_tokens", 0)
    else:
        input_tokens = _safe_getattr(metrics, "input_tokens", 0)
        output_tokens = _safe_getattr(metrics, "output_tokens", 0)
        reasoning_tokens = _safe_getattr(metrics, "reasoning_tokens", 0)
        total_tokens = _safe_getattr(metrics, "total_tokens", 0)

    if input_tokens:
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if completion_tokens := output_tokens + reasoning_tokens:
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

    return attributes


def extract_input_content(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[str]:
    """Extract user query or input content from wrapper call arguments.

    Checks the first positional arg first, then common keyword argument names
    (``input``, ``input_message``, ``message``).

    Args:
        args: Positional arguments from the wrapped call.
        kwargs: Keyword arguments from the wrapped call.

    Returns:
        The extracted input as a string, or ``None`` if not found.
    """
    if args and (first_arg := args[0]):
        if not (inspect.isasyncgen(first_arg) or inspect.iscoroutine(first_arg) or inspect.isgenerator(first_arg)):
            return serialize_value(first_arg, clean=True)

    for key in ("input", "input_message", "message"):
        value = kwargs.get(key)
        if value is not None:
            return serialize_value(value, clean=True)

    return None


def extract_output_content(response: Any) -> Optional[str]:
    """Extract output content from an Agno response object.

    Returns output as a JSON-encoded list ``[{"role": "assistant", "content": "..."}]``.

    Args:
        response: An Agno RunResponse or similar result object.

    Returns:
        JSON string ``[{"role": "assistant", "content": "..."}]``, or ``None`` if unavailable.
    """
    content: Optional[str] = None

    for attr in ("content", "message", "result"):
        value = _safe_getattr(response, attr)
        if not value:
            continue
        if hasattr(value, "model_dump_json"):
            try:
                content = str(value.model_dump_json())
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to serialize Pydantic output: %s", e)
                content = _safe_str(value)
        else:
            content = _safe_str(value)
        break

    if content is None and isinstance(response, str):
        content = response

    if content is None:
        return None

    try:
        return json.dumps([{"role": "assistant", "content": content}])
    except Exception:
        return content


def format_messages_as_input(messages: Any) -> Optional[str]:
    """Format an agno message list as a JSON ``[{"role": "...", "content": "..."}]`` string.

    Args:
        messages: List of agno message objects, each with ``role`` and ``content`` attributes.

    Returns:
        JSON-encoded list string, or ``None`` if no valid messages found.
    """
    if not messages:
        return None

    msg_list: List[Dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role is None:
            continue
        raw = getattr(msg, "content", None)
        if raw is None:
            content: Any = ""
        elif isinstance(raw, str):
            content = raw
        else:
            try:
                content = _normalize(raw, clean=False)
            except Exception:
                content = _safe_str(raw)
        if not content and str(role) == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                try:
                    msg_list.append({"role": str(role), "tool_calls": _normalize(tool_calls, clean=True)})
                    continue
                except Exception as e:
                    logger.debug("netra.instrumentation.agno: failed to normalize tool_calls in message: %s", e)

        msg_list.append({"role": str(role), "content": content})

    if not msg_list:
        return None

    try:
        return json.dumps(msg_list)
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to format messages as input: %s", e)
        return None


def format_response_as_output(response: Any) -> Optional[str]:
    """Format a model response as JSON ``[{"role": "assistant", "content": "..."}]``.

    Args:
        response: An agno ModelResponse or similar object.

    Returns:
        JSON-encoded assistant message list, or ``None`` if no content found.
    """
    content: Optional[str] = None

    for attr in ("content", "message", "result"):
        value = _safe_getattr(response, attr)
        if not value:
            continue
        if hasattr(value, "model_dump_json"):
            try:
                content = str(value.model_dump_json())
            except Exception:
                content = _safe_str(value)
        else:
            content = _safe_str(value)
        break

    if content is None and isinstance(response, str):
        content = response

    if content is None:
        tool_calls = _safe_getattr(response, "tool_calls")
        if tool_calls:
            try:
                return json.dumps([{"role": "assistant", "tool_calls": _normalize(tool_calls, clean=True)}])
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to serialize tool_calls as output: %s", e)

    if content is None:
        return None

    try:
        return json.dumps([{"role": "assistant", "content": content}])
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to format response as output: %s", e)
        return None


def extract_response_id(response: Any) -> Optional[str]:
    """Extract the run/response ID from an Agno response object.

    Args:
        response: An Agno RunResponse or similar result object.

    Returns:
        The response ID string, or ``None`` if not available.
    """
    run_id = _safe_getattr(response, "run_id")
    if run_id:
        return _safe_str(run_id)
    return None


def get_tool_name(instance: Any) -> str:
    """Derive the tool function name from a FunctionCall instance.

    Args:
        instance: An Agno FunctionCall object.

    Returns:
        The tool name string, defaulting to ``"unknown_tool"``.
    """
    func = _safe_getattr(instance, "function")
    if func:
        name = _safe_getattr(func, "name")
        if name:
            return str(name)
    name = _safe_getattr(instance, "name")
    return str(name) if name else "unknown_tool"


def get_tool_arguments(instance: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    """Serialize tool call arguments for span attributes.

    Prefers ``instance.arguments``; falls back to the wrapper ``kwargs``.

    Args:
        instance: An Agno FunctionCall object.
        kwargs: Keyword arguments from the wrapped call.

    Returns:
        JSON-serialized arguments string, or ``None``.
    """
    arguments = _safe_getattr(instance, "arguments")
    if arguments is not None:
        return serialize_value(arguments) if isinstance(arguments, dict) else _safe_str(arguments)
    if kwargs:
        return serialize_value(kwargs)
    return None


def set_request_attributes(
    span: Span,
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    entity_type: str,
) -> None:
    """Set request-side span attributes for an Agno entity call.

    Writes system, entity type, and span-type attributes, then entity-specific
    attributes (agent name/model/tools, team members, etc.), then the input content.

    Args:
        span: The active OpenTelemetry span.
        instance: The Agno object (Agent, Team, Tool, etc.).
        args: Positional arguments from the wrapped call.
        kwargs: Keyword arguments from the wrapped call.
        entity_type: Entity type string (``"agent"``, ``"tool"``, ``"team"``, etc.).
    """
    if not span.is_recording():
        return

    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))

    extractor_name = _ENTITY_EXTRACTOR_MAP.get(entity_type)
    extractor = globals().get(extractor_name) if extractor_name else None
    if extractor:
        try:
            if entity_type == "agent":
                span.set_attributes(extractor(instance, kwargs))
            else:
                span.set_attributes(extractor(instance))
        except Exception as e:
            logger.debug(
                "netra.instrumentation.agno: failed to extract %s attributes: %s",
                entity_type,
                e,
            )

    input_content = extract_input_content(args, kwargs)
    if entity_type == "agent":
        input_content = build_agent_input(input_content)

    if input_content is not None:
        span.set_attribute("input", input_content)


def set_response_attributes(span: Span, response: Any) -> None:
    """Set response-side span attributes from an Agno response object.

    Writes token usage, output content, response ID, and output type.

    Args:
        span: The active OpenTelemetry span.
        response: The Agno response object (RunResponse, TeamRunOutput, etc.).
    """
    if not span.is_recording():
        return

    usage = extract_token_usage(response)
    if usage:
        span.set_attributes(usage)

    output = extract_output_content(response)
    if output:
        span.set_attribute("output", output)

    response_id = extract_response_id(response)
    if response_id:
        span.set_attribute(ATTR_RESPONSE_ID, response_id)


def sanitize_headers(raw_headers: List[Tuple[bytes, bytes]]) -> Dict[str, str]:
    """Convert ASGI raw header pairs to a dict with sensitive values redacted.

    Args:
        raw_headers: List of ``(name_bytes, value_bytes)`` tuples from the ASGI scope.

    Returns:
        Dict mapping lower-cased header names to values, with sensitive headers
        replaced by ``"[REDACTED]"``.
    """
    result: Dict[str, str] = {}
    for name_bytes, value_bytes in raw_headers:
        name = name_bytes.decode("latin-1").lower()
        result[name] = "[REDACTED]" if name in _SENSITIVE_HEADERS else value_bytes.decode("latin-1")
    return result


def build_scope_url(scope: Dict[str, Any]) -> str:
    """Reconstruct the full request URL from an ASGI scope.

    Args:
        scope: ASGI connection scope dict.

    Returns:
        Full URL string including scheme, host, port, path, and query string.
    """
    scheme = scope.get("scheme", "http")
    server = scope.get("server")
    path = scope.get("path", "/")
    query_string = scope.get("query_string", b"")

    if server:
        host, port = server
        if ":" in host:
            host = f"[{host}]"
        default_port = 443 if scheme == "https" else 80
        url = f"{scheme}://{host}{path}" if port == default_port else f"{scheme}://{host}:{port}{path}"
    else:
        url = path

    if query_string:
        url = f"{url}?{query_string.decode('latin-1')}"
    return url


def extract_http_request_attributes(scope: Dict[str, Any]) -> Dict[str, Any]:
    """Extract OTel-standard HTTP attributes from an ASGI scope.

    Args:
        scope: ASGI connection scope dict.

    Returns:
        Dict of OTel HTTP attribute key-value pairs (method, url, scheme, etc.).
    """
    attrs: Dict[str, Any] = {}

    method = scope.get("method", "")
    if method:
        attrs[ATTR_HTTP_METHOD] = method

    url = build_scope_url(scope)
    if url:
        attrs[ATTR_HTTP_URL] = url

    scheme = scope.get("scheme", "")
    if scheme:
        attrs[ATTR_HTTP_SCHEME] = scheme

    path = scope.get("path", "")
    if path:
        attrs[ATTR_HTTP_TARGET] = path

    flavor = scope.get("http_version", "")
    if flavor:
        attrs[ATTR_HTTP_FLAVOR] = flavor

    server = scope.get("server")
    if server:
        _, port = server
        attrs[ATTR_NET_HOST_PORT] = str(port)

    client = scope.get("client")
    if client:
        peer_ip, peer_port = client
        attrs[ATTR_NET_PEER_IP] = str(peer_ip)
        attrs[ATTR_NET_PEER_PORT] = str(peer_port)

    for name_bytes, value_bytes in scope.get("headers", []):
        if name_bytes.lower() == b"user-agent":
            attrs[ATTR_HTTP_USER_AGENT] = value_bytes.decode("latin-1")
            break

    return attrs


def set_agentos_request_input(
    span: Span,
    scope: Dict[str, Any],
    body: bytes,
) -> None:
    """Serialize the AgentOS HTTP request and set it as the span ``input`` attribute.

    Args:
        span: The active OpenTelemetry span.
        scope: ASGI connection scope dict.
        body: Raw request body bytes.
    """
    if not span.is_recording():
        return
    try:
        input_data: Dict[str, Any] = {
            "method": scope.get("method", ""),
            "url": build_scope_url(scope),
            "headers": sanitize_headers(scope.get("headers", [])),
        }
        if body:
            try:
                input_data["body"] = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    input_data["body"] = body.decode("utf-8")
                except UnicodeDecodeError:
                    input_data["body"] = f"<binary: {len(body)} bytes>"
        span.set_attribute("input", json.dumps(input_data))
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to set agentos request input: %s", e)


def set_agentos_response_output(
    span: Span,
    status_code: int,
    raw_headers: List[Tuple[bytes, bytes]],
    body: bytes,
) -> None:
    """Serialize the AgentOS HTTP response and set it as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        status_code: HTTP response status code.
        raw_headers: List of ``(name_bytes, value_bytes)`` response header tuples.
        body: Raw response body bytes.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": status_code,
            "headers": sanitize_headers(raw_headers),
        }
        if body:
            try:
                output_data["body"] = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    output_data["body"] = body.decode("utf-8")
                except UnicodeDecodeError:
                    output_data["body"] = f"<binary: {len(body)} bytes>"
        span.set_attribute("output", json.dumps(output_data))
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to set agentos response output: %s", e)


def set_llm_prompt_attributes(span: Span, messages: Any) -> None:
    """Set ``gen_ai.prompt.N.role`` and ``gen_ai.prompt.N.content`` attributes from agno message objects.

    Args:
        span: The active OpenTelemetry span.
        messages: Iterable of agno message objects with ``role`` and ``content`` attributes.
    """
    if not messages or not span.is_recording():
        return
    try:
        for index, msg in enumerate(messages):
            role = getattr(msg, "role", None)
            if role is None:
                continue
            raw = getattr(msg, "content", None)
            if raw is None:
                content: Any = ""
            elif isinstance(raw, str):
                content = raw
            else:
                try:
                    content = json.dumps(_normalize(raw, clean=False))
                except Exception:
                    content = _safe_str(raw)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", str(role))
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", content)
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to set prompt attributes: %s", e)


def set_llm_completion_attributes(span: Span, output_str: Optional[str]) -> None:
    """Set ``gen_ai.completion.N.role`` and ``gen_ai.completion.N.content`` from a JSON output string.

    Args:
        span: The active OpenTelemetry span.
        output_str: JSON string of the form ``[{"role": "...", "content": "..."}]``.
    """
    if not output_str or not span.is_recording():
        return
    try:
        completions = json.loads(output_str)
        for index, msg in enumerate(completions):
            role = msg.get("role", "assistant")
            content = msg.get("content") or msg.get("tool_calls", "")
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", str(role))
            if content:
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content",
                    content if isinstance(content, str) else json.dumps(content),
                )
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to set completion attributes: %s", e)


def extract_agentos_attributes(
    instance: Any,
    entity_type: str,
    entity_id: str,
) -> Dict[str, Any]:
    """Extract span attributes from an AgentOS instance.

    Captures AgentOS identity, registered entities, and which entity is being
    invoked. HTTP-level attributes (method, path, status) are omitted here
    because they are already captured by FastAPI instrumentation.

    Args:
        instance: The AgentOS object.
        entity_type: The entity type being run (``"agent"``, ``"team"``, ``"workflow"``).
        entity_id: The ID of the entity being invoked.

    Returns:
        Dict of span attribute key-value pairs for the AgentOS request.
    """
    attributes: Dict[str, Any] = {}

    os_id = _safe_getattr(instance, "id")
    if os_id:
        attributes[ATTR_AGENTOS_ID] = _safe_str(os_id)

    name = _safe_getattr(instance, "name")
    if name:
        attributes[ATTR_AGENTOS_NAME] = name

    description = _safe_getattr(instance, "description")
    if description:
        attributes[ATTR_AGENTOS_DESCRIPTION] = description

    version = _safe_getattr(instance, "version")
    if version:
        attributes[ATTR_AGENTOS_VERSION] = _safe_str(version)

    agents = _safe_getattr(instance, "agents")
    if agents:
        agent_ids = [_safe_getattr(a, "name") or _safe_getattr(a, "id") or "unknown" for a in agents]
        attributes[ATTR_AGENTOS_AGENTS] = serialize_value(agent_ids)

    teams = _safe_getattr(instance, "teams")
    if teams:
        team_ids = [_safe_getattr(t, "name") or _safe_getattr(t, "id") or "unknown" for t in teams]
        attributes[ATTR_AGENTOS_TEAMS] = serialize_value(team_ids)

    workflows = _safe_getattr(instance, "workflows")
    if workflows:
        workflow_ids = [_safe_getattr(w, "name") or _safe_getattr(w, "id") or "unknown" for w in workflows]
        attributes[ATTR_AGENTOS_WORKFLOWS] = serialize_value(workflow_ids)

    if entity_type:
        attributes[ATTR_AGENTOS_ENTITY_TYPE] = entity_type
    if entity_id:
        attributes[ATTR_AGENTOS_ENTITY_ID] = entity_id

    return attributes
