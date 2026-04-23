import json
import logging
from typing import Any, Dict, List, Optional, Tuple

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

ATTR_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_OUTPUT_TYPE = "gen_ai.output.type"

ATTR_ENTITY = "gen_ai.entity"

_ENTITY_SPAN_TYPE_MAP: Dict[str, SpanType] = {
    "agent": SpanType.AGENT,
    "team": SpanType.AGENT,
    "workflow": SpanType.SPAN,
    "tool": SpanType.TOOL,
    "vectordb": SpanType.SPAN,
    "memory": SpanType.SPAN,
    "knowledge": SpanType.SPAN,
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

        if isinstance(result, str):
            return result

        return json.dumps(result)

    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to serialize value: %s", e)
        return _safe_str(data)


def is_run_content(event: Any) -> bool:
    """
    Return True if the event represents a "run_content" event.
    Falls back to checking event.value == "RunContent" if import fails.

    Args:
        event: The event object to evaluate. May be None or any arbitrary type.

    Returns:
        True if the event corresponds to "run_content", otherwise False.

    """
    if event is None or (event_type := getattr(event, "event", None)) is None:
        return False

    try:
        from agno.agent import RunEvent

        return bool(event_type == RunEvent.run_content)
    except Exception:
        return bool(getattr(event_type, "value", None) == "RunContent")


def build_agent_input(agent: Any, input_content: Any) -> str:
    """
    Normalize and assemble agent input.

    Args:
        agent: Agno agent that may contain system messages.
        input_content: Input payload.

    Returns:
        - raw string if no system messages and input_content not list or dict
        - JSON-serialized string of list of messages otherwise
    """

    # --- Build system context first ---
    system_messages = []

    if instructions := getattr(agent, "instructions", None):
        system_messages.append({"role": "system", "content": instructions})

    if system_message := getattr(agent, "system_message", None):
        system_messages.append({"role": "system", "content": system_message})

    try:
        input_content = json.loads(input_content)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to parse input_content as JSON: %s", e)

    # --- Fast path: return raw string if no system context ---
    if not system_messages and not isinstance(input_content, (list, dict)):
        return str(input_content)

    # --- Normalize input_content into message list when system_messages are present ---
    if isinstance(input_content, str):
        messages = [{"role": "user", "content": input_content}]

    elif isinstance(input_content, dict):
        role = input_content.get("role")
        content = input_content.get("content")
        messages = [{"role": role, "content": content}] if role and content is not None else []

    elif isinstance(input_content, list):
        messages = [
            {
                "role": str(item.get("role")),
                "content": str(item.get("content")),
            }
            for item in input_content
            if isinstance(item, dict) and item.get("role") and item.get("content") is not None
        ]

    else:
        messages = [{"role": "user", "content": str(input_content)}]

    # --- Combine system + messages ---
    total_messages = [*system_messages, *messages] if system_messages else messages

    try:
        return json.dumps(total_messages)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to convert input messages to JSON string: %s", e)
        return str(total_messages)


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
    else:
        input_tokens = _safe_getattr(metrics, "input_tokens", 0)
        output_tokens = _safe_getattr(metrics, "output_tokens", 0)

    if input_tokens:
        attributes[ATTR_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens:
        attributes[ATTR_USAGE_OUTPUT_TOKENS] = output_tokens

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
        return serialize_value(first_arg, clean=True)

    for key in ("input", "input_message", "message"):
        value = kwargs.get(key)
        if value is not None:
            return serialize_value(value, clean=True)

    return None


def extract_output_content(response: Any) -> Optional[str]:
    """Extract output content from an Agno response object.

    Checks ``content``, ``message``, and ``result`` attributes in order,
    handling Pydantic models with ``model_dump_json``.

    Args:
        response: An Agno RunResponse or similar result object.

    Returns:
        The output as a string, or ``None`` if not available.
    """
    for attr in ("content", "message", "result"):
        value = _safe_getattr(response, attr)
        if not value:
            continue
        if hasattr(value, "model_dump_json"):
            try:
                return str(value.model_dump_json())
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to serialize Pydantic output: %s", e)
        return _safe_str(value)

    if isinstance(response, str):
        return response

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
        input_content = build_agent_input(instance, input_content)

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
