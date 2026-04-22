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

# GenAI semantic convention attribute keys used across Agno instrumentation
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

_ENTITY_EXTRACT_MAP = {
    "agent": "extract_agent_attributes",
    "team": "extract_team_attributes",
    "workflow": "extract_workflow_attributes",
    "tool": "extract_tool_attributes",
}

_ENTITY_SPAN_TYPE_MAP: Dict[str, SpanType] = {
    "agent": SpanType.AGENT,
    "team": SpanType.AGENT,
    "workflow": SpanType.SPAN,
    "tool": SpanType.TOOL,
    "vectordb": SpanType.SPAN,
    "memory": SpanType.SPAN,
    "knowledge": SpanType.SPAN,
}


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed."""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def set_request_attributes(
    span: Span,
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    entity_type: str,
) -> None:
    """Set request attributes on span for an Agno entity.

    Combines common attributes (system, entity, span_type), entity-specific
    attributes extracted from the instance, and input content.

    Parameters:
        span: The OpenTelemetry span.
        instance: The Agno object (Agent, Team, Tool, etc.).
        args: Positional arguments from the wrapped call.
        kwargs: Keyword arguments from the wrapped call.
        entity_type: Entity type string (e.g. "agent", "tool", "team").
    """
    if not span.is_recording():
        return

    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))

    extractor_name = _ENTITY_EXTRACT_MAP.get(entity_type)
    if extractor_name:
        extractor = globals().get(extractor_name)
        if extractor:
            span.set_attributes(extractor(instance))

    input_content = extract_input_content(args, kwargs)
    if input_content:
        span.set_attribute("input", input_content)


def set_response_attributes(span: Span, response: Any) -> None:
    """Set response attributes on span from an Agno response object.

    Sets token usage, output content, response ID, and output type.

    Parameters:
        span: The OpenTelemetry span.
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
        span.set_attribute(ATTR_OUTPUT_TYPE, "text")

    response_id = extract_response_id(response)
    if response_id:
        span.set_attribute(ATTR_RESPONSE_ID, response_id)


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely retrieve an attribute from an object, returning default on failure."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    """Convert a value to string, falling back to repr on failure."""
    try:
        return str(value)
    except Exception:
        return repr(value)


def extract_agent_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Agent instance.

    Parameters:
        instance: An Agno Agent object.

    Returns:
        Dictionary of span attribute key-value pairs.
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
            try:
                attributes[ATTR_AGENT_TOOLS] = json.dumps(tool_defs)
            except Exception:
                attributes[ATTR_AGENT_TOOLS] = _safe_str(tool_defs)

    conversation_id = _safe_getattr(instance, "session_id") or _safe_getattr(instance, "conversation_id")
    if conversation_id:
        attributes[ATTR_AGENT_CONVERSATION_ID] = _safe_str(conversation_id)

    user_id = _safe_getattr(instance, "user_id")
    if user_id:
        attributes[ATTR_AGENT_USER_ID] = _safe_str(user_id)

    return attributes


def extract_team_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Team instance.

    Parameters:
        instance: An Agno Team object.

    Returns:
        Dictionary of span attribute key-value pairs.
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
            try:
                attributes[ATTR_TEAM_AGENTS] = json.dumps(member_names)
            except Exception:
                attributes[ATTR_TEAM_AGENTS] = _safe_str(member_names)

    conversation_id = _safe_getattr(instance, "session_id") or _safe_getattr(instance, "conversation_id")
    if conversation_id:
        attributes[ATTR_TEAM_CONVERSATION_ID] = _safe_str(conversation_id)

    return attributes


def extract_workflow_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Workflow instance.

    Parameters:
        instance: An Agno Workflow object.

    Returns:
        Dictionary of span attribute key-value pairs.
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

    Parameters:
        instance: An Agno FunctionCall object.

    Returns:
        Dictionary of span attribute key-value pairs.
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

    Parameters:
        instance: The Memory or MemoryManager object.
        args: Positional arguments passed to the memory method.
        operation: The memory operation name (e.g. "add_user_memory", "search_user_memories").

    Returns:
        Dictionary of span attribute key-value pairs.
    """
    attributes: Dict[str, Any] = {
        ATTR_MEMORY_OPERATION: operation,
    }

    db_type = _safe_getattr(instance, "db")
    if db_type:
        db_class = type(db_type).__name__
        attributes[ATTR_MEMORY_DB_TYPE] = db_class

    user_id = _safe_getattr(instance, "user_id")
    if user_id:
        attributes[ATTR_MEMORY_USER_ID] = _safe_str(user_id)

    if args:
        try:
            attributes[ATTR_MEMORY_INPUT] = _safe_str(args[0])
        except (IndexError, Exception):
            pass

    return attributes


def extract_knowledge_attributes(instance: Any) -> Dict[str, Any]:
    """Extract span attributes from an Agno Knowledge instance.

    Parameters:
        instance: An Agno AgentKnowledge or Knowledge object.

    Returns:
        Dictionary of span attribute key-value pairs.
    """
    attributes: Dict[str, Any] = {}

    data_source_id = _safe_getattr(instance, "id") or _safe_getattr(instance, "name")
    if data_source_id:
        attributes[ATTR_KNOWLEDGE_DATASOURCE_ID] = _safe_str(data_source_id)

    return attributes


def extract_vectordb_attributes(instance: Any, operation: str) -> Dict[str, Any]:
    """Extract span attributes from an Agno VectorDb operation.

    Parameters:
        instance: An Agno VectorDb object.
        operation: The vectordb operation name (e.g. "search", "upsert").

    Returns:
        Dictionary of span attribute key-value pairs.
    """
    attributes: Dict[str, Any] = {
        ATTR_VECTORDB_OPERATION: operation,
    }

    data_source_id = _safe_getattr(instance, "name") or _safe_getattr(instance, "id")
    if data_source_id:
        attributes[ATTR_VECTORDB_DATASOURCE_ID] = _safe_str(data_source_id)

    return attributes


def extract_token_usage(response: Any) -> Dict[str, Any]:
    """Extract token usage metrics from an Agno response object.

    Parameters:
        response: An Agno RunResponse / RunOutput / TeamRunOutput object.

    Returns:
        Dictionary of token usage span attributes.
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
    """Extract user query / input content from wrapper arguments.

    Checks the first positional arg, then common keyword arg names.

    Parameters:
        args: Positional arguments from the wrapped call.
        kwargs: Keyword arguments from the wrapped call.

    Returns:
        The extracted input string, or None if not found.
    """
    if args:
        first_arg = args[0]
        if isinstance(first_arg, str) and first_arg:
            return first_arg

    for key in ("input", "input_message", "message"):
        value = kwargs.get(key)
        if value is not None:
            return _safe_str(value)

    if args:
        return _safe_str(args[0])

    return None


def extract_output_content(response: Any) -> Optional[str]:
    """Extract output content from an Agno response object.

    Parameters:
        response: An Agno RunResponse or similar result object.

    Returns:
        The extracted output string, or None if not available.
    """
    for attr in ("content", "message", "result"):
        value = _safe_getattr(response, attr)
        if value is not None:
            return _safe_str(value)

    if isinstance(response, str):
        return response

    return None


def extract_response_id(response: Any) -> Optional[str]:
    """Extract the run/response ID from an Agno response object.

    Parameters:
        response: An Agno RunResponse or similar result object.

    Returns:
        The response ID string, or None if not available.
    """
    run_id = _safe_getattr(response, "run_id")
    if run_id:
        return _safe_str(run_id)
    return None


def get_tool_name(instance: Any) -> str:
    """Derive the tool function name from a FunctionCall instance.

    Parameters:
        instance: An Agno FunctionCall object.

    Returns:
        The tool name string, defaulting to "unknown_tool".
    """
    func = _safe_getattr(instance, "function")
    if func:
        name = _safe_getattr(func, "name")
        if name:
            return name
    name = _safe_getattr(instance, "name")
    return name if name else "unknown_tool"


def get_tool_arguments(instance: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    """Serialize tool call arguments for span attributes.

    Parameters:
        instance: An Agno FunctionCall object.
        kwargs: Keyword arguments from the wrapped call.

    Returns:
        JSON-serialized arguments string, or None.
    """
    arguments = _safe_getattr(instance, "arguments")
    if arguments is not None:
        try:
            return json.dumps(arguments) if isinstance(arguments, dict) else _safe_str(arguments)
        except Exception:
            return _safe_str(arguments)
    if kwargs:
        try:
            return json.dumps(kwargs)
        except Exception:
            return _safe_str(kwargs)
    return None
