import json
import logging
from typing import Any, Dict, List, Tuple

from opentelemetry.semconv_ai import SpanAttributes

from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

NETRA_SPAN_TYPE = "netra.span.type"


def build_llm_request_for_trace(llm_request: Any) -> Dict[str, Any]:
    """Serialize an ADK LlmRequest into a plain dict suitable for tracing, stripping binary/schema fields."""
    from google.genai import types

    result: Dict[str, Any] = {
        "model": llm_request.model,
        "config": llm_request.config.model_dump(exclude_none=True, exclude={"response_schema"}),
        "contents": [],
    }

    for content in llm_request.contents or []:
        parts = [part for part in content.parts if not hasattr(part, "inline_data") or not part.inline_data]
        result["contents"].append(types.Content(role=content.role, parts=parts).model_dump(exclude_none=True))
    return result


def extract_llm_request_attributes(llm_request_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a serialized LLM request dict into OpenTelemetry span attributes."""
    attributes: Dict[str, Any] = {}

    if "model" in llm_request_dict:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = llm_request_dict["model"]

    if "config" in llm_request_dict:
        config = llm_request_dict["config"]

        if "temperature" in config:
            attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = config["temperature"]

        if "max_output_tokens" in config:
            attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = config["max_output_tokens"]

        if "top_p" in config:
            attributes[SpanAttributes.LLM_REQUEST_TOP_P] = config["top_p"]

        if "top_k" in config:
            attributes[SpanAttributes.LLM_TOP_K] = config["top_k"]

        if "candidate_count" in config:
            attributes["gen_ai.request.candidate_count"] = config["candidate_count"]

        if "stop_sequences" in config:
            attributes[SpanAttributes.LLM_CHAT_STOP_SEQUENCES] = json.dumps(config["stop_sequences"])

        if "response_mime_type" in config:
            attributes["gen_ai.request.response_mime_type"] = config["response_mime_type"]

        if "tools" in config:
            for i, tool in enumerate(config["tools"]):
                if "function_declarations" in tool:
                    for j, func in enumerate(tool["function_declarations"]):
                        attributes[f"gen_ai.request.tools.{j}.name"] = func.get("name", "")
                        attributes[f"gen_ai.request.tools.{j}.description"] = func.get("description", "")

    message_index = 0
    all_inputs: List[Dict[str, Any]] = []

    if "config" in llm_request_dict and "system_instruction" in llm_request_dict["config"]:
        system_instruction = llm_request_dict["config"]["system_instruction"]
        attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = "system"
        attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"] = system_instruction
        all_inputs.append({"role": "system", "content": system_instruction})
        message_index += 1

    if "contents" in llm_request_dict:
        for content in llm_request_dict["contents"]:
            raw_role = content.get("role", "user")
            role = "assistant" if raw_role == "model" else raw_role
            parts = content.get("parts", [])

            attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = role

            text_parts: List[str] = []
            func_calls: List[Dict[str, Any]] = []
            func_responses: List[Dict[str, Any]] = []

            for part in parts:
                if "text" in part and part.get("text") is not None:
                    text_parts.append(str(part["text"]))
                elif "function_call" in part:
                    func_call = part["function_call"]
                    attributes[f"gen_ai.prompt.{message_index}.function_call.name"] = func_call.get("name", "")
                    attributes[f"gen_ai.prompt.{message_index}.function_call.args"] = json.dumps(
                        func_call.get("args", {})
                    )
                    if "id" in func_call:
                        attributes[f"gen_ai.prompt.{message_index}.function_call.id"] = func_call["id"]
                    entry: Dict[str, Any] = {"name": func_call.get("name", ""), "args": func_call.get("args", {})}
                    if "id" in func_call:
                        entry["id"] = func_call["id"]
                    func_calls.append(entry)
                elif "function_response" in part:
                    func_resp = part["function_response"]
                    attributes[f"gen_ai.prompt.{message_index}.function_response.name"] = func_resp.get("name", "")
                    attributes[f"gen_ai.prompt.{message_index}.function_response.result"] = json.dumps(
                        func_resp.get("response", {})
                    )
                    if "id" in func_resp:
                        attributes[f"gen_ai.prompt.{message_index}.function_response.id"] = func_resp["id"]
                    resp_entry: Dict[str, Any] = {
                        "name": func_resp.get("name", ""),
                        "result": func_resp.get("response", {}),
                    }
                    if "id" in func_resp:
                        resp_entry["id"] = func_resp["id"]
                    func_responses.append(resp_entry)

            msg: Dict[str, Any] = {"role": role}
            if text_parts:
                content_str = "\n".join(text_parts)
                attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"] = content_str
                msg["content"] = content_str
            if func_calls:
                msg["function_calls"] = func_calls
            if func_responses:
                msg["function_responses"] = func_responses
            all_inputs.append(msg)

            message_index += 1

    try:
        attributes["input"] = json.dumps(all_inputs)
    except Exception:
        logger.exception("Failed to serialize LLM request inputs to JSON")
        attributes["input"] = str(all_inputs)

    return attributes


def _get_event_content(event: Any) -> Tuple[List[Any], "str | None"]:
    """Return (parts, role) from an ADK event, falling back to empty on error."""
    try:
        parts = event.content.parts if event.content and event.content.parts else []
        role = event.content.role if event.content else None
    except Exception:
        logger.exception("Failed to extract content parts and role from event")
        parts, role = [], None
    return parts, role


def _extract_pending_tool_calls(event: Any) -> Dict[str, Any]:
    """Collect function_call info from an LLM event keyed by call id (or name) for later matching."""
    pending: Dict[str, Any] = {}
    try:
        parts = event.content.parts if event.content and event.content.parts else []
        for part in parts:
            if func_call := getattr(part, "function_call", None):
                call_id = getattr(func_call, "id", None)
                call_name = getattr(func_call, "name", None)
                entry: Dict[str, Any] = {}
                if call_name:
                    entry["name"] = call_name
                entry["arguments"] = getattr(func_call, "args", {})
                key = call_id or call_name
                if key:
                    pending[key] = entry
    except Exception:
        logger.exception("Failed to extract pending tool calls from event")
    return pending


def _resolve_span_type(role: str | None) -> SpanType:
    """Map an ADK content role to the corresponding Netra SpanType."""
    if role == "model":
        return SpanType.GENERATION
    if role == "user":
        return SpanType.TOOL
    return SpanType.SPAN


def extract_scalar_event_attributes(event: Any) -> Tuple[str | None, Dict[str, Any]]:
    """Extract flat (non-content) attributes from an ADK event. Returns (span_name_hint, attributes)."""
    attributes: Dict[str, Any] = {}
    span_name = None

    if model_version := getattr(event, "model_version", None):
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_version
        span_name = model_version

    if author := getattr(event, "author", None):
        attributes["gen_ai.event.author"] = author

    if invocation_id := getattr(event, "invocation_id", None):
        attributes["gen_ai.invocation.id"] = invocation_id

    if (timestamp := getattr(event, "timestamp", None)) is not None:
        attributes["gen_ai.event.timestamp"] = timestamp

    if (finish_reason := getattr(event, "finish_reason", None)) is not None:
        attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = (
            finish_reason.value if hasattr(finish_reason, "value") else str(finish_reason)
        )

    if (error_code := getattr(event, "error_code", None)) is not None:
        attributes["gen_ai.error.code"] = error_code

    if error_message := getattr(event, "error_message", None):
        attributes["gen_ai.error.message"] = error_message

    if usage := getattr(event, "usage_metadata", None):
        if (v := getattr(usage, "prompt_token_count", None)) is not None:
            attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = v
        if (v := getattr(usage, "candidates_token_count", None)) is not None:
            attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = v
        if (v := getattr(usage, "total_token_count", None)) is not None:
            attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = v
        if (v := getattr(usage, "cached_content_token_count", None)) is not None:
            attributes["gen_ai.usage.cached_tokens"] = v
        if (v := getattr(usage, "thoughts_token_count", None)) is not None:
            attributes["gen_ai.usage.thoughts_tokens"] = v

    if (avg_logprobs := getattr(event, "avg_logprobs", None)) is not None:
        attributes["gen_ai.response.avg_logprobs"] = avg_logprobs

    if actions := getattr(event, "actions", None):
        if v := getattr(actions, "transfer_to_agent", None):
            attributes["gen_ai.actions.transfer_to_agent"] = v
        if v := getattr(actions, "state_delta", None):
            attributes["gen_ai.actions.state_delta"] = json.dumps(v)
        if (v := getattr(actions, "escalate", None)) is not None:
            attributes["gen_ai.actions.escalate"] = v

    return span_name, attributes


def _process_content_parts(
    parts: List[Any], role: "str | None", pending_tool_calls: "Dict[str, Any] | None" = None
) -> Tuple[str | None, Dict[str, Any]]:
    """Build input/output attributes from content parts, optionally correlating with pending tool calls."""
    input_parts: List[Any] = []
    output_parts: List[Any] = []
    span_name = None

    for part in parts:
        if func_call := getattr(part, "function_call", None):
            entry: Dict[str, Any] = {}
            if call_id := getattr(func_call, "id", None):
                entry["id"] = call_id
            if call_name := getattr(func_call, "name", None):
                entry["name"] = call_name
                span_name = span_name or call_name
            entry["arguments"] = getattr(func_call, "args", {})
            # function_call is the LLM's output decision (what tool to invoke)
            output_parts.append(entry)

        elif func_resp := getattr(part, "function_response", None):
            entry = {}
            resp_id = getattr(func_resp, "id", None)
            resp_name = getattr(func_resp, "name", None)
            if resp_id:
                entry["id"] = resp_id
            if resp_name:
                entry["name"] = resp_name
                span_name = span_name or resp_name
            entry["result"] = getattr(func_resp, "response", {})
            output_parts.append(entry)

            # Populate input from the matching function_call recorded earlier
            if pending_tool_calls:
                lookup_key = resp_id or resp_name
                if lookup_key and lookup_key in pending_tool_calls:
                    input_parts.append(pending_tool_calls[lookup_key])

        elif (text := getattr(part, "text", None)) is not None:
            if role == "user":
                input_parts.append(str(text))
            else:
                output_parts.append(str(text))

    attributes: Dict[str, Any] = {}
    if input_parts:
        attributes["input"] = json.dumps(input_parts if len(input_parts) > 1 else input_parts[0])
    if output_parts:
        attributes["output"] = json.dumps(output_parts if len(output_parts) > 1 else output_parts[0])

    return span_name, attributes


def extract_event_attributes(
    event: Any, pending_tool_calls: "Dict[str, Any] | None" = None
) -> Tuple[str, Dict[str, Any]]:
    """Extract all tracing attributes from an ADK event. Returns (span_name, attributes)."""
    parts, role = _get_event_content(event)

    scalar_span_name, attributes = extract_scalar_event_attributes(event)
    parts_span_name, parts_attributes = _process_content_parts(parts, role, pending_tool_calls)

    attributes[NETRA_SPAN_TYPE] = _resolve_span_type(role)
    attributes.update(parts_attributes)

    span_name = scalar_span_name or parts_span_name or "unknown"
    return span_name, attributes


def extract_llm_response_attributes(last_response: Any, accumulated_text: List[str]) -> Dict[str, Any]:
    """Build span attributes from the final LLM response event and accumulated streamed text."""
    attributes: Dict[str, Any] = {}
    _, scalar_attrs = extract_scalar_event_attributes(last_response)
    attributes.update(scalar_attrs)

    content = getattr(last_response, "content", None)
    parts = (content.parts or []) if content else []

    text_parts: List[str] = accumulated_text if accumulated_text else []
    tool_calls: List[Dict[str, Any]] = []
    tool_call_index = 0

    for part in parts:
        if func_call := getattr(part, "function_call", None):
            entry: Dict[str, Any] = {}
            if call_id := getattr(func_call, "id", None):
                entry["id"] = call_id
                attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.id"] = call_id
            if call_name := getattr(func_call, "name", None):
                entry["name"] = call_name
                attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.name"] = call_name
            args = getattr(func_call, "args", {})
            entry["arguments"] = args
            attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.arguments"] = json.dumps(args)
            tool_calls.append(entry)
            tool_call_index += 1
        elif not accumulated_text and (text := getattr(part, "text", None)) is not None:
            text_parts.append(str(text))

    attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = "assistant"

    output: Dict[str, Any] = {"role": "assistant"}
    if text_parts:
        full_text = "\n".join(text_parts)
        attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = full_text
        output["content"] = full_text
    if tool_calls:
        output["tool_calls"] = tool_calls

    if len(output) > 1:
        try:
            attributes["output"] = json.dumps(output)
        except Exception:
            logger.exception("Failed to serialize LLM response output to JSON")
            attributes["output"] = str(output)

    return attributes


def extract_llm_attributes(llm_request: Dict[str, Any], llm_response: Any) -> Dict[str, Any]:
    """Merge request and response attributes into a single attribute dict."""
    return {**extract_llm_request_attributes(llm_request), **extract_llm_response_attributes(llm_response, [])}


def extract_agent_attributes(instance: Any) -> Dict[str, Any]:
    """Extract agent metadata attributes from a BaseAgent instance, including nested sub-agents."""
    attributes: Dict[str, Any] = {}
    attributes["gen_ai.agent.name"] = getattr(instance, "name", "unknown")
    if hasattr(instance, "description") and instance.description:
        attributes["gen_ai.agent.description"] = instance.description
    if hasattr(instance, "model") and instance.model:
        attributes["gen_ai.agent.model"] = instance.model
    if hasattr(instance, "instruction") and instance.instruction:
        attributes["gen_ai.agent.instruction"] = instance.instruction
    if hasattr(instance, "tools"):
        for idx, tool in enumerate(instance.tools):
            if hasattr(tool, "name"):
                attributes[f"gen_ai.agent.tools.{idx}.name"] = tool.name
            if hasattr(tool, "description"):
                attributes[f"gen_ai.agent.tools.{idx}.description"] = tool.description
    if hasattr(instance, "output_key") and instance.output_key:
        attributes["gen_ai.agent.output_key"] = instance.output_key
    if hasattr(instance, "sub_agents"):
        for i, sub_agent in enumerate(instance.sub_agents):
            sub_attrs = extract_agent_attributes(sub_agent)
            for key, value in sub_attrs.items():
                if value:
                    attributes[f"gen_ai.agent.sub_agents.{i}.{key}"] = value
    return attributes
