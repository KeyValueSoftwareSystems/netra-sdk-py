import json
from typing import Any, Dict

from opentelemetry.semconv_ai import SpanAttributes


def _build_llm_request_for_trace(llm_request: Any) -> Dict[str, Any]:
    from google.genai import types

    result: Dict[str, Any] = {
        "model": llm_request.model,
        "config": llm_request.config.model_dump(exclude_none=True, exclude="response_schema"),
        "contents": [],
    }

    for content in llm_request.contents:
        parts = [part for part in content.parts if not hasattr(part, "inline_data") or not part.inline_data]
        result["contents"].append(types.Content(role=content.role, parts=parts).model_dump(exclude_none=True))
    return result


def _extract_llm_attributes(llm_request_dict: Dict[str, Any], llm_response: Any) -> Dict[str, Any]:
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
    if "config" in llm_request_dict and "system_instruction" in llm_request_dict["config"]:
        system_instruction = llm_request_dict["config"]["system_instruction"]
        attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = "system"
        attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"] = system_instruction
        message_index += 1

    if "contents" in llm_request_dict:
        for content in llm_request_dict["contents"]:
            raw_role = content.get("role", "user")
            role = "assistant" if raw_role == "model" else raw_role
            parts = content.get("parts", [])

            attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role"] = role

            text_parts = []
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
                elif "function_response" in part:
                    func_resp = part["function_response"]
                    attributes[f"gen_ai.prompt.{message_index}.function_response.name"] = func_resp.get("name", "")
                    attributes[f"gen_ai.prompt.{message_index}.function_response.result"] = json.dumps(
                        func_resp.get("response", {})
                    )
                    if "id" in func_resp:
                        attributes[f"gen_ai.prompt.{message_index}.function_response.id"] = func_resp["id"]

            if text_parts:
                attributes[f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content"] = "\n".join(text_parts)

            message_index += 1

    if llm_response:
        try:
            response_dict = json.loads(llm_response) if isinstance(llm_response, str) else llm_response

            if "model" in response_dict:
                attributes[SpanAttributes.LLM_RESPONSE_MODEL] = response_dict["model"]

            if "content" in response_dict and "parts" in response_dict["content"]:
                parts = response_dict["content"]["parts"]
                attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = "assistant"

                text_parts = []
                tool_call_index = 0
                for part in parts:
                    if "text" in part and part.get("text") is not None:
                        text_parts.append(str(part["text"]))
                    elif "function_call" in part:
                        func_call = part["function_call"]
                        attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.name"] = func_call.get(
                            "name", ""
                        )
                        attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.arguments"] = json.dumps(
                            func_call.get("args", {})
                        )
                        if "id" in func_call:
                            attributes[f"gen_ai.completions.0.tool_calls.{tool_call_index}.id"] = func_call["id"]
                        tool_call_index += 1

                if text_parts:
                    attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = "\n".join(text_parts)

            if "finish_reason" in response_dict:
                attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = response_dict["finish_reason"]

            if "id" in response_dict:
                attributes[SpanAttributes.LLM_RESPONSE_ID] = response_dict["id"]

        except Exception:
            pass

    return attributes


def extract_agent_attributes(instance: Any) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    attributes["gen_ai.agent.name"] = getattr(instance, "name", "unknown")
    if hasattr(instance, "description"):
        attributes["gen_ai.agent.description"] = instance.description
    if hasattr(instance, "model"):
        attributes["gen_ai.agent.model"] = instance.model
    if hasattr(instance, "instruction"):
        attributes["gen_ai.agent.instruction"] = instance.instruction
    if hasattr(instance, "tools"):
        for idx, tool in enumerate(instance.tools):
            if hasattr(tool, "name"):
                attributes[f"gen_ai.agent.tools.{idx}.name"] = tool.name
            if hasattr(tool, "description"):
                attributes[f"gen_ai.agent.tools.{idx}.description"] = tool.description
    if hasattr(instance, "output_key"):
        attributes["gen_ai.agent.output_key"] = instance.output_key
    if hasattr(instance, "sub_agents"):
        for i, sub_agent in enumerate(instance.sub_agents):
            sub_attrs = extract_agent_attributes(sub_agent)
            for key, value in sub_attrs.items():
                attributes[f"gen_ai.agent.sub_agents.{i}.{key}"] = value
    return attributes
