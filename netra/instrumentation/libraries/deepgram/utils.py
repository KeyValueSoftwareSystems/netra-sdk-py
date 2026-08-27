import json
import logging
from typing import Any, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed."""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def set_request_attributes(span: Span, kwargs: Dict[str, Any], source_type: Optional[str] = None) -> None:
    """
    Set the request attributes for the span.

    Args:
        span: The span to set the attributes on.
        kwargs: The keyword arguments to extract the attributes from.
        source_type: The type of the source (e.g. "url" or "file").
    """
    if not span.is_recording():
        return

    GENERIC_ATTRIBUTES = [
        "callback",
        "extra",
        "model",
        "language",
        "encoding",
        "multichannel",
        "diarize",
        "detect_language",
        "detect_entities",
        "sentiment",
        "summarize",
        "topics",
        "intents",
        "tag",
        "custom_topic",
        "custom_intent",
    ]

    for key in GENERIC_ATTRIBUTES:
        value = kwargs.get(key)
        if value is not None:
            span.set_attribute(f"gen_ai.request.{key}", value)

    if source_type == "url" and "url" in kwargs:
        span.set_attribute("gen_ai.request.url", str(kwargs["url"]))

    if text := kwargs.get("text"):
        character_count = len(text)
        span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
        span.set_attribute("gen_ai.prompt.0.role", "Input")
        span.set_attribute("gen_ai.prompt.0.content", str(text))

    if request := kwargs.get("request"):
        if isinstance(request, dict):
            text = request.get("text", "")
            character_count = len(text)
            span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
            span.set_attribute("gen_ai.prompt.0.role", "Input")
            span.set_attribute("gen_ai.prompt.0.content", str(text))

    if source_type == "file" and "request" in kwargs:
        request_value = kwargs["request"]
        try:
            if isinstance(request_value, (bytes, bytearray)):
                span.set_attribute("gen_ai.request.size_bytes", len(request_value))
        except Exception as e:
            logger.debug("Failed to set Deepgram request size: %s", e)


def _set_websocket_results_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV1ResultsEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.results")

    if duration := getattr(message, "duration", None):
        span.set_attribute("deepgram.websocket.results.duration", duration)
    if start := getattr(message, "start", None):
        span.set_attribute("deepgram.websocket.results.start", start)
    if is_final := getattr(message, "is_final", None):
        span.set_attribute("deepgram.websocket.results.is_final", is_final)
    if speech_final := getattr(message, "speech_final", None):
        span.set_attribute("deepgram.websocket.results.speech_final", speech_final)
    if channel_index := getattr(message, "channel_index", None):
        span.set_attribute("deepgram.websocket.results.channel_index", channel_index)

    # Extract transcript from channel.alternatives
    if channel := getattr(message, "channel", None):
        if alternatives := getattr(channel, "alternatives", None):
            first_alt = alternatives[0] if isinstance(alternatives, list) else next(iter(alternatives), None)
            if first_alt:
                if transcript := getattr(first_alt, "transcript", None):
                    span.set_attribute("gen_ai.completion.0.role", "Transcribed Text")
                    span.set_attribute("gen_ai.completion.0.content", transcript)
                if confidence := getattr(first_alt, "confidence", None):
                    span.set_attribute("deepgram.websocket.results.confidence", confidence)
                if languages := getattr(first_alt, "languages", None):
                    span.set_attribute("deepgram.websocket.results.languages", languages)

    # Extract model info from metadata
    if metadata := getattr(message, "metadata", None):
        if request_id := getattr(metadata, "request_id", None):
            span.set_attribute("gen_ai.response.request_id", str(request_id))
        if model_uuid := getattr(metadata, "model_uuid", None):
            span.set_attribute("gen_ai.response.model_uuid", str(model_uuid))
        if model_info := getattr(metadata, "model_info", None):
            if name := getattr(model_info, "name", None):
                span.set_attribute("gen_ai.request.model", str(name))
            if version := getattr(model_info, "version", None):
                span.set_attribute("gen_ai.request.model_version", str(version))
            if arch := getattr(model_info, "arch", None):
                span.set_attribute("gen_ai.request.model_arch", str(arch))


def _set_websocket_metadata_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV1MetadataEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.metadata")

    if request_id := getattr(message, "request_id", None):
        span.set_attribute("gen_ai.response.request_id", str(request_id))
    if created := getattr(message, "created", None):
        span.set_attribute("deepgram.websocket.metadata.created", str(created))
    if duration := getattr(message, "duration", None):
        span.set_attribute("gen_ai.audio.duration", duration / 60)
    if channels := getattr(message, "channels", None):
        span.set_attribute("gen_ai.response.channels", int(channels))


def _set_websocket_utterance_end_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV1UtteranceEndEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.utterance_end")

    if channel := getattr(message, "channel", None):
        span.set_attribute("deepgram.websocket.utterance_end.channel", channel)
    if last_word_end := getattr(message, "last_word_end", None):
        span.set_attribute("deepgram.websocket.utterance_end.last_word_end", last_word_end)


def _set_websocket_speech_started_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV1SpeechStartedEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.speech_started")

    if channel := getattr(message, "channel", None):
        span.set_attribute("deepgram.websocket.speech_started.channel", channel)
    if timestamp := getattr(message, "timestamp", None):
        span.set_attribute("deepgram.websocket.speech_started.timestamp", timestamp)


def _set_websocket_v2_connected_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV2ConnectedEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.v2.connected")

    if request_id := getattr(message, "request_id", None):
        span.set_attribute("gen_ai.response.request_id", str(request_id))
    if sequence_id := getattr(message, "sequence_id", None):
        span.set_attribute("deepgram.websocket.v2.sequence_id", sequence_id)


def _set_websocket_v2_turn_info_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV2TurnInfoEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.v2.turn_info")

    if request_id := getattr(message, "request_id", None):
        span.set_attribute("gen_ai.response.request_id", str(request_id))
    if sequence_id := getattr(message, "sequence_id", None):
        span.set_attribute("deepgram.websocket.v2.sequence_id", sequence_id)
    if event := getattr(message, "event", None):
        span.set_attribute("deepgram.websocket.v2.event", str(event))
    if turn_index := getattr(message, "turn_index", None):
        span.set_attribute("deepgram.websocket.v2.turn_index", turn_index)
    if audio_window_start := getattr(message, "audio_window_start", None):
        span.set_attribute("deepgram.websocket.v2.audio_window_start", audio_window_start)
    if audio_window_end := getattr(message, "audio_window_end", None):
        span.set_attribute("deepgram.websocket.v2.audio_window_end", audio_window_end)
    if transcript := getattr(message, "transcript", None):
        span.set_attribute("gen_ai.completion.0.role", "Transcribed Text")
        span.set_attribute("gen_ai.completion.0.content", transcript)
    if end_of_turn_confidence := getattr(message, "end_of_turn_confidence", None):
        span.set_attribute("deepgram.websocket.v2.end_of_turn_confidence", end_of_turn_confidence)


def _set_websocket_v2_fatal_error_event_attributes(span: Span, message: Any) -> None:
    """
    Handle ListenV2FatalErrorEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.v2.fatal_error")

    if sequence_id := getattr(message, "sequence_id", None):
        span.set_attribute("deepgram.websocket.v2.sequence_id", sequence_id)
    if code := getattr(message, "code", None):
        span.set_attribute("deepgram.websocket.v2.error.code", str(code))
    if description := getattr(message, "description", None):
        span.set_attribute("deepgram.websocket.v2.error.description", str(description))


def _set_agent_v1_welcome_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1WelcomeMessage WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.welcome")

    if request_id := getattr(message, "request_id", None):
        span.set_attribute("gen_ai.response.request_id", str(request_id))


def _set_agent_v1_history_message_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1HistoryMessage WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.history")

    if role := getattr(message, "role", None):
        span.set_attribute("deepgram.agent.history.role", str(role))
    if content := getattr(message, "content", None):
        span.set_attribute("deepgram.agent.history.content", str(content))


def _set_agent_v1_conversation_text_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1ConversationTextEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.conversation_text")

    if role := getattr(message, "role", None):
        span.set_attribute("deepgram.agent.conversation.role", str(role))
    if content := getattr(message, "content", None):
        span.set_attribute("deepgram.agent.conversation.content", str(content))


def _set_agent_v1_agent_thinking_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1AgentThinkingEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.thinking")

    if content := getattr(message, "content", None):
        span.set_attribute("deepgram.agent.thinking.content", str(content))


def _set_agent_v1_agent_started_speaking_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1AgentStartedSpeakingEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.started_speaking")

    if total_latency := getattr(message, "total_latency", None):
        span.set_attribute("deepgram.agent.latency.total", total_latency)
    if tts_latency := getattr(message, "tts_latency", None):
        span.set_attribute("deepgram.agent.latency.tts", tts_latency)
    if ttt_latency := getattr(message, "ttt_latency", None):
        span.set_attribute("deepgram.agent.latency.ttt", ttt_latency)


def _set_agent_v1_injection_refused_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1InjectionRefusedEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.injection_refused")

    if msg := getattr(message, "message", None):
        span.set_attribute("deepgram.agent.injection_refused.message", str(msg))


def _set_agent_v1_error_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1ErrorEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.error")

    if description := getattr(message, "description", None):
        span.set_attribute("deepgram.agent.error.description", str(description))
    if code := getattr(message, "code", None):
        span.set_attribute("deepgram.agent.error.code", str(code))


def _set_agent_v1_warning_attributes(span: Span, message: Any) -> None:
    """
    Handle AgentV1WarningEvent WebSocket message.

    Args:
        span: The span to set the attributes on.
        message: The message to extract the attributes from.
    """
    span.add_event("deepgram.agent.warning")

    if description := getattr(message, "description", None):
        span.set_attribute("deepgram.agent.warning.description", str(description))
    if code := getattr(message, "code", None):
        span.set_attribute("deepgram.agent.warning.code", str(code))


def _set_http_response_attributes(span: Span, response: Any) -> None:
    """
    Handle HTTP API response attributes (transcribe_url, transcribe_file, analyze, generate).

    Args:
        span: The span to set the attributes on.
        response: The response to extract the attributes from.
    """
    if results := getattr(response, "results", None):
        if channels := getattr(results, "channels", None):
            first_channel = next(iter(channels))
            if alternatives := getattr(first_channel, "alternatives", None):
                first_alt = next(iter(alternatives))
                if transcript := getattr(first_alt, "transcript", None):
                    span.set_attribute("gen_ai.completion.0.role", "Transcribed Text")
                    span.set_attribute("gen_ai.completion.0.content", transcript)

        if summary := getattr(results, "summary", None):
            if result := getattr(summary, "result", None):
                span.set_attribute("gen_ai.response.summary.result", result)
            if text := getattr(summary, "text", None):
                span.set_attribute("gen_ai.response.summary.text", text)
            if short := getattr(summary, "short", None):
                span.set_attribute("gen_ai.response.summary.short", short)

        if topics := getattr(results, "topics", None):
            if segments := getattr(topics, "segments", None):
                for i, segment in enumerate(segments):
                    span.set_attribute("gen_ai.response.topics", json.dumps(segment))

        if intents := getattr(results, "intents", None):
            if segments := getattr(intents, "segments", None):
                for i, segment in enumerate(segments):
                    span.set_attribute("gen_ai.response.intents", json.dumps(segment))

        if sentiments := getattr(results, "sentiments", None):
            if average := getattr(sentiments, "average", None):
                if sentiment := getattr(average, "sentiment", None):
                    span.set_attribute("gen_ai.response.sentiments.average", sentiment)
                if sentiment_score := getattr(average, "sentiment_score", None):
                    span.set_attribute("gen_ai.response.sentiments.average_score", sentiment_score)

    if metadata := getattr(response, "metadata", None):
        if request_id := getattr(metadata, "request_id", None):
            span.set_attribute("gen_ai.response.request_id", str(request_id))

        if duration := getattr(metadata, "duration", None):
            span.set_attribute("gen_ai.audio.duration", duration / 60)

        if channel_count := getattr(metadata, "channels", None):
            span.set_attribute("gen_ai.response.channels", channel_count)

        if models := getattr(metadata, "models", None):
            span.set_attribute("gen_ai.response.model_list", list(models) if not isinstance(models, str) else [models])

        if model_info := getattr(metadata, "model_info", None):
            if isinstance(model_info, dict) and model_info:
                first_model_key = next(iter(model_info))
                info = model_info.get(first_model_key, {})
            if name := info.get("name", None):
                span.set_attribute("gen_ai.request.model", str(name))
            if version := info.get("version", None):
                span.set_attribute("gen_ai.request.model_version", str(version))
            if arch := info.get("arch", None):
                span.set_attribute("gen_ai.request.model_arch", str(arch))

        if summary_info := getattr(metadata, "summary_info", None):
            if model_uuid := getattr(summary_info, "model_uuid", None) or summary_info.get("model_uuid", None):
                span.set_attribute("gen_ai.response.summary.model_uuid", str(model_uuid))
            if summary_input_tokens := getattr(summary_info, "input_tokens", None) or summary_info.get(
                "input_tokens", None
            ):
                span.set_attribute("gen_ai.usage.summary.prompt_tokens", str(summary_input_tokens))
            if summary_output_tokens := getattr(summary_info, "output_tokens", None) or summary_info.get(
                "output_tokens", None
            ):
                span.set_attribute("gen_ai.usage.summary.completion_tokens", str(summary_output_tokens))
        if sentiment_info := getattr(metadata, "sentiment_info", None):
            if model_uuid := getattr(sentiment_info, "model_uuid", None) or sentiment_info.get("model_uuid", None):
                span.set_attribute("gen_ai.response.sentiment.model_uuid", str(model_uuid))
            if sentiment_input_tokens := getattr(sentiment_info, "input_tokens", None) or sentiment_info.get(
                "input_tokens", None
            ):
                span.set_attribute("gen_ai.usage.sentiment.prompt_tokens", str(sentiment_input_tokens))
            if sentiment_output_tokens := getattr(sentiment_info, "output_tokens", None) or sentiment_info.get(
                "output_tokens", None
            ):
                span.set_attribute("gen_ai.usage.sentiment.completion_tokens", str(sentiment_output_tokens))

        if topics_info := getattr(metadata, "topics_info", None):
            if model_uuid := getattr(topics_info, "model_uuid", None) or topics_info.get("model_uuid", None):
                span.set_attribute("gen_ai.response.topics.model_uuid", str(model_uuid))
            if topics_input_tokens := getattr(topics_info, "input_tokens", None) or topics_info.get(
                "input_tokens", None
            ):
                span.set_attribute("gen_ai.usage.topics.prompt_tokens", str(topics_input_tokens))
            if topics_output_tokens := getattr(topics_info, "output_tokens", None) or topics_info.get(
                "output_tokens", None
            ):
                span.set_attribute("gen_ai.usage.topics.completion_tokens", str(topics_output_tokens))

        if intents_info := getattr(metadata, "intents_info", None):
            if model_uuid := getattr(intents_info, "model_uuid", None) or intents_info.get("model_uuid", None):
                span.set_attribute("gen_ai.response.intents.model_uuid", str(model_uuid))
            if intents_input_tokens := getattr(intents_info, "input_tokens", None) or intents_info.get(
                "input_tokens", None
            ):
                span.set_attribute("gen_ai.usage.intents.prompt_tokens", str(intents_input_tokens))
            if intents_output_tokens := getattr(intents_info, "output_tokens", None) or intents_info.get(
                "output_tokens", None
            ):
                span.set_attribute("gen_ai.usage.intents.completion_tokens", str(intents_output_tokens))


def set_response_attributes(span: Span, response: Any) -> None:
    """
    Set the response attributes for the span.

    Args:
        span: The span to set the attributes on.
        response: The response to extract the attributes from.
    """
    if not span.is_recording():
        return

    try:
        msg_type = getattr(response, "type", None)

        # ListenV1 WebSocket message types
        if msg_type == "Results":
            _set_websocket_results_event_attributes(span, response)
        elif msg_type == "Metadata":
            _set_websocket_metadata_event_attributes(span, response)
        elif msg_type == "UtteranceEnd":
            _set_websocket_utterance_end_event_attributes(span, response)
        elif msg_type == "SpeechStarted":
            _set_websocket_speech_started_event_attributes(span, response)

        # ListenV2 WebSocket message types
        elif msg_type == "Connected":
            _set_websocket_v2_connected_event_attributes(span, response)
        elif msg_type == "TurnInfo":
            _set_websocket_v2_turn_info_event_attributes(span, response)
        elif msg_type == "FatalError":
            _set_websocket_v2_fatal_error_event_attributes(span, response)

        # AgentV1 WebSocket message types
        elif msg_type == "Welcome":
            _set_agent_v1_welcome_attributes(span, response)
        elif msg_type == "History":
            _set_agent_v1_history_message_attributes(span, response)
        elif msg_type == "ConversationText":
            _set_agent_v1_conversation_text_attributes(span, response)
        elif msg_type == "AgentThinking":
            _set_agent_v1_agent_thinking_attributes(span, response)
        elif msg_type == "AgentStartedSpeaking":
            _set_agent_v1_agent_started_speaking_attributes(span, response)
        elif msg_type == "InjectionRefused":
            _set_agent_v1_injection_refused_attributes(span, response)
        elif msg_type == "Error":
            _set_agent_v1_error_attributes(span, response)
        elif msg_type == "Warning":
            _set_agent_v1_warning_attributes(span, response)
        else:
            _set_http_response_attributes(span, response)

        span.set_attribute("netra.span.type", "GENERATION")

    except Exception as e:
        logger.debug("Failed to set Deepgram response attributes: %s", e)
