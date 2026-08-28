"""Pure mapping tables and helpers for the LiveKit span processor.

Free of OTel *tracing* imports so the mapping rules can be unit-tested as plain
data — the one exception is Netra's own ``SpanType`` enum, imported so the
``netra.span.type`` values here cannot drift from the vocabulary every other
Netra instrumentation stamps. Every table here was checked against
``livekit-agents`` 1.6.7 (``telemetry/trace_types.py`` for the attribute names,
``telemetry/traces.py`` ``_chat_ctx_to_otel_events`` for the event shape).

Layout: constants, then the types the mapping tables are built from, then the
tables themselves, then one section of helpers per payload LiveKit produces
(span attributes, conversation events, chat contexts, TTS metrics).
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from netra.span_wrapper import SpanType

# ---------------------------------------------------------------------------
# LiveKit instrumentation scope
# ---------------------------------------------------------------------------

# livekit-agents' OTel instrumentation scope. Everything this package does is
# gated on it: our processors are registered process-wide and must not touch a
# span from any other instrumentation.
LIVEKIT_SCOPE_NAME = "livekit-agents"

# ---------------------------------------------------------------------------
# Span names
# ---------------------------------------------------------------------------

# Netra's own span for a whole call — the one span in this package Netra creates
# rather than annotates. See ``call_span.py``. Kebab-case where every LiveKit span
# is snake_case, deliberately: the name says at a glance which side authored it.
CALL_SPAN_NAME = "livekit-call"

# livekit-agents' own root span for a job (``ipc/job_proc_lazy_main.py``:
# ``_traceable_entrypoint``). It ends when the user's entrypoint coroutine returns
# — normally moments after ``session.start()`` — so it is a poor trace root for a
# call that runs for minutes. ``call_span.py`` reparents it under ``livekit-call``.
JOB_ENTRYPOINT_SPAN_NAME = "job_entrypoint"

# livekit-agents' span for one ``AgentSession``, opened inside ``start()`` and
# ended inside ``_aclose_impl``. Its end is the authoritative "the call is over"
# signal this package ends ``livekit-call`` on.
AGENT_SESSION_SPAN_NAME = "agent_session"

# livekit-agents' span for one turn of agent speech (``voice/agent_activity.py``,
# opened by each of the three reply tasks). It is the span the dispatched agent's
# name is stamped on — see ``NETRA_AGENT_NAME``.
AGENT_TURN_SPAN_NAME = "agent_turn"

# livekit-agents' span for one turn of user speech (``voice/audio_recognition.py``:
# ``_ensure_user_turn_span``), carrying the transcript and the STT model. It is
# where this package puts the transcription usage LiveKit reports out-of-band —
# there is no STT span below it to carry them, and pricing needs the usage on the
# same span as the model.
USER_TURN_SPAN_NAME = "user_turn"

# The speaking spans LiveKit creates for each run of speech. Due to a context
# propagation bug in livekit-agents, ``user_speaking`` is sometimes created
# without an explicit OTel context, causing it to be orphaned (no parent).
SPEAKING_SPAN_NAMES = frozenset({"user_speaking", "agent_speaking"})

# ---------------------------------------------------------------------------
# Netra target attribute keys
# ---------------------------------------------------------------------------

NETRA_TOOL_NAME = "netra.tool.name"

# The dispatched agent's name, written on every ``agent_turn`` span. Same key the
# ``@agent`` decorator emits (``SessionManager.get_current_entity_attributes``), so
# a voice turn names its agent the way every other Netra agent span does.
#
# The value is the *worker dispatch* name — ``JobContext.job.agent_name``, which
# LiveKit itself writes as ``lk.agent_name`` on ``agent_session`` and
# ``job_entrypoint`` but not on the turns. It is NOT the per-``Agent`` label
# (``lk.agent_label``), so it does not change when a session hands off between
# agents, and it is absent for a worker that declares no ``agent_name`` — LiveKit
# leaves the field empty for automatic dispatch, and an empty name is not written.
NETRA_AGENT_NAME = "netra.agent.name"
NETRA_USAGE_SOURCE = "netra.usage.source"
USAGE_SOURCE_FRAMEWORK = "framework"

# The ``netra.span.type`` every other Netra instrumentation stamps
# (``hermes_agent``, ``google_adk``, ``agno``, ``claude_agent_sdk``). The only
# span-type contract this package emits: LiveKit spans carry no package-local
# ``span_type`` attribute.
NETRA_SPAN_TYPE = "netra.span.type"

# ``SpanType`` has no TTS or STT member, so the audio spans take this default
# rather than being given a value that means something else.
DEFAULT_NETRA_SPAN_TYPE = SpanType.SPAN

# The entity marker the ``@workflow``/``@agent``/``@task`` decorators stamp
# (``netra/decorators.py:_add_span_attributes``) and that ``agno`` emits as
# ``ATTR_ENTITY``. Separate from ``netra.span.type``: ``SpanType`` has no
# ``WORKFLOW`` member, so the workflow marking rides on the entity contract while
# the span type stays at its ``SPAN`` default.
NETRA_ENTITY_TYPE = "netra.entity.type"
ENTITY_TYPE_WORKFLOW = "workflow"

# The audio marker, written on the interaction-level spans named in
# ``AUDIO_TYPE_BY_SPAN_NAME`` rather than on every LiveKit span. Its value says at
# which granularity the call audio for that span is addressable: the whole call
# (``session``) versus a single turn (``span``).
NETRA_AUDIO_TYPE = "netra.audio.type"
AUDIO_TYPE_SESSION = "session"
AUDIO_TYPE_SPAN = "span"

# The reason the session closed, stamped on the ``livekit-call`` span by
# ``wrap_aclose``. Values mirror LiveKit's ``CloseReason`` enum.
NETRA_CLOSE_REASON = "netra.livekit.close_reason"

# ---------------------------------------------------------------------------
# The gen_ai conventions this package emits into
# ---------------------------------------------------------------------------

# The conversation-attribute convention SpanIOProcessor already consumes
# (``_PROMPT_RE`` in netra/processors/span_io_processor.py). Emitting into this
# shape rather than inventing a third convention is what makes voice turns render
# like every other LLM span.
GEN_AI_PROMPT_ROLE = "gen_ai.prompt.{index}.role"
GEN_AI_PROMPT_CONTENT = "gen_ai.prompt.{index}.content"

# The completion-side counterpart (``_COMPLETION_RE`` in the same processor),
# which fills ``output``.
GEN_AI_COMPLETION_ROLE = "gen_ai.completion.{index}.role"
GEN_AI_COMPLETION_CONTENT = "gen_ai.completion.{index}.content"

# Marks a span as one an LLM-aware instrumentation wrote. Gates the verbatim
# ``input``/``output`` fallback in ``messages_for_parent``: without it, an
# ``HTTP POST`` span under ``llm_request_run`` (netra/instrumentation/httpx/utils.py:124
# writes the URL, headers and body into ``input``) would be copied up as if it
# were a user message.
GEN_AI_ATTRIBUTE_PREFIX = "gen_ai."

# Prefix identifying token-usage attributes, whoever wrote them.
GEN_AI_USAGE_PREFIX = "gen_ai.usage."

# The keys Netra's backend prices a speech call from. Which of them a given model
# is billed on is the backend's decision, not ours: a model's price rows name one
# usage type each (``character_count``, ``input``/``output``, ``audio_duration``),
# and a value with no matching row simply does not price. So every value LiveKit
# reports is written, and the model is written alongside them because pricing needs
# the model and its usage on the *same* span.
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_CHARACTER_COUNT = "gen_ai.usage.prompt.character_count"
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"

# Billable audio length, in **seconds**. The unit is the backend's contract, not a
# choice: its price rows carry the divisor (``unitValue`` 60 for a per-minute
# price, 3600 for a per-hour one) and apply it to a value it reads as seconds.
GEN_AI_AUDIO_DURATION = "gen_ai.audio.duration"

# The assembled input/output ``SpanIOProcessor`` builds from the indexed pairs. Read
# off a child span as the fallback when it carries no indexed pairs of its own.
INPUT_ATTRIBUTE = "input"
OUTPUT_ATTRIBUTE = "output"

# The fallback carries text with no role attached, so one has to be supplied. Named
# for what the side means to the parent: its request and its reply.
FALLBACK_PROMPT_ROLE = "user"
FALLBACK_COMPLETION_ROLE = "assistant"

# The most conversation messages this package writes onto one span, per side.
#
# A span's attribute capacity is bounded — ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT``,
# default 128 — and OTel's ``BoundedAttributes`` evicts the *oldest* entry on
# overflow. So an unbounded conversation does not merely truncate itself: it
# silently deletes the attributes written earliest, which are exactly the markers
# ``SpanMappingProcessor.on_start`` and ``SessionSpanProcessor`` stamp
# (``netra.span.type``, ``netra.instrumentation.name``, ``netra.session_id``).
#
# Two LiveKit sources grow with the length of the call, both verified against
# livekit-agents 1.6.7:
#   * ``lk.chat_ctx`` on ``llm_node`` — the whole serialised ChatContext
#     (``voice/generation.py``);
#   * one conversation event per context item on ``llm_request``
#     (``llm/llm.py`` -> ``_chat_ctx_to_otel_events``).
# Each message costs two attributes (role + content), so without a cap a ~30-turn
# call is enough to evict every marker.
#
# 20 per side is 80 attributes at worst, which leaves the markers, LiveKit's own
# attributes and the latencies inside the default budget. Nothing is lost that is
# not still on the span: the full context remains verbatim in ``lk.chat_ctx``.
# LiveKit bounds its own ``eou_detection`` context the same way, via
# ``_EOU_MAX_HISTORY_TURNS``.
MAX_CONVERSATION_MESSAGES_PER_SIDE = 20

# Marks a span whose conversation was cut short by the cap above, so the
# truncation is visible on the span rather than silent. Written once per span.
NETRA_CONVERSATION_TRUNCATED = "netra.conversation.truncated"

# ---------------------------------------------------------------------------
# LiveKit attributes this package reads by name
# ---------------------------------------------------------------------------

# The attribute holding a serialised ``ChatContext`` (on ``llm_node`` and
# ``eou_detection``). Expanded into indexed ``gen_ai.prompt.*`` attributes rather
# than mirrored verbatim into ``input``: the raw JSON also contains non-message
# items (``agent_config_update``, handoffs) and reads as an opaque blob.
CHAT_CTX_ATTRIBUTE = "lk.chat_ctx"

# LiveKit's serialised ``TTSMetrics`` (``trace_types.ATTR_TTS_METRICS``, written on
# ``tts_request``). It is the only place on that span carrying the values pricing
# needs — ``characters_count``, ``input_tokens``/``output_tokens``,
# ``audio_duration`` and the model name nested under ``metadata`` — and as one
# opaque JSON blob the backend can read none of them. The sibling ``tts_node`` span
# does carry ``gen_ai.request.model``, but pricing needs the model and the usage on
# the *same* span.
TTS_METRICS_ATTRIBUTE = "lk.tts_metrics"

# The ``type`` discriminator on a serialised ``STTMetrics`` (``metrics/base.py``).
# LiveKit has no ``ATTR_STT_METRICS`` — unlike LLM and TTS metrics, the STT ones are
# never written onto a span — so they are read off the session's ``metrics_collected``
# event instead, and this is what distinguishes them from the other metrics types
# that same event carries. Matched by value rather than by ``isinstance``: the SDK
# stays importable with livekit-agents absent.
STT_METRICS_TYPE = "stt_metrics"

# LiveKit's completion event (``trace_types.EVENT_GEN_AI_CHOICE``, emitted from
# ``llm/llm.py`` once the reply is complete). Handled separately from
# ``EVENT_ROLE`` because it carries the model's reply and so belongs in the
# completion convention — without it, ``llm_request`` exports with an empty
# ``output`` even though the reply is right there on the span.
EVENT_CHOICE = "gen_ai.choice"

# Role LiveKit puts on the choice event; only used if the event omits it.
DEFAULT_CHOICE_ROLE = "assistant"

# The job identifiers LiveKit stamps on ``job_entrypoint``
# (``trace_types.ATTR_JOB_ID`` / ``ATTR_ROOM_NAME``). ``call_span.py`` writes the
# same two keys on ``livekit-call`` so the call's own root names the job and room
# it belongs to, read from the job context rather than copied off LiveKit's span.
LK_JOB_ID_ATTRIBUTE = "lk.job_id"
LK_ROOM_NAME_ATTRIBUTE = "lk.room_name"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ConversationSide(Enum):
    """Which half of the ``gen_ai`` conversation convention a value belongs to.

    ``PROMPT`` feeds ``input``, ``COMPLETION`` feeds ``output`` — the assembly
    happens in ``SpanIOProcessor``, which this package only has to emit into.
    """

    PROMPT = "prompt"
    COMPLETION = "completion"


class ConversationTarget(NamedTuple):
    """The gen_ai slot an ``lk.*`` content attribute is mirrored into.

    Attributes:
        side: Whether the text is a prompt message or a completion message.
        role: The conversation role to stamp alongside the text. Named for the
            *speaker*, not for the span's position in the pipeline, so a chat
            preview assembled from these attributes reads as the actual dialogue.
    """

    side: ConversationSide
    role: str


class ConversationMessage(NamedTuple):
    """One message to append to a span's indexed gen_ai sequences.

    Attributes:
        side: Which sequence the message belongs in.
        role: The conversation role to stamp alongside the text.
        content: The message text.
    """

    side: ConversationSide
    role: str
    content: str


class SpanConversation(NamedTuple):
    """The conversation content read back off a finished span.

    Attributes:
        prompts: The ``(role, content)`` pairs of the indexed prompt sequence, in
            index order.
        completions: The same for the completion sequence.
        raw_input: The span's assembled ``input``, but only when it carries no
            indexed prompt pairs — otherwise it is derived from them and copying
            both would duplicate the conversation.
        raw_output: The same for ``output``.
        carries_gen_ai: Whether the span has any ``gen_ai.*`` attribute at all,
            i.e. whether an LLM-aware instrumentation wrote it.
    """

    prompts: List[Tuple[str, str]]
    completions: List[Tuple[str, str]]
    raw_input: Optional[str]
    raw_output: Optional[str]
    carries_gen_ai: bool


class AudioPricingAttributes(NamedTuple):
    """The billable facts of one speech call, as LiveKit reported them.

    Shared by synthesis and transcription because LiveKit reports both the same
    way: ``TTSMetrics`` and ``STTMetrics`` differ only in that the latter has no
    character count. Which fields actually price is the backend's decision — see
    ``GEN_AI_REQUEST_MODEL`` and the keys beside it.

    Every field is ``None`` when LiveKit reported nothing, or reported a value of
    zero: a zero prices to nothing and claims a measurement nobody made. That
    matters in practice — a provider billed by characters reports
    ``input_tokens: 0``, and a streaming STT connection reports an
    ``audio_duration`` of 0.0 purely to record when the socket was acquired.

    Attributes:
        model: The model, verbatim from LiveKit — including the
            ``provider/model`` prefix it uses for its inference gateway
            (``cartesia/sonic-3``).
        character_count: The number of characters synthesised. Always ``None`` for
            transcription, which reports no such count.
        prompt_tokens: Input tokens — synthesised text for TTS, input audio for STT.
        completion_tokens: Output tokens — output audio for TTS, transcribed text
            for STT.
        audio_duration: The billable audio length in seconds.
    """

    model: Optional[str]
    character_count: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    audio_duration: Optional[float]


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# lk.* -> Netra key. Additive: the original lk.* attribute is always preserved.
#
# Conversation *content* is deliberately absent from this table — see
# ``CONVERSATION_MAP``, which routes it through the indexed ``gen_ai.*``
# convention instead of writing ``input``/``output`` directly.
ATTRIBUTE_MAP: Dict[str, str] = {
    # Function tools
    "lk.function_tool.name": NETRA_TOOL_NAME,
    "lk.function_tool.arguments": INPUT_ATTRIBUTE,
    "lk.function_tool.output": OUTPUT_ATTRIBUTE,
    # Latencies, all in seconds
    "lk.response.ttft": "netra.latency.ttft",
    "lk.response.ttfb": "netra.latency.ttfb",
    "lk.e2e_latency": "netra.latency.e2e",
    "lk.end_of_turn_delay": "netra.latency.end_of_turn_delay",
    # Turn quality
    "lk.transcript_confidence": "netra.stt.confidence",
    "lk.interrupted": "netra.turn.interrupted",
}

# lk.* content attribute -> gen_ai slot. One table for every LiveKit span:
# ``lk.response.text`` means the same thing on ``agent_turn`` as it does on
# ``llm_node``, so nothing here needs gating on the span name.
#
# Additive: the original lk.* attribute is always preserved, and these never write
# ``input``/``output`` directly — indexed ``gen_ai.prompt.*``/``gen_ai.completion.*``
# attributes are emitted instead, the same convention Netra's own provider
# instrumentations use, which is what lets ``SpanIOProcessor`` assemble a
# multi-message ``input``.
CONVERSATION_MAP: Dict[str, ConversationTarget] = {
    # agent_turn: the system instructions in force for the turn. LiveKit writes it
    # before ``lk.user_input``, so it lands at prompt index 0 and the assembled
    # ``input`` reads [system, user] like any other LLM span.
    "lk.instructions": ConversationTarget(ConversationSide.PROMPT, "system"),
    # agent_turn: the utterance that opened the turn (LiveKit sets this only when a
    # new message did).
    "lk.user_input": ConversationTarget(ConversationSide.PROMPT, "user"),
    # agent_turn and llm_node: the generated reply.
    "lk.response.text": ConversationTarget(ConversationSide.COMPLETION, "assistant"),
    # tts_request: the text handed to the TTS provider. The words are the agent's.
    "lk.input_text": ConversationTarget(ConversationSide.PROMPT, "assistant"),
    # user_turn: the STT transcript — the output of the transcription. The words are
    # the caller's.
    "lk.user_transcript": ConversationTarget(ConversationSide.COMPLETION, "user"),
}

# span name -> ``netra.span.type``.
NETRA_SPAN_TYPE_BY_NAME: Dict[str, SpanType] = {
    "agent_turn": SpanType.AGENT,
    "llm_node": SpanType.GENERATION,
    "llm_request": SpanType.GENERATION,
    "function_tool": SpanType.TOOL,
    "tts_request": SpanType.GENERATION,
}

# span name -> ``netra.entity.type``. Empty by design, and kept rather than
# removed because it is the hook for classifying a future LiveKit span.
#
# ``job_entrypoint`` used to be listed here as the workflow: it was the trace root,
# so it was the one span wrapping everything the user's entrypoint did. It no
# longer is — ``livekit-call`` roots the trace and ``job_entrypoint`` is one of its
# two children — so the workflow marker moved onto ``livekit-call``
# (``call_span.py``), leaving exactly one workflow entity per voice trace.
NETRA_ENTITY_TYPE_BY_NAME: Dict[str, str] = {}

# LiveKit span name -> the ``netra.audio.type`` value it carries. Matched against
# the LiveKit span name, so only spans this package already gates on (scope
# ``livekit-agents``) are eligible — a nested provider span such as ``openai.chat``
# never reaches the lookup.
AUDIO_TYPE_BY_SPAN_NAME: Dict[str, str] = {
    "agent_session": AUDIO_TYPE_SESSION,
    "agent_turn": AUDIO_TYPE_SPAN,
    "user_turn": AUDIO_TYPE_SPAN,
}

# LiveKit spans that carry no conversation content of their own: the text exists
# only on a direct child. ``llm_request_run`` wraps the provider call, so the
# prompt and completion are on the provider's own span (``openai.chat`` and
# friends, a *non*-LiveKit scope); ``tts_node`` wraps the synthesis, so the text
# is on ``tts_request``. See ``SpanMappingProcessor.on_end``.
#
# Verified against livekit-agents 1.6.7 that in both cases the child ends while
# the parent is still recording: the provider span ends inside
# ``LLMStream._run()``, and ``tts_request`` is ended by the ``async with
# wrapped_tts.stream()`` exit inside the generator ``_tts_inference_task``
# iterates.
IO_FROM_CHILD_SPAN_NAMES = frozenset({"llm_request_run", "tts_node"})

# LiveKit conversation event name -> gen_ai role. From ``trace_types.EVENT_*``;
# note LiveKit folds OpenAI's ``developer`` role into the system message event.
# These are all request-side messages, hence the prompt convention.
EVENT_ROLE: Dict[str, str] = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}

# ---------------------------------------------------------------------------
# Payload field names (private)
# ---------------------------------------------------------------------------

# Reads an indexed conversation attribute back off a span, for the child-to-parent
# propagation in ``conversation_from_attributes``. The plural forms match what
# ``SpanIOProcessor`` accepts, so a child written by any instrumentation in the SDK
# is readable here.
_INDEXED_MESSAGE_RE = re.compile(r"^gen_ai\.(prompt|completion)s?\.(\d+)\.(role|content)$")

_PROMPT_GROUP = "prompt"
_ROLE_FIELD = "role"
_CONTENT_FIELD = "content"

# The attribute key LiveKit puts message text under in a conversation event
# (``_chat_ctx_to_otel_events``: ``{"content": item.raw_text_content or ""}``).
_EVENT_CONTENT_KEY = "content"
_EVENT_ROLE_KEY = "role"

# On the choice event, a tool-only reply carries no ``content`` — the requested
# calls are the whole output. LiveKit sends them as a list of JSON strings.
_EVENT_TOOL_CALLS_KEY = "tool_calls"

_CHAT_CTX_ITEMS_KEY = "items"
_CHAT_CTX_MESSAGE_TYPE = "message"
_CHAT_CTX_TYPE_KEY = "type"
_CHAT_CTX_ROLE_KEY = "role"
_CHAT_CTX_CONTENT_KEY = "content"

_TTS_METRICS_CHARACTERS_KEY = "characters_count"

# Shared by ``TTSMetrics`` and ``STTMetrics``.
_METRICS_METADATA_KEY = "metadata"
_METRICS_MODEL_KEY = "model_name"
_METRICS_INPUT_TOKENS_KEY = "input_tokens"
_METRICS_OUTPUT_TOKENS_KEY = "output_tokens"
_METRICS_AUDIO_DURATION_KEY = "audio_duration"


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def is_absent(value: Any) -> bool:
    """Whether *value* should be treated as "not set".

    Empty and missing values are treated as absent so a mapped write can never
    blank out a value another processor supplied.

    Args:
        value: The candidate attribute value.

    Returns:
        True if the value carries no information.
    """
    return value is None or value == ""


def as_attribute_text(value: Any) -> str:
    """Render a value as the string an OTel text attribute needs.

    Args:
        value: The value LiveKit wrote, or one read back off a span.

    Returns:
        The value unchanged if it is already a string, else its ``str()``.
    """
    return value if isinstance(value, str) else str(value)


def is_usage_attribute(key: str) -> bool:
    """Whether *key* is a token-usage attribute.

    Args:
        key: An attribute name.

    Returns:
        True for ``gen_ai.usage.*`` keys.
    """
    return key.startswith(GEN_AI_USAGE_PREFIX)


def is_zero_usage(value: Any) -> bool:
    """Whether a usage value is a zero that is worse than no value at all.

    A framework that cannot surface token counts reports 0 rather than omitting
    them, and livekit-agents forwards ``metrics.prompt_tokens`` verbatim — so a
    custom LLM node that does not report usage produces
    ``gen_ai.usage.input_tokens = 0``. Writing that claims a measurement nobody
    made, and the real counts are on the provider span underneath. Dropping the
    attribute lets the provider's numbers stand unopposed.

    Args:
        value: The candidate usage value.

    Returns:
        True for a numeric zero; False for every other value, including ``None``
        and booleans.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return False


def netra_span_type_for(span_name: Optional[str]) -> str:
    """Return the ``netra.span.type`` value for a LiveKit span name.

    Args:
        span_name: The LiveKit span's name, or ``None``.

    Returns:
        A ``SpanType`` value — ``AGENT``, ``GENERATION``, ``TOOL``, or ``SPAN``.
    """
    return NETRA_SPAN_TYPE_BY_NAME.get(span_name or "", DEFAULT_NETRA_SPAN_TYPE).value


# ---------------------------------------------------------------------------
# Conversation content on spans
# ---------------------------------------------------------------------------


def conversation_from_attributes(attributes: Optional[Mapping[str, Any]]) -> SpanConversation:
    """Read a finished span's conversation content back out of its attributes.

    The inverse of what this package (and every other Netra instrumentation) writes
    when it emits indexed ``gen_ai.prompt.*``/``gen_ai.completion.*`` pairs, so a
    child span's conversation can be re-emitted onto its parent.

    An entry carrying a role but no text is skipped — a role alone is not a message.
    An entry carrying text but no role keeps the text; the caller supplies a role.

    Args:
        attributes: The finished span's attributes, or ``None``.

    Returns:
        The prompt and completion pairs in index order, the verbatim
        ``input``/``output`` for the sides that have no pairs, and whether the span
        carries any ``gen_ai.*`` attribute. A malformed or empty mapping yields an
        empty result rather than an error — a mapping failure must never break the
        user's trace.
    """
    prompt_entries: Dict[int, Dict[str, str]] = {}
    completion_entries: Dict[int, Dict[str, str]] = {}
    raw_input: Optional[str] = None
    raw_output: Optional[str] = None
    carries_gen_ai = False

    for key, value in (attributes or {}).items():
        if key == INPUT_ATTRIBUTE:
            raw_input = None if is_absent(value) else as_attribute_text(value)
            continue
        if key == OUTPUT_ATTRIBUTE:
            raw_output = None if is_absent(value) else as_attribute_text(value)
            continue
        if not key.startswith(GEN_AI_ATTRIBUTE_PREFIX):
            continue
        carries_gen_ai = True
        match = _INDEXED_MESSAGE_RE.match(key)
        if match is None:
            continue
        entries = prompt_entries if match.group(1) == _PROMPT_GROUP else completion_entries
        entries.setdefault(int(match.group(2)), {})[match.group(3)] = as_attribute_text(value)

    prompts = _ordered_messages(prompt_entries)
    completions = _ordered_messages(completion_entries)
    return SpanConversation(
        prompts=prompts,
        completions=completions,
        raw_input=raw_input if not prompts else None,
        raw_output=raw_output if not completions else None,
        carries_gen_ai=carries_gen_ai,
    )


def messages_for_parent(conversation: SpanConversation, *, allow_raw_io: bool) -> List[ConversationMessage]:
    """Turn a child's conversation into the messages to append to its parent.

    Args:
        conversation: What ``conversation_from_attributes`` read off the child.
        allow_raw_io: Whether the verbatim ``input``/``output`` fallback may be
            used. False for a child no LLM-aware instrumentation touched, whose
            ``input`` is something else entirely — an HTTP request envelope, a SQL
            statement — and would read as a fabricated user message on the parent.

    Returns:
        The messages to append, prompts first. Empty when the child carried no
        conversation.
    """
    messages = [
        ConversationMessage(ConversationSide.PROMPT, role or FALLBACK_PROMPT_ROLE, content)
        for role, content in conversation.prompts
    ]
    messages.extend(
        ConversationMessage(ConversationSide.COMPLETION, role or FALLBACK_COMPLETION_ROLE, content)
        for role, content in conversation.completions
    )
    if not allow_raw_io:
        return messages

    if conversation.raw_input is not None:
        messages.append(ConversationMessage(ConversationSide.PROMPT, FALLBACK_PROMPT_ROLE, conversation.raw_input))
    if conversation.raw_output is not None:
        messages.append(
            ConversationMessage(ConversationSide.COMPLETION, FALLBACK_COMPLETION_ROLE, conversation.raw_output)
        )
    return messages


def _ordered_messages(entries: Mapping[int, Mapping[str, str]]) -> List[Tuple[str, str]]:
    """Flatten indexed role/content entries into ``(role, content)`` pairs.

    Args:
        entries: Index -> the fields collected for that index.

    Returns:
        The pairs in index order, skipping any index that carried no text. The
        indices themselves are discarded: the caller re-numbers against the
        parent's own counters.
    """
    pairs: List[Tuple[str, str]] = []
    for index in sorted(entries):
        fields = entries[index]
        content = fields.get(_CONTENT_FIELD)
        if content is None or content == "":
            continue
        pairs.append((fields.get(_ROLE_FIELD) or "", content))
    return pairs


# ---------------------------------------------------------------------------
# LiveKit conversation events
# ---------------------------------------------------------------------------


def content_of_event(attributes: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Extract the message text from a LiveKit conversation event's attributes.

    Args:
        attributes: The event attributes LiveKit passed to ``add_event``.

    Returns:
        The message text, or ``None`` when the event carries none. A
        ``function_call`` event legitimately has no ``content`` — its payload is
        already on the ``function_tool`` span as ``lk.function_tool.*``, so
        returning ``None`` here drops nothing from the trace.
    """
    if not attributes:
        return None
    content = attributes.get(_EVENT_CONTENT_KEY)
    if is_absent(content):
        return None
    return as_attribute_text(content)


def content_of_choice_event(attributes: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Extract the reply text from LiveKit's ``gen_ai.choice`` event.

    Args:
        attributes: The event attributes LiveKit passed to ``add_event``.

    Returns:
        The reply text; the serialised tool calls when the reply was tool-only;
        or ``None`` when the event carries neither.
    """
    if not attributes:
        return None

    content = attributes.get(_EVENT_CONTENT_KEY)
    if not is_absent(content):
        return as_attribute_text(content)

    return _joined_tool_calls(attributes.get(_EVENT_TOOL_CALLS_KEY))


def role_of_choice_event(attributes: Optional[Mapping[str, Any]]) -> str:
    """Return the role LiveKit put on a ``gen_ai.choice`` event.

    Args:
        attributes: The event attributes LiveKit passed to ``add_event``.

    Returns:
        The event's ``role``, or ``assistant`` when it carries none.
    """
    role = (attributes or {}).get(_EVENT_ROLE_KEY)
    if isinstance(role, str) and role:
        return role
    return DEFAULT_CHOICE_ROLE


def _joined_tool_calls(tool_calls: Any) -> Optional[str]:
    """Render LiveKit's list of JSON tool-call strings as one JSON array.

    Args:
        tool_calls: The event's ``tool_calls`` value.

    Returns:
        A JSON array string, or ``None`` when there is nothing to render.
    """
    if isinstance(tool_calls, str):
        return tool_calls or None
    if not isinstance(tool_calls, (list, tuple)) or not tool_calls:
        return None
    # Each element is already a JSON object string, so concatenating them into an
    # array yields valid JSON.
    return "[" + ", ".join(str(call) for call in tool_calls) + "]"


# ---------------------------------------------------------------------------
# LiveKit chat contexts
# ---------------------------------------------------------------------------


def messages_from_chat_ctx(payload: Any) -> List[Tuple[str, str]]:
    """Extract ``(role, text)`` pairs from a serialised LiveKit ``ChatContext``.

    Accepts either the JSON string LiveKit puts in ``lk.chat_ctx`` or the dict
    ``ChatContext.to_dict()`` returns, so the same rules apply whether the
    context is read off a span or off a live session.

    Non-message items (``agent_config_update``, ``function_call``,
    ``agent_handoff``, ...) are skipped: they are not conversation turns, and
    their payloads are already on the spans that produced them.

    Args:
        payload: A ``ChatContext`` JSON string or dict.

    Returns:
        The conversation turns in order. Empty when *payload* is malformed,
        which is treated as "no messages" rather than an error — a mapping
        failure must never break the user's trace.
    """
    items = _chat_ctx_items(payload)

    messages: List[Tuple[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if item.get(_CHAT_CTX_TYPE_KEY) != _CHAT_CTX_MESSAGE_TYPE:
            continue
        role = item.get(_CHAT_CTX_ROLE_KEY)
        if not isinstance(role, str) or not role:
            continue
        text = _text_of_chat_content(item.get(_CHAT_CTX_CONTENT_KEY))
        if text is None:
            continue
        messages.append((role, text))

    return messages


def _chat_ctx_items(payload: Any) -> List[Any]:
    """Decode a ``ChatContext`` payload down to its item list.

    Args:
        payload: A ``ChatContext`` JSON string or mapping.

    Returns:
        The raw items, or an empty list for any payload that is not a decodable
        ``ChatContext``.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return []

    if not isinstance(payload, Mapping):
        return []

    items = payload.get(_CHAT_CTX_ITEMS_KEY)
    return items if isinstance(items, list) else []


def _text_of_chat_content(content: Any) -> Optional[str]:
    """Join the text parts of a ``ChatContext`` message's ``content``.

    Mirrors ``ChatMessage.raw_text_content`` in livekit-agents
    (``llm/chat_context.py``): string parts joined by newline, non-text parts
    (image/audio content objects) skipped.

    Args:
        content: The item's ``content`` value.

    Returns:
        The message text, or ``None`` when the item carries none.
    """
    if isinstance(content, str):
        return content or None
    if not isinstance(content, list):
        return None

    parts = [part for part in content if isinstance(part, str) and part]
    if not parts:
        return None
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LiveKit speech metrics
# ---------------------------------------------------------------------------

_NO_PRICING = AudioPricingAttributes(None, None, None, None, None)


def tts_pricing_attributes_from(payload: Any) -> AudioPricingAttributes:
    """Extract the priceable fields from a serialised LiveKit ``TTSMetrics``.

    Accepts either the JSON string LiveKit puts in ``lk.tts_metrics`` or the
    equivalent dict, so the same rules apply whether the metrics are read off a
    span or off a live ``TTSMetrics.model_dump()``.

    Args:
        payload: A ``TTSMetrics`` JSON string or mapping.

    Returns:
        The billable facts, each field ``None`` when absent or zero. Malformed
        input yields all ``None`` rather than an error — a mapping failure must
        never break the user's trace.
    """
    metrics = _as_metrics_mapping(payload)
    if metrics is None:
        return _NO_PRICING

    return _pricing_from(metrics)._replace(
        character_count=_positive_count(metrics.get(_TTS_METRICS_CHARACTERS_KEY)),
    )


def stt_pricing_attributes_from(payload: Any) -> AudioPricingAttributes:
    """Extract the priceable fields from a serialised LiveKit ``STTMetrics``.

    ``character_count`` is always ``None``: transcription reports no such count.

    Args:
        payload: An ``STTMetrics`` JSON string or mapping.

    Returns:
        The billable facts, each field ``None`` when absent or zero. Malformed
        input yields all ``None`` rather than an error.
    """
    metrics = _as_metrics_mapping(payload)
    if metrics is None:
        return _NO_PRICING

    return _pricing_from(metrics)


def _as_metrics_mapping(payload: Any) -> Optional[Mapping[str, Any]]:
    """Read a serialised metrics payload as a mapping.

    Args:
        payload: A metrics JSON string or mapping.

    Returns:
        The mapping, or ``None`` if the payload is neither.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None

    return payload if isinstance(payload, Mapping) else None


def _pricing_from(metrics: Mapping[str, Any]) -> AudioPricingAttributes:
    """Read the fields ``TTSMetrics`` and ``STTMetrics`` report identically.

    Args:
        metrics: A parsed metrics mapping.

    Returns:
        The billable facts less ``character_count``, which is TTS-only.
    """
    metadata = metrics.get(_METRICS_METADATA_KEY)
    model = metadata.get(_METRICS_MODEL_KEY) if isinstance(metadata, Mapping) else None

    return AudioPricingAttributes(
        model=model if isinstance(model, str) and model else None,
        character_count=None,
        prompt_tokens=_positive_count(metrics.get(_METRICS_INPUT_TOKENS_KEY)),
        completion_tokens=_positive_count(metrics.get(_METRICS_OUTPUT_TOKENS_KEY)),
        audio_duration=_positive_duration(metrics.get(_METRICS_AUDIO_DURATION_KEY)),
    )


def _positive_count(value: Any) -> Optional[int]:
    """Coerce a reported count to a positive int, or None if it is not one.

    A zero or negative count is treated as absent: it prices to nothing and would
    only claim a measurement that says less than no attribute at all.

    Args:
        value: The candidate count.

    Returns:
        The count as an int, or ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return int(value)


def _positive_duration(value: Any) -> Optional[float]:
    """Coerce a reported duration to a positive float, or None if it is not one.

    Absent for the same reason as a zero count, and zero arrives here routinely: a
    streaming STT reports ``audio_duration = 0.0`` on connection acquisition purely
    to record the socket timing.

    Args:
        value: The candidate duration, in seconds.

    Returns:
        The duration as a float, or ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)
