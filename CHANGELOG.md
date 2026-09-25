# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [1.1.0] - 2026-09-25

First stable release of the 1.1.0 line. Everything below shipped across `1.1.0b1`–`1.1.0b4`,
plus the changes that landed after `1.1.0b4`: `NETRA_REDACT_HEADERS`, redaction of upstream
header attributes, the generator-exit and non-recording-span fixes, and the `json-repair`
security fix under **Security**.

### Added

- **LiveKit voice-agent instrumentation** - `InstrumentSet.LIVEKIT` traces `livekit-agents` sessions, normalizing LiveKit's `lk.*` span tree into Netra's shape: a `livekit-call` root holding `agent_session`, then `user_turn` / `agent_turn` per exchange with the `llm_node`, `stt_node` and `tts_node` spans beneath. Enabled by default, and applied when `livekit.agents` is first imported. `agent_turn` spans are typed as agents and stamped with `netra.agent.name`; `llm_node` spans are typed as generations. LiveKit is also in the default root allow-list, since `agent_session` is the root of every voice trace — removing it peels the whole voice tree and leaves the provider spans underneath as orphaned roots.

- **Call audio capture for LiveKit sessions** - Both speakers' audio is batched and streamed to Netra's audio ingest endpoint alongside the trace, each chunk correlated to the span it belongs to by parent span ID. **This is on by default**: the endpoint resolves to `<otlp_endpoint>/v1/audio/chunk` whenever an auth credential (`x-api-key` or `Authorization`) is configured, so an existing LiveKit deployment that upgrades will begin sending call audio. Capture is all of a call's audio or none of it, never one speaker. Settings are environment-only, with no `Netra.init()` parameter: `NETRA_AUDIO_ENDPOINT` overrides the URL, and `NETRA_AUDIO_BATCH_BYTES` (32768), `NETRA_AUDIO_BATCH_INTERVAL_MS` (1000), `NETRA_AUDIO_BUFFER_BYTES` (2097152) and `NETRA_AUDIO_MAX_REQUEST_BYTES` (262144) tune batching; out-of-range values log a warning and fall back to the default. There is no separate audio toggle — capture is part of the LiveKit instrumentation, so `Netra.init(block_instruments={InstrumentSet.LIVEKIT})` is what turns it off. The SDK logs at INFO whether capture resolved ON or OFF, and warns when an endpoint resolves but no credential is present (capture stays off).

- **Audio duration and token usage on LiveKit spans** - Voice spans now carry `gen_ai.audio.duration` alongside `gen_ai.usage.prompt_tokens` and `gen_ai.usage.completion_tokens`, so audio-billed turns report usage the same way text turns do.

- **`get_session_details` accepts an optional time window** - `Netra.dashboard.get_session_details(session_id, start_time=..., end_time=...)` now takes two optional ISO 8601 UTC timestamps and forwards them to the session details endpoint as `startTime` / `endTime` query parameters. Each bound is independent: `start_time` alone leaves the window open-ended on the right, `end_time` alone open-ended on the left, and omitting both is exactly the previous behavior — the request is byte-identical to before, down to the absence of a query string. **The response shape is unchanged** (`{"sessionId": ..., "traces": [...]}`, each trace carrying the same `tokens`, `cost`, `models` and `toolCalls` keys), so existing callers need no changes.

  Timestamps must carry milliseconds and a `Z` suffix (`2026-09-01T00:00:00.000Z`). The backend rejects any other shape with a 400, which the SDK logs and surfaces as `None`, matching how every other dashboard utility reports a failed request. Note that `datetime.isoformat()` does not produce this format.

  The window filters individual spans before they are aggregated per trace, so a trace straddling a bound is returned with its tokens, cost, latency, models and tool calls computed from its in-window spans only, rather than being dropped.

  **Requires a backend carrying the matching change.** An older backend ignores the two parameters rather than rejecting them, so a windowed call against one silently returns the whole session. Separately, on an updated backend the endpoint now validates its query string, so an unrecognized query parameter — something no SDK version sends — returns a 400 where it was previously ignored.

- **Opt-in TTL caching for `Netra.prompts.get_prompt`** - `get_prompt(name, label, use_cache=..., cache_ttl=...)` can now serve a prompt version from a process-local in-memory cache instead of calling the backend. **It is off by default** (`use_cache=False`), so a call that does not name it is byte-identical to before — same request, same return value. With `use_cache=True` the entry is keyed on name *and* label, so `production` and `staging` of the same prompt never share a slot. The default lifetime is `PROMPT_CACHE_TTL_SECONDS` (60 s); `cache_ttl` overrides it for that call only, and a value of zero or less stores nothing rather than caching forever. Only a successful lookup is cached — `None` and `{}` are what the client returns for a failed or missing prompt, and neither is stored, so an outage cannot pin a hole in the cache for a full TTL.

  Entries expire against a monotonic clock, so a system clock adjustment cannot extend or collapse a TTL. The cache is per-process and per-initialized SDK, never shared between processes, so a prompt edited in the dashboard can still be served from an older revision by any worker whose entry has not yet expired — the TTL is the staleness bound. `Netra.prompts.clear_cache()` drops every entry, and `Netra.shutdown()` now clears it too.

- **Opt-in TTL caching for `Netra.models.get_model_pricing`** - `get_model_pricing(name=None, use_cache=..., cache_ttl=...)` takes the same two parameters and is likewise **off by default**. The key covers the `name` filter, with unfiltered calls under their own key, so a filtered and an unfiltered call do not serve each other's results. The default lifetime is `MODEL_PRICING_CACHE_TTL_SECONDS` (300 s) — longer than prompts, since pricing tables move rarely — again overridable per call via `cache_ttl` and skipped entirely for a non-positive value. What is cached is the unwrapped `data` list, matching what the method returns; the client's failure sentinel and a response whose `data` is not a list both return early without writing, so only a well-formed response is stored.

  **A cache hit returns the same list and dicts as the previous hit, not a copy.** Mutating a returned entry — sorting it in place, editing a price — corrupts what every later caller sees until the TTL elapses; copy before mutating. `Netra.models.clear_cache()` drops every entry, and `Netra.shutdown()` clears it too.

- **`NETRA_REDACT_HEADERS` extends the set of redacted HTTP headers** - A comma-separated list of header names (case-insensitive, surrounding whitespace ignored) merged with the built-in set: `authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key`, `x-auth-token` and `proxy-authorization`. The merged set applies both to Netra's own HTTP header capture and to the header attributes recorded by upstream OTel instrumentations (see **Security**). The variable can only be set in the environment; there is no `Netra.init()` parameter. It is read when `Netra.init()` builds the config, so the list cannot be changed after initialization, and it can only add headers, never remove a built-in one.

### Changed

- **`init_instrumentations()` no longer takes `base64_image_uploader`** - Netra hosts no image store, so the only call site had always passed `None` through four layers to reach traceloop, where the parameter is typed as required and mistyped besides (three arguments here, four in traceloop). It is now passed as `None` at the traceloop boundary and gone from the SDK's own signature. Internal helper; `Netra.init()` is unaffected.

- **A named instrumentation with no instrumentor now warns instead of logging at debug** - `Netra.init(instruments={InstrumentSet.PYRAMID})` is a no-op — no Pyramid instrumentor ships with the SDK — and said so only at `DEBUG`. Naming one explicitly now logs a warning. An `InstrumentSet.ALL` expansion still logs at debug, since it sweeps in six such members every time.

- **`netra.instrumentation.lazy` is now `netra.instrumentation.wiring.deferral`**, matching the noun-per-module naming of its siblings, which moved alongside it into `netra.instrumentation.wiring` (`selection`, `registry`, `activation`, `triggers`). Internal modules.

- **`netra.instrumentation` is now four subpackages rather than a flat directory** - the 25 per-library instrumentors moved to `netra.instrumentation.libraries.<library>`, `http_body` split into `netra.instrumentation.capture` (`bounded_capture`, `stream_formats`, `stream_utils`) and `netra.instrumentation.http` (`headers`, `body`), and `utils` became `span_utils` so it no longer reads as a sibling of `opentelemetry.instrumentation.utils` at import sites. `netra.instrumentation.instruments` is unchanged and remains the public path for `InstrumentSet`; the exported OpenTelemetry scope name of every instrumentor is unchanged too, now pinned in a `_TRACER_NAME` constant rather than derived from `__name__`, with a test that fails if one drifts. All internal modules.

- **`netra.utils.TRUNCATION_MARKER_KEY` now lives in `netra.instrumentation.capture.bounded_capture`**, next to the code that stamps it. Still importable from `netra.utils`, and the marker string itself is unchanged.

### Fixed

- **`requests` spans no longer lose their whole `output` attribute on an empty streaming response** - a `stream=True` response whose body carried no bytes left `requests` with nothing to replay, and reading the body back to record it raised `RuntimeError` instead of returning empty. That took the status code and headers down with the body, so an empty SSE stream or a bodiless chunked response produced a span with no `output` at all. The body state is now checked rather than the read attempted.

- **`httpx` and `requests` now agree on the shape of a bodiless stream** - `httpx` recorded `"body": ""` where `requests` omitted the key, for the same response. Both now omit it, matching the non-streaming path: a stream that yielded nothing is bodiless, not a body that happens to be empty.

- **`CustomInstruments`, `InstrumentSet` and `DEFAULT_INSTRUMENTS` are importable from `netra.instrumentation` again** - all three were reachable as `from netra.instrumentation import ...` before activation was split out of that module in 1.0.1b1, and the split dropped them without intending to. Re-exported. The supported public path remains `from netra import NetraInstruments`.

- **Session attributes now fall back to a span's declared parent context** - `SessionSpanProcessor` read `session_id`, `user_id`, `tenant_id` and custom keys from the ambient context only. A span started with an explicit `context=` is not created *inside* that context — the SDK fires `on_start` before making it current — so the ambient context belongs to whichever task happened to create the span and may carry no session baggage at all. LiveKit does exactly this, parenting every `agent_turn` onto a context snapshotted when the session started, so a turn triggered from outside the session's task tree was the one span in the trace missing `netra.session_id`. Each key now falls back to the parent context, ambient-first, so the fallback can only supply a value that was missing and never overrides one a later `Netra.set_session_id()` resolved.

- **Two instrumentations were listed twice in the trigger table** - `ASYNCIO` and `SQLITE3` each had a duplicate row in `INSTRUMENT_TRIGGERS`. The duplicated values were identical so nothing was mistriggered, but the later row silently wins, and pyflakes' `F601` only fires when repeated keys have *different* values — so an edit to either copy would have been dropped without warning. Deduplicated, with a test that parses the source to catch a recurrence.

- **LiveKit user-speech onset is no longer missing from call audio** - VAD opens the `user_speaking` span ~200–300 ms after the caller actually starts talking, so frames that arrived in that window were tagged as noise (or dropped) because no speaking span was active yet. User frames captured while no `user_speaking` span is open are now held in a short pre-speech buffer (~500 ms / ~26 KB at the livekit-agents default rate). When the span opens, the buffer is split at LiveKit's VAD-backdated `start_time`: frames at or after that onset are attributed to the speaking span; earlier frames stay noise. Frames older than the window flush as noise so long silence between turns still records. The audio coordinator is also registered before `sender.start()` yields to the event loop, so speaking spans created during that await are visible and not mis-tagged as noise.

- **Mid-speech hangups no longer orphan `user_speaking` under a missing `user_turn`** - When the caller hangs up mid-utterance, LiveKit's `_aclose_impl` can end `user_speaking` without ending its parent `user_turn` (the turn never reached end-of-utterance). An unended span is never queued by `BatchSpanProcessor`, so the backend received `user_speaking` with a `parent_id` that resolved to nothing. After LiveKit's close path returns, the SDK now ends any still-recording `user_turn` for that call and stamps `netra.turn.interrupted_by_session_close`. Streaming `user_input_transcribed` events are buffered for the open turn (finals append, interims overlay, matching LiveKit's own accumulation) and stamped as `lk.user_transcript` on that forced end — LiveKit only writes that attribute when EOU commits the turn, so a mid-speech hangup previously exported the turn with no words even though the LiveKit UI had already shown them. `lk.pii.user_transcript` is also accepted in the conversation map so older and newer `livekit-agents` agree.

- **Closing a generator early no longer marks its span as an error** - When a generator paused at a `yield` inside `with Netra.start_span(...)` is closed with `.close()`, or garbage-collected before it is exhausted, Python raises `GeneratorExit` at the `yield`. `SpanWrapper` recorded that as an exception, so a consumer that simply stopped iterating produced an errored span. `GeneratorExit` is now treated as normal completion. `StopIteration` and `StopAsyncIteration` are still recorded as errors: a `for` loop consumes them internally, so one reaching the span means a manual `next()` call went wrong.

- **Decorators skip attribute capture on non-recording spans, and bound the cost of serializing binary values** - `@workflow`, `@agent`, `@task` and `@span` now return early when the span is not recording (sampled out, for example), instead of serializing arguments and return values that would be discarded. Bytes-like values (`bytes`, `bytearray`, `memoryview`) are sliced to the 1000-character attribute limit *before* they are decoded, so a 30 MB payload no longer builds a ~120 MB string only to truncate it, and JSON serialization of lists and dicts stops once it reaches the limit. **Bytes are now decoded as UTF-8 rather than passed through `str()`**: a `b"hello"` argument is recorded as `hello` where it was previously `b'hello'`. Bytes that are not valid UTF-8 are recorded as an empty string (as `null` inside a list or dict) and a warning is logged, matching OTel's own attribute cleaning.

### Removed

- **`TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA`** - the set could never match anything: eight of its twelve names belong to `InstrumentSet` members tagged `_Origin.CUSTOM` (which never reach traceloop selection) and the other four name no member at all. The invariant it was meant to protect — that Netra's own instrumentations are never also delegated to traceloop — is enforced by `_Origin` and covered by `test_every_registered_instrumentor_belongs_to_the_custom_family`.

### Security

- **Sensitive headers recorded by upstream OTel instrumentations are now redacted** - When header capture is turned on for an upstream OpenTelemetry HTTP instrumentation, it records headers as `http.request.header.*` / `http.response.header.*` span attributes. Netra's header redaction only covered its own HTTP capture, so these attributes were exported exactly as recorded, including `Authorization`, `Cookie` and API-key values. `InstrumentationSpanProcessor` now replaces them with `[REDACTED]` in `on_end`, before the exporting processor runs. It matches both the built-in set and any names added through `NETRA_REDACT_HEADERS`, allowing for the upstream key format (`x-api-key` is recorded as `http.request.header.x_api_key`). Header attributes that are not on the list are left unchanged.

- **Raised the `json-repair` floor to `0.60.1`** ([GHSA-xf7x-x43h-rpqh](https://github.com/advisories/GHSA-xf7x-x43h-rpqh), CVSS 7.5) - versions below `0.60.1` resolve a circular `$ref` in a caller-supplied JSON Schema by following it in an unbounded loop, pinning CPU indefinitely. The dependency constraint was already a range (`>=0.44.1,<1.0.0`) rather than a hard pin, but the floor still allowed the vulnerable release to resolve, and it's what the currently published PyPI release hard-pins. `netra-sdk`'s only call site (`netra/utils.py::truncate_and_repair_json`) calls `repair_json(json_str)` without a `schema` argument, so this specific loop was never reachable through the SDK itself — this closes the dependency-scanner alert and removes the exposure for any consumer that might add schema-based repair later. No behavior change otherwise; `repair_json`'s signature is backward compatible for the arguments the SDK passes.

## [1.0.1] - 2026-09-01

First stable release of the 1.0.1 line. Everything in it shipped in `1.0.1b1` and
`1.0.1b2` — see those sections below for the full detail — plus one fix that had no
entry of its own:

### Fixed

- **`import netra` no longer fails on Python 3.11** - `InstrumentorSpec.constructor_kwargs`
  defaulted to a shared `mappingproxy` via `field(default=...)`. Python 3.11 changed the
  `dataclasses` default check to reject any default whose type is unhashable, and
  `mappingproxy` only became hashable in 3.12 — so on 3.11, and only on 3.11, building the
  instrumentor registry raised `ValueError: mutable default <class 'mappingproxy'>` at import
  time. The default is now supplied by a factory. 3.10 and 3.12+ were never affected.

## [1.0.1b2] - 2026-08-27

### Fixed

- **Root spans from instrumentations outside `root_instruments` are now dropped even when they have no children** - Standalone spans from non-root instrumentations — a bare `redis` command, `requests` call or `sqlalchemy` query that is its own trace root — leaked to the backend as root traces. The leak was intermittent: whether such a span was filtered depended on what else happened to share its export batch.

  The cause was where the candidacy marker lived. `RootInstrumentFilterProcessor` marked a root-block candidate by setting an instance attribute on the live span at `on_start`, but `Span.end()` hands span processors a fresh `ReadableSpan` built from a fixed field list, so the marker never reached the export batch. `FilteringSpanExporter` then found no in-batch candidate and fell back to the cross-batch candidate registry, which it consults only when some batch span's *parent* is a registered candidate — and a blocked root with no children in the batch is nobody's parent, so it matched nothing and was exported. The marker is now re-stamped at `on_end`, on the copy the exporter actually reads.

  Traces that were already being filtered correctly are unaffected, but anyone relying on the default `root_instruments` will see these standalone spans stop arriving. If the marker can ever no longer be stamped, the SDK now logs a warning instead of silently exporting every span.

## [1.0.1b1] - 2026-08-25

### Performance

- **Instrumentations are applied when their library is first imported, not during `Netra.init()`** - Each enabled instrumentation now registers a post-import hook (via `wrapt`, already a transitive dependency) instead of being applied up front. Applying an instrumentation means importing the library it patches, so `Netra.init()` previously paid for every LLM library present in the environment whether or not the process used any of them — around 3 s in a typical LLM venv. An instrumentation is now applied at the moment its target library is first imported, and never if that library is never imported. wrapt fires a hook immediately and synchronously when the module is already loaded, so this holds whether the client imports their library before or after `Netra.init()`.

  Total instrumentation work is strictly lower, but its position changes: the first `import openai` in a process that uses OpenAI becomes correspondingly slower. Objects a library builds during its own module execution keep unpatched bound methods — a limitation that already applied to any client importing their library before `Netra.init()`, now on the common path rather than the rare one.

### Fixed

- **Blocking one traceloop instrumentation no longer enables every other one** - `Netra.init(instruments={InstrumentSet.OPENAI}, block_instruments={InstrumentSet.ANTHROPIC})` previously enabled langchain, bedrock, vertexai and every other installed traceloop instrumentation. Selection inherited traceloop's "an empty instrument list means all of them" rule, and a request naming only Netra-backed instrumentations partitioned to an empty traceloop list — so adding a block list flipped the request into its opposite. A request now enables exactly what it names, minus what it blocks. This was also the only code path that imported `traceloop-sdk` during `Netra.init()`; selection is now free of it on every path.

- **Instrumentations gated on a module name rather than a distribution now apply** - `ASYNCIO`, `AWS_LAMBDA`, `LOGGING` and `SQLITE3` were gated on `asyncio`, `aws_lambda`, `logging` and `sqlite3`. Those are import names, not installed distributions, so the gate never matched and requesting one of these instrumentations was a silent no-op. They are now ungated, matching `THREADING` and `URLLIB`. Distribution gates are additionally matched per PEP 503, so a gate spelled with an underscore matches a distribution published with a hyphen — this revives `AIO_PIKA` (`aio_pika`) and `CEREBRAS` (`cerebras_cloud_sdk`), which had the same problem.

  **`CEREBRAS` is in `DEFAULT_INSTRUMENTS`.** Every other instrumentation named above is opt-in, but Cerebras is enabled by default, and its gate has never matched — `cerebras-cloud-sdk` is published with hyphens and the old check compared lower-cased names only. Any process on default configuration with the Cerebras SDK installed will run `NetraCerebrasInstrumentor` for the first time on upgrade. The remaining instrumentations here affect only callers who asked for them explicitly or passed `InstrumentSet.ALL`.

- **`AIOHTTP` is now actually instrumented when requested** - the instrumentor existed but was never reachable from the dispatch chain, so enabling `InstrumentSet.AIOHTTP` did nothing. It is now registered against `AioHttpClientInstrumentor`. Not in `DEFAULT_INSTRUMENTS`.

- **Concurrent instrumentation no longer corrupts `sys.stdout`/`sys.stderr`** - traceloop's "no valid instruments set" warning is suppressed by swapping the process streams. With activation deferred into the client's own `import`, two libraries first imported on two threads could interleave that swap and leave `sys.stdout` pointing at a discarded buffer for the rest of the process, silently swallowing every later `print` and traceback. Suppression is now depth-counted, so the last thread out restores the real streams whatever order they arrive in.

- **`InstrumentSet.PYRAMID`** no longer claims a trigger module; no Pyramid instrumentor ships with the SDK, so the entry implied support that did not exist.

- **`import netra` no longer imports `traceloop-sdk`** - `traceloop.sdk` costs ~620 ms to import (it pulls in pandas, numpy and aiohttp) and was reached from module scope in `netra/instrumentation/`, so every `import netra` paid it — including in processes that never call `Netra.init()`. Every traceloop symbol is now imported inside the function that needs it, and traceloop is loaded only when a traceloop-backed instrumentation actually activates. Measured: `import netra` 463 ms / 1033 modules to 275 ms / 760 modules.

### Breaking changes

- **The per-library `init_*_instrumentation()` helpers were removed from `netra.instrumentation`** - The ~60 functions of the form `init_openai_instrumentation()`, `init_redis_instrumentation()`, and so on have been replaced by a declarative table (`netra.instrumentation.registry.CUSTOM_INSTRUMENTORS`) that a single generic activator applies. They were an internal dispatch mechanism with no callers outside the package, and the set of instrumentations enabled by `Netra.init()` is unchanged. Code calling one directly should pass the corresponding `InstrumentSet` member to `Netra.init(instruments=...)` instead. `CustomInstruments` is retained but no longer keys anything inside the SDK; activation is keyed on `InstrumentSet`.

- **`InstrumentSet.<member>.origin` changed type** - `origin` was the enum *class* backing the member (`CustomInstruments` or `traceloop.sdk.Instruments`) and is now a member of the internal `_Origin` enum (`_Origin.CUSTOM` / `_Origin.TRACELOOP`). Tagging members with `traceloop.sdk.Instruments` forced the traceloop import onto every `import netra`. Code comparing `instrument.origin == CustomInstruments` must compare against `_Origin.CUSTOM` instead. `InstrumentSet` values, names and membership are unchanged.

## [1.0.0] - 2026-08-23

### Breaking changes

- **Remove `enable_root_span` configuration** - The `enable_root_span` option and the `NETRA_ENABLE_ROOT_SPAN` environment variable have been removed from `Netra.init()`. Netra no longer creates a long-lived process root span at initialization. Passing `enable_root_span` to `Netra.init()` is now a `TypeError`; remove the argument and the environment variable from your setup.

- **`create_dataset` parameter order changed** - `Netra.evaluation.create_dataset()` now takes `(name, dataset_type=DatasetType.TEXT, turn_type=TurnType.SINGLE, tags=None)`; `tags` moved from the second positional parameter to last. Positional calls such as `create_dataset("my-set", ["tag"])` now bind the tag list to `dataset_type` and must be updated to keyword form (`create_dataset("my-set", tags=["tag"])`). Keyword callers are unaffected.

### Features

- **Add `dataset_type` to the dataset creation utility** - `Netra.evaluation.create_dataset()` accepts a new `dataset_type` argument backed by the `DatasetType` enum (`TEXT`, `IMAGE`), exported from `netra.evaluation`, and forwards it to the backend. Defaults to `DatasetType.TEXT`, matching the previous behavior.

- **Propagate trace context across thread boundaries** - `InstrumentSet.THREADING` is now enabled by default, so spans created inside `threading.Thread` and `ThreadPoolExecutor` workers attach to the parent workflow trace instead of starting independent root traces. `SessionManager` span bookkeeping moved from shared mutable state to thread-isolated `ContextVar`s with copy-on-write updates, and entity frames are tracked through OTel context values rather than `attach`/`detach` token pairs, which required strict LIFO ordering that parallel workers cannot guarantee.

- **Label evaluation/simulation traces on the root span** - Root spans produced by evaluation test runs (`Netra.evaluation`) and simulation runs (`Netra.simulation`) now carry a `netra.trace.origin` attribute set to `evaluation`, letting the backend and frontend distinguish these traces from normal workflow invocations.

- **Add `escaped` flag to `record_exception`** - `Netra.record_exception(exception, attributes=..., escaped=...)` now forwards the OpenTelemetry `escaped` flag to the underlying span event, so callers can mark whether the exception escaped the instrumented scope. Defaults to `False`, preserving existing behavior.

- **Map `db.statement` to span input for DB instrumentations** - `SpanIOProcessor` now promotes the `db.statement` attribute (set by PyMySQL and other OTel DB instrumentations) into the canonical `input` attribute when `input` is still empty, so database spans show the SQL query in trace previews instead of blank input. `db.statement` is preserved as an attribute, and `output` is never populated from DB attributes, since query results and bound parameters are user data.

### Fixes

- **Fix `set_session_id` / `set_user_id` / `set_tenant_id` not stamping the active span** — These methods previously only set OTel baggage but did not set the attribute on the span that was active at call time. `SessionManager.set_session_context` now also stamps the currently recording span immediately, so the caller's span carries the identity attribute without relying solely on processor-based propagation.

- **Resolve attribute/conversation truncation limits at init time instead of import time** - `NETRA_ATTRIBUTE_MAX_LEN`, `NETRA_CONVERSATION_CONTENT_MAX_LEN`, and `TRIAL_BLOCK_DURATION_SECONDS` are now resolved when `Netra.init()` builds the config rather than when `netra` is first imported. Overrides applied before `init()` — including a late `load_dotenv()` placed after the `netra` import — are now honored, so import order no longer affects these limits. Invalid values fall back to the defaults (50000/50000/900).

- **Surface backend error messages in HTTP client logs** - Backend API clients (prompts, models, usage, dashboard, evaluation, simulation) now log the backend-provided error message (e.g. `Prompt 'X' not found`) on request failures instead of the raw HTTP exception string, via a shared `extract_error_message` helper in `netra.utils`. This also fixes a latent error in the dashboard and evaluation clients — including the session details endpoint — that could raise while handling a request that failed before a response was received.

- **Centralize session attribute key constants** — Introduced `ATTR_SESSION_ID`, `ATTR_USER_ID`, and `ATTR_TENANT_ID` constants in `session_manager.py` as the single source of truth for span-attribute keys. `SessionSpanProcessor` now imports these instead of constructing keys inline, eliminating duplication and divergence risk.

## [0.1.99] - 2026-08-19

- **Add support to capture cache-write tokens in OpenAI instrumentation** — The OpenAI instrumentor now extracts `cache_write_tokens` from `prompt_tokens_details` (or `input_tokens_details` for the Responses API) and maps it to the `gen_ai.usage.cache_creation_input_tokens` span attribute. This enables accurate cost calculation for new OpenAI models that report cache-write tokens separately, as well as OpenAI-compatible proxies that expose cache-write usage.

- **Add Honcho memory SDK instrumentation** — Added automatic sync, async, and streaming instrumentation for `honcho-ai` (>= 2.0.0), with declarative patching, unified wrappers, dynamic response serialization, OTel GenAI semantic conventions, and centralized instrumentation constants.

## [0.1.98] - 2026-08-03

- **Add `set_root_output_stream` utility for streaming output on root span** - New `Netra.set_root_output_stream(stream)` method that wraps a sync or async iterable so the accumulated output is automatically written to the root span's `netra.user.output` attribute when iteration ends. Works transparently with Netra-instrumented stream wrappers (extracting `_netra_output`) and generic iterables (concatenating chunks). All instrumentation streaming wrappers (OpenAI, Cerebras, Groq, LiteLLM, Google GenAI, Agno) now expose `_netra_stream_wrapper` and `_netra_output` for content extraction.

- **Refactor `SessionManager` input/output methods to use shared `serialize_value` utility** - Extracted a common `serialize_value` helper in `netra/utils.py` that serializes a value to a JSON string (for dicts/lists) or plain string, capped at `Config.ATTRIBUTE_MAX_LEN`. `set_input`, `set_output`, `set_root_input`, and `set_root_output` now use this instead of duplicating serialization logic inline.

## [0.1.97] - 2026-08-03

- **Add simulation lifecycle hooks** - Prescript/postscript support for multi-turn simulations via `SimulationHooks` (`before_all`, `before`, `after`, `after_all`). Hooks can return setup context passed into `BaseTask.run`, and the run uses a two-phase initialize / first-turn flow so hooks execute before any LLM spend. `before_all` failure aborts the run as `prescript_failed`; item `before` failure marks only that scenario; `after` / `after_all` failures are logged and do not affect status.

- **Add simulation `before_each` / `after_each` hooks** - Per-item lifecycle hooks that run for every dataset item. Execution order is `before_all` → `before_each` → item-specific `before` → task → item-specific `after` → `after_each` → `after_all`.

- **Use explicit `.description` for simulation hook metadata** - Hook descriptions sent to the backend now come from an explicit `.description` attribute on each hook function (aligned with the TypeScript SDK), instead of reading Python docstrings.

- **Fix OpenAI Responses API stream handling** — `Response API` instrumentation spans now include token usage and completion output when a stream ends due to reaching the token limit (`incomplete` status).

- **Fix Hermes Agent tool call duplication** - Registry tools that pass through both `_run_agent_tool_execution_middleware` and `handle_function_call` no longer produce duplicate spans. The middleware wrapper now claims tool_call_ids in a `_middleware_traced_ids` context var; `handle_function_call_wrapper` checks this set and passes through when the ID is already traced. Also updated result extraction to handle the `_ManagedToolResult` dataclass in addition to legacy tuples.

- **Hide empty user and assistant messages from OpenAI span preview** - OpenAI instrumentation now skips emitting prompt attributes for messages with empty or null content, empty function call entries, and reasoning/reasoning_summary items, keeping span previews clean and only showing meaningful conversation turns.

- **Add title generation instrumentation to Hermes Agent** - New `title_generation_wrapper` traces `agent.title_generator.generate_title` as a `hermes-agent.title_generation` workflow span. Since title generation runs in a daemon thread with no parent context, the wrapper's span becomes the trace root, correctly identifying these traces as title-generation workflows.

- **Add `get_session_details` dashboard wrapper** - New `Netra.dashboard.get_session_details(session_id)` method that calls the public session details endpoint and returns session traces with tokens, cost, models, and tool calls.

- **Add `USER_ID` to `SessionFilterField`** - Session stats and session summary queries can now filter by `user_id`.

- **Fix OpenAI streaming wrapper span lifecycle** - Made `_finalize_span()` idempotent with a `_span_ended` guard, added `close()` and `__del__()` to both sync and async wrappers so spans are properly finalized even on early exit or GC. `AsyncStreamingWrapper` now exposes `aclose()` per the async iterator protocol, with `close()` as an async alias for OpenAI SDK compatibility.

- **Add instrumentation for Hermes Agent** - New monkey-patching based instrumentation for the `hermes-agent` SDK (>= 0.17.0). Captures conversation runs, skill invocations (single, stacked, and bundle), tool executions, function calls, and approval gates as OpenTelemetry spans with full input/output attributes, token usage, and model metadata.

- **Fix span attributes in OpenAI instrumentation** - Assistant completions no longer emit empty entries when the model returns `content: null` alongside tool calls, request messages now correctly handle non-dictionary objects (such as Pydantic ChatCompletionMessage instances) by converting them with model_as_dict() instead of skipping them, and assistant `tool_calls` arrays as well as `tool_call_id` values on tool messages are now captured and serialized as indexed prompt and completion span attributes.

## [0.1.96] - 2026-07-23

- **Reparent children of blocked root instruments instead of dropping the subtree** - When an instrumentation is not allowed to emit root-level spans, its children are now re-parented onto the nearest valid ancestor rather than dropping the entire subtree, so downstream spans are preserved.

- **Add utility to explicitly record exceptions on a span** - New `Netra.record_exception(exception, attributes=...)` utility to attach a caught exception to the currently active span from within an `except` block. It adds a standard OpenTelemetry exception event (type, message, stacktrace), sets the span status to ERROR, and records the `netra.error_message` attribute.


## [0.1.95] - 2026-06-26

- **Added get_all_datasets with tag as optional param** - If tag is provided, we get details of all the datasets with that particular tag attached.

## [0.1.94] - 2026-06-22

- **Introduce synthetic usage spans to fix cost calculation in the Claude Agent SDK** — Create separate spans for each model's usage to provide more accurate cost reporting. If separate spans cannot be created, usage is recorded on the main span as a fallback.


## [0.1.93] - 2026-06-19
- **Fix the bypassing of attribute truncation in Google ADK and Agno instrumentations** - This enables universal truncation of attributes based on the default/env value provided by the user


## [0.1.92] - 2026-06-16
- **Add support for file handling in simulation workflow** - This provides support for passing files in simulation workflow to provide user context


## [0.1.91] - 2026-06-08

- **Expand SDK dependencies to include latest versions** - This provides support for new versions of OTel instrumentations and traceloop


## [0.1.90] - 2026-06-08

- **Add missing dependency of "opentelemetry-instrumentation-pymysql"** - This enables tracing of PyMySQL workflows using Netra


## [0.1.89] - 2026-05-29

- **Support for metadata alias for tokens in Anthropic instrumentation** - Capture various token alias for anthropic instrumentation


## [0.1.88] - 2026-05-20

- **Support for distributed tracing during sub-process invocation** - Auto instrument subprocess module to automatically set current context as traceparent in sub-process environment whenever a new sub-process is created. Update `Netra.init` to automatically activate context from traceparent if traceparent is found in current environment.

- **Add new utility `models` to fetch model pricing from Netra** - Add SDK utility `get_model_pricing` to fetch model details and their pricing from Netra

- **Add timestamp info of Time to First Token (TTFT) in LLM spans** - Add timestamp data of TTFT as a new attribute, `gen_ai.performance.time_to_first_token.timestamp`, in LLM spans from OpenAI, LiteLLM, Google GenAI, Cerebras, Claude Agent, Agno, ADK, and Groq

## [0.1.87] - 2026-05-20

- **Prioritize input and output attributes explicitly set by user over attributes from instrumentation.**
Users can be now overwrite the input and ouput attributes of spans created by instrumentations. The input and output values auto-captured by the instruments will be overwritten by values explicitly passed by users using the exposed utilities.

## [0.1.86] - 2026-05-15

- Modify instrument resolution in traceloop to manual transfer of instruments


## [0.1.85] - 2026-05-15

- Remove duplicate instrumentation from URLLIB3 and COHERE from traceloop

## [0.1.84] - 2026-05-14

- Update agno instrumentation to capture token usage for streaming llm spans
- Cleanup metadata for claude agent sdk spans
- Add time_to_first_token and relative_time_to_first_token for claude agent sdk


## [0.1.83] - 2026-05-04

- Implement custom instrumentation for Agno.


## [0.1.82] - 2026-04-21

- Refine custom ADK instrumentation to produce a cleaner trace hierarchy, include sufficient metadata, and eliminate duplicate spans.


## [0.1.81] - 2026-04-16

- Fix root span attachment issue in tracer provider


## [0.1.80] - 2026-04-16

- Add relative_time_to_first_token attribute on LLM spans
- Add time_to_first_token and relative_time_to_first_token for litellm instrumentation


## [0.1.79] - 2026-04-02

- Added version-safe check for _shutdown attribute in _JsonOTLPMetricExporter for compatability with opentelemetry libraries


## [0.1.78] - 2026-03-31

- Added descriptor based binding of class methods when using decorators.


## [0.1.77] - 2026-03-27

- Added custom-metric utility in SDK
- Added support for custom-metric in dashboard utility


## [0.1.76] - 2026-03-19

- Update block instrument functionality to correctly block Redis and SQLAlchemy
- Remove httpx based check for blocking url


## [0.1.75] - 2026-03-18

- Added custom instrumentation for Claude Agent SDK


## [0.1.74] - 2026-03-13

- Add utility for prompt management


## [0.1.73] - 2026-03-12

- Extended dependency support for opentelemetry and traceloop-sdk
- Added TTFT for Cerebras and Groq instrumentation


## [0.1.72] - 2026-02-24

- Fixed bug in blocking internal request calls

## [0.1.71] - 2026-02-24

- Lock all dependency versions to avoid conflicts

## [0.1.69] - 2026-02-19

- Added support for blocked URL pattern in span blocking utility
- Fixed bug in run item failure reporting when an exception is raised from netra agent

## [0.1.68] - 2026-02-17

- Added support for audio duration & character count metric in dashboard query

## [0.1.67] - 2026-02-06

- Added support for simulation utility to trigger multi-turn simulation

## [0.1.66] - 2026-02-02

- Added Service and Environment filter for session summary and session stats dashboard utilities

## [0.1.65] - 2026-01-27

- Added session summary and session stats dashboard utilities

## [0.1.64] - 2026-01-27

- Added session summary and session stats dashboard utilities

## [0.1.63] - 2026-01-21

- Added support for first token time in OpenAI & Google GenAI instrumentations

## [0.1.62] - 2026-01-19

- Fixed bug in dashboard query models
- Added support for auto evaluation
- Added support for turn-based evaluation

## [0.1.61] - 2026-01-14

- Added dashboard-query utility

## [0.1.60] - 2025-12-22

- Fixed conversation attribute handling to use OTel context first, then fallback to SessionManager spans
- Added backward compatability and bug fixes in ElevenLabs instrumentation
- Added utility for subscription based trace blocking

## [0.1.59] - 2025-12-15

- Added support for Cartesia, ElevenLabs and Deepgram voice agent instrumentations

## [0.1.58] - 2025-11-28

- Added support for explicit filter params in usage utilities

## [0.1.57] - 2025-11-28

- Added support for trace list, and span list in usage tracking utility

## [0.1.56] - 2025-11-26

- Extended usage tracking utility to support cost tracking

## [0.1.56] - 2025-11-26

- Extended usage tracking utility to support cost tracking

## [0.1.55] - 2025-11-20

- Added utility to get session and tenant based usage
- Refactored litellm instrumentation
- Fixed bug in capturing ADK tool call args

## [0.1.54] - 2025-11-18

- Added support for agent type in spans

## [0.1.53] - 2025-11-17

- Added custom instrumentation for ADK framework
- Refactored DSPy instrumentation

## [0.1.52] - 2025-11-11

- Fixed attribute max length issue

## [0.1.51] - 2025-11-10

- Added custom instrumentation for Cerebras framework
- Fixed bug in traceloop instrumentation

## [0.1.50] - 2025-11-07

- Added custom dataset and entries

## [0.1.49] - 2025-11-06

- Fixed token count calculation for OpenAI response API

## [0.1.48] - 2025-11-05

- Added custom instrumentation for Groq framework

## [0.1.47] - 2025-10-21

- Added support for existing tracer provider usage

## [0.1.46] - 2025-10-17

- Fixed exception during add conversation
- Added support for observation type in spans

## [0.1.45] - 2025-09-29

- Added utility to locally block specific spans within a particular span scope.

## [0.1.44] - 2025-09-29

- Added utility to globally block specific spans from being exported to the tracing backend.

## [0.1.43] - 2025-09-17

- Fixed conversation content length issue
- Added utils module to handle common tasks

## [0.1.42] - 2025-09-09

- Refactored conversation attribute format to be more consistent with OpenTelemetry

## [0.1.41] - 2025-09-09

- Refactored codebase to remove duplicate code

## [0.1.40] - 2025-09-08

- Added span level conversation support

## [0.1.39] - 2025-09-02

- Refactored code to remove duplicate code

## [0.1.38] - 2025-09-02

- Fixed instrumentation name detection issue

## [0.1.37] - 2025-09-01

- Fixed context detachment issue in session manager

## [0.1.36] - 2025-09-01

- Added a trace level method set_prompt to set prompt on any active span

## [0.1.35] - 2025-09-01

- Patch fix for set_input and set_output methods to set attributes on root span if no span is provided
- Patch fix to create streaming aware decorators

## [0.1.34] - 2025-08-29

- Changed block spans from being exported to block root level spans from being exported

## [0.1.33] - 2025-08-29

- Added utility to block specific spans from being exported to the tracing backend.
- Fixed context detachment issue in span wrapper.

## [0.1.32] - 2025-08-28

- Added support for scrubbing sensitive data from spans.

## [0.1.31] - 2025-08-28

- Added custom instrumentation for LiteLLM framework

## [0.1.30] - 2025-08-27

- Added utility to set input and output data for any active span in a trace

[1.1.0]: https://github.com/KeyValueSoftwareSystems/netra-sdk-py/tree/main
