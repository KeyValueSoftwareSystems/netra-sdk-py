# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [1.0.0] - 2026-08-23
- **Label evaluation/simulation traces on the root span** - Root spans produced by evaluation test runs (`Netra.evaluation`) and simulation runs (`Netra.simulation`) now carry a `netra.trace.origin` attribute set to `evaluation`, letting the backend and frontend distinguish these traces from normal workflow invocations.


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

- **Add opt-in TTL caching for `get_prompt`** - `Netra.prompts.get_prompt` now accepts `use_cache` and `cache_ttl` parameters for in-memory caching. Default TTL is `PROMPT_CACHE_TTL_SECONDS` (60); override per call with `cache_ttl`. Use `Netra.prompts.clear_cache()` to invalidate cached entries.
- **Add opt-in TTL caching for `get_model_pricing`** - `Netra.models.get_model_pricing` now accepts `use_cache` and `cache_ttl` parameters for in-memory caching. Default TTL is `MODEL_PRICING_CACHE_TTL_SECONDS` (300); override per call with `cache_ttl`. Use `Netra.models.clear_cache()` to invalidate cached entries.

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

[0.1.99]: https://github.com/KeyValueSoftwareSystems/netra-sdk-py/tree/main
