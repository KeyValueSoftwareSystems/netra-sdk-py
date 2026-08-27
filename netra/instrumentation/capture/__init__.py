"""Bounding and shaping the values instrumentations record on spans.

An instrumentation records data whose size the application, not the SDK,
controls: a streamed response body, an LLM completion, an agent's output. These
modules keep that from costing unbounded memory or arriving on the span as a
mid-token slice:

* ``bounded_capture`` — bounded buffers and budgeted serialization; transport-agnostic
* ``stream_formats``  — parsing SSE, NDJSON and concatenated JSON off a captured stream
* ``stream_utils``    — wrapping a single-pass stream so its output is committed on exhaustion

Nothing here knows what produced the value, so HTTP bodies, LLM token streams
and agent output share one implementation.
"""
