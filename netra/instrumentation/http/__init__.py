"""Pieces shared by the HTTP instrumentations that record request and response data.

* ``headers`` — the one set of headers that must never be recorded, and the two
  shapes it is applied to (string mapping, raw ASGI byte pairs). Used by
  ``httpx``, ``requests``, ``fastapi`` and ``agno``, which each carried a
  private copy of the same frozenset until it moved here. Four copies of a
  redaction policy is a credential leak waiting on the next divergence, which
  is why the set now lives in exactly one place.
* ``body``    — request and response bodies onto a span within the attribute
  budget. Used by ``httpx``, ``requests`` and ``fastapi``.

Two HTTP instrumentations deliberately do not appear above. ``aiohttp`` records
no headers or bodies at all -- it builds a header dict only to inject trace
context -- so it has nothing to share. ``agno`` shares ``headers`` but still
carries its own AgentOS body handling, which is unbounded and spells its binary
placeholder ``<binary: N bytes>`` where this package writes
``<binary content: N bytes>``; folding it in is outstanding work, not a
deliberate exception.
"""
