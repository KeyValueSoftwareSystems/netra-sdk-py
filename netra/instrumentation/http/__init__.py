"""Pieces shared by every HTTP instrumentation.

``httpx``, ``requests``, ``fastapi``, ``aiohttp`` and ``agno`` all record request
and response data on spans, and used to each carry their own copy of how:

* ``headers`` — the one set of headers that must never be recorded, and the two
  shapes it is applied to (string mapping, raw ASGI byte pairs)
* ``body``    — request and response bodies onto a span within the attribute budget

Four private copies of a redaction policy is a credential leak waiting on the
next divergence, which is why the header set lives in exactly one place.
"""
