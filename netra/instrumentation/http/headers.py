"""The one list of HTTP headers Netra must never record, and how to apply it.

Four instrumentations record request or response headers on a span: ``httpx``
and ``requests`` from a string mapping, ``fastapi`` and ``agno`` from raw ASGI
``(name, value)`` byte pairs. They each used to carry a private copy of the
same frozenset, which makes the set a credential leak waiting on the next
divergence -- adding ``x-amz-security-token`` to one copy redacts it on one
transport and exports it on the other three.

The set lives here so there is exactly one place to add to. The two sanitizers
differ only in the shape they read from, not in policy.

A header is sensitive when its name is in the explicit set, or when one of its
segments is a credential word (``x-goog-api-key``, ``x-session-token``) or ends
in one (``apikey``, ``x-authtoken``). Names are compared with ``_`` folded to
``-``, so ``api_key`` and ``api-key`` are the same header -- which is also how
the upstream OTel instrumentations record them. Whole segments, not
substrings, so rate-limit headers such as ``x-ratelimit-remaining-tokens``
stay readable.
"""

import re
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from netra.config import get_active_config

REDACTED = "[REDACTED]"

SENSITIVE_HEADERS: FrozenSet[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "apikey",
        "x-apikey",
        "x-auth-token",
        "proxy-authorization",
        "x-goog-api-key",
        "x-amz-security-token",
        "ocp-apim-subscription-key",
        "x-csrf-token",
        "x-xsrf-token",
    }
)

# Any header containing one of these as a whole ``-``/``_``-separated segment is redacted.
SENSITIVE_HEADER_SEGMENTS: FrozenSet[str] = frozenset({"key", "token", "secret", "password", "signature", "credential"})

# A segment ending in one of these is a run-together credential name (``apikey``,
# ``authtoken``, ``clientsecret``) and is redacted too.
SENSITIVE_SEGMENT_SUFFIXES: Tuple[str, ...] = ("key", "token", "secret", "password")

_SEGMENT_SEPARATORS = re.compile(r"[-_]")

# Segment matches that carry no credential and are useful for debugging.
NON_SENSITIVE_HEADERS: FrozenSet[str] = frozenset({"idempotency-key", "sec-websocket-key"})


def normalize_header_name(name: str) -> str:
    """Return *name* lower-cased with ``_`` folded to ``-``, the form every lookup uses."""
    return name.lower().replace("_", "-")


def is_sensitive_name_segment(segment: str) -> bool:
    """Return whether one ``-``/``_``-separated piece of a name marks it as a credential."""
    return segment in SENSITIVE_HEADER_SEGMENTS or segment.endswith(SENSITIVE_SEGMENT_SUFFIXES)


def has_sensitive_segment(name: str) -> bool:
    """Return whether any ``-``/``_``-separated segment of a lower-cased *name* is a credential word."""
    return any(is_sensitive_name_segment(segment) for segment in _SEGMENT_SEPARATORS.split(name))


def get_sensitive_headers() -> FrozenSet[str]:
    """Return the built-in sensitive headers plus any configured extras.

    Returns:
        :data:`SENSITIVE_HEADERS` merged with the active config's
        ``redact_headers``, or :data:`SENSITIVE_HEADERS` alone when
        ``Netra.init()`` has not run or configured no extras.
    """

    extras = getattr(get_active_config(), "redact_headers", None)
    if not isinstance(extras, frozenset) or not extras:
        return SENSITIVE_HEADERS
    return SENSITIVE_HEADERS | frozenset(normalize_header_name(name) for name in extras)


def is_sensitive_header(name: str, sensitive: Optional[FrozenSet[str]] = None) -> bool:
    """Return whether a header's value must be redacted.

    Args:
        name: The header name, in any case.
        sensitive: The explicit name set to check, from :func:`get_sensitive_headers`.
            Looked up when omitted.

    Returns:
        True when the name (with ``_`` read as ``-``) is in the explicit set, or
        when a ``-``/``_``-separated segment is or ends in a credential word and
        the name is not in :data:`NON_SENSITIVE_HEADERS`.
    """
    normalized = normalize_header_name(name)
    if normalized in (get_sensitive_headers() if sensitive is None else sensitive):
        return True
    if normalized in NON_SENSITIVE_HEADERS:
        return False
    return has_sensitive_segment(normalized)


def sanitize_header_mapping(headers: Mapping[str, str]) -> Dict[str, str]:
    """Redact sensitive values in a string-keyed header mapping.

    Args:
        headers: A mapping of header names to values, as ``httpx.Headers`` and
            ``requests``' ``CaseInsensitiveDict`` both provide.

    Returns:
        A new dict with sensitive values replaced by :data:`REDACTED`. Header
        names are returned as the mapping yielded them.
    """
    sensitive = get_sensitive_headers()
    return {name: REDACTED if is_sensitive_header(name, sensitive) else value for name, value in headers.items()}


def sanitize_asgi_headers(raw_headers: Iterable[Tuple[bytes, bytes]]) -> Dict[str, str]:
    """Redact sensitive values in raw ASGI header pairs.

    Args:
        raw_headers: ``(name_bytes, value_bytes)`` tuples from an ASGI scope or
            response-start message.

    Returns:
        A dict mapping lower-cased header names to their values, with sensitive
        headers replaced by :data:`REDACTED`. Values are decoded as latin-1,
        which is the encoding the ASGI spec defines for header bytes.
    """
    sensitive = get_sensitive_headers()
    sanitized: Dict[str, str] = {}
    for name_bytes, value_bytes in raw_headers:
        name = name_bytes.decode("latin-1").lower()
        sanitized[name] = REDACTED if is_sensitive_header(name, sensitive) else value_bytes.decode("latin-1")
    return sanitized
