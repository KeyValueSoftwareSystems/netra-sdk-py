"""The one list of HTTP headers Netra must never record, and how to apply it.

Four instrumentations record request or response headers on a span: ``httpx``
and ``requests`` from a string mapping, ``fastapi`` and ``agno`` from raw ASGI
``(name, value)`` byte pairs. They each used to carry a private copy of the
same frozenset, which makes the set a credential leak waiting on the next
divergence -- adding ``x-amz-security-token`` to one copy redacts it on one
transport and exports it on the other three.

The set lives here so there is exactly one place to add to. The two sanitizers
differ only in the shape they read from, not in policy.
"""

from typing import Dict, FrozenSet, Iterable, Mapping, Tuple

from netra.config import get_active_config

REDACTED = "[REDACTED]"

SENSITIVE_HEADERS: FrozenSet[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)


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
    return SENSITIVE_HEADERS | extras


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
    return {name: REDACTED if name.lower() in sensitive else value for name, value in headers.items()}


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
        sanitized[name] = REDACTED if name in sensitive else value_bytes.decode("latin-1")
    return sanitized
