"""Redact credentials and secret query parameters from URLs before they are recorded.

``opentelemetry.util.http.remove_url_credentials`` only strips ``user:pass@``.
API keys, presigned signatures and OAuth codes travel in the query string
(``?key=AIza...``, ``X-Amz-Signature=...``, ``sig=...``, ``?code=...``) and were
exported verbatim in ``http.url`` / ``input.url``.

Upstream ``redact_url`` is not used: its key list is short and case-sensitive,
and it re-encodes the whole query through ``parse_qs``/``urlencode``, which
drops blank values and changes encoding. This module edits the query as a
string so a URL with nothing to redact comes back byte-identical.
"""

from typing import FrozenSet
from urllib.parse import unquote_plus

from opentelemetry.util.http import remove_url_credentials

from netra.instrumentation.http.headers import has_sensitive_segment

REDACTED_VALUE = "REDACTED"

# Compared case-insensitively against the percent-decoded parameter name.
SENSITIVE_QUERY_PARAMS: FrozenSet[str] = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "code",
        "client_secret",
        "sig",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "x-goog-credential",
        "password",
        "auth",
        "jwt",
        "client_assertion",
    }
)

# Segment matches that name a kind of credential rather than carry one (OAuth's
# ``token_type=bearer`` / ``token_type_hint=refresh_token``).
NON_SENSITIVE_QUERY_PARAMS: FrozenSet[str] = frozenset({"token_type", "token_type_hint"})


def is_sensitive_query_param(name: str) -> bool:
    """Return whether a query parameter's value must be redacted.

    Uses the same rule as headers: an exact name from
    :data:`SENSITIVE_QUERY_PARAMS`, or any ``-``/``_``-separated segment that is
    or ends in a credential word (``access_key``, ``subscription-key``,
    ``auth_token``, ``private_token``), unless the name is in
    :data:`NON_SENSITIVE_QUERY_PARAMS`.

    Args:
        name: The raw (still percent-encoded) parameter name.
    """
    decoded = unquote_plus(name).strip().lower()
    if decoded in SENSITIVE_QUERY_PARAMS:
        return True
    return decoded not in NON_SENSITIVE_QUERY_PARAMS and has_sensitive_segment(decoded)


def redact_query_secrets(params: str) -> str:
    """Redact sensitive values in an ``&``-separated ``name=value`` string.

    Args:
        params: A query string or fragment, without the leading ``?`` or ``#``.

    Returns:
        The same string with sensitive values replaced by :data:`REDACTED_VALUE`;
        every other byte, including order, encoding and blank values, is kept.
    """
    parts = params.split("&")
    changed = False
    for index, part in enumerate(parts):
        name, sep, value = part.partition("=")
        if not sep or not value:
            continue
        if is_sensitive_query_param(name):
            parts[index] = f"{name}={REDACTED_VALUE}"
            changed = True
    return "&".join(parts) if changed else params


def redact_url_secrets(url: str) -> str:
    """Strip userinfo credentials and redact secret query and fragment parameters.

    Args:
        url: The URL to sanitize.

    Returns:
        The sanitized URL. A URL with no credentials and no sensitive parameters
        is returned unchanged.
    """
    if not isinstance(url, str) or not url:
        return url

    url = remove_url_credentials(url)

    base, hash_sep, fragment = url.partition("#")
    path, query_sep, query = base.partition("?")
    if query:
        query = redact_query_secrets(query)
    if fragment:
        fragment = redact_query_secrets(fragment)
    return f"{path}{query_sep}{query}{hash_sep}{fragment}"
