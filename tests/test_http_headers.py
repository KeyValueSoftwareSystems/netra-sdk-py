"""Every HTTP instrumentation redacts the same headers.

``httpx``, ``requests``, ``fastapi`` and ``agno`` each used to carry a private
copy of the sensitive-header frozenset. Four copies of a redaction policy is a
credential leak waiting on the next divergence: adding a header to one copy
redacts it on one transport and exports it on the other three.

These tests pin the single source of truth and the fact that all four reach it.
"""

import pytest

from netra.instrumentation.http.headers import (
    REDACTED,
    SENSITIVE_HEADERS,
    sanitize_asgi_headers,
    sanitize_header_mapping,
)

pytestmark = pytest.mark.unit


class TestOneRedactionPolicy:
    """The four HTTP instrumentations share one frozenset, not four copies."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "netra.instrumentation.libraries.httpx.utils",
            "netra.instrumentation.libraries.requests.utils",
            "netra.instrumentation.libraries.fastapi.utils",
            "netra.instrumentation.libraries.agno.utils",
        ],
    )
    def test_no_instrumentation_defines_its_own_header_set(self, module_path):
        module = __import__(module_path, fromlist=["_"])

        private_copy = getattr(module, "_SENSITIVE_HEADERS", None)

        assert private_copy is None or private_copy is SENSITIVE_HEADERS

    @pytest.mark.parametrize("header", sorted(SENSITIVE_HEADERS))
    def test_both_sanitizers_redact_every_declared_header(self, header):
        assert sanitize_header_mapping({header: "secret"})[header] == REDACTED
        assert sanitize_asgi_headers([(header.encode(), b"secret")])[header] == REDACTED

    def test_a_credential_header_is_redacted_regardless_of_casing(self):
        assert sanitize_header_mapping({"Authorization": "Bearer t"})["Authorization"] == REDACTED
        assert sanitize_asgi_headers([(b"AUTHORIZATION", b"Bearer t")])["authorization"] == REDACTED

    def test_ordinary_headers_pass_through_untouched(self):
        mapping = {"content-type": "application/json", "accept": "*/*"}

        assert sanitize_header_mapping(mapping) == mapping
        assert sanitize_asgi_headers([(b"content-type", b"application/json")]) == {"content-type": "application/json"}

    def test_asgi_names_are_lower_cased_so_the_set_lookup_cannot_miss(self):
        assert sanitize_asgi_headers([(b"X-Api-Key", b"k")]) == {"x-api-key": REDACTED}

    def test_no_headers_yields_no_entries(self):
        assert sanitize_header_mapping({}) == {}
        assert sanitize_asgi_headers([]) == {}
