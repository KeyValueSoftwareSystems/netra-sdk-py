"""Secrets must not reach a span through the API key, header values or URLs.

* H-2: a ``NETRA_API_KEY`` or header value with stray whitespace made every
  export fail, and the upstream exporter logged the full key on each batch.
* H-5: ``x-goog-api-key``, ``x-amz-security-token`` and similar credential
  headers were exported verbatim because only seven names were redacted.
* H-6: only ``user:pass@`` was stripped from URLs, so ``?key=``, presigned
  signatures and OAuth codes were exported in ``http.url``.
"""

import warnings

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra import config as config_module
from netra.config import Config
from netra.instrumentation.http.headers import (
    REDACTED,
    is_sensitive_header,
    sanitize_asgi_headers,
    sanitize_header_mapping,
)
from netra.instrumentation.http.url import REDACTED_VALUE, redact_query_secrets, redact_url_secrets
from netra.instrumentation.libraries.agno.utils import build_scope_url
from netra.instrumentation.libraries.fastapi.utils import build_request_url
from netra.instrumentation.libraries.httpx.wrappers import send_wrapper
from netra.processors.instrumentation_span_processor import InstrumentationSpanProcessor

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_active_config():
    """Isolate the process-global active config from other tests."""
    original = config_module._active_config
    config_module._active_config = None
    try:
        yield
    finally:
        config_module._active_config = original


@pytest.fixture
def netra_endpoint(monkeypatch):
    monkeypatch.setenv("NETRA_OTLP_ENDPOINT", "https://api.getnetra.ai/telemetry")
    monkeypatch.delenv("NETRA_HEADERS", raising=False)


@pytest.fixture
def exporter():
    """A tracer provider with the instrumentation processor ahead of an in-memory exporter."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(memory))
    memory.tracer = provider.get_tracer("test")
    return memory


class TestApiKeyWhitespace:
    """H-2: the key and header values are stripped before they become headers."""

    def test_trailing_newline_is_stripped_from_the_api_key(self, monkeypatch, netra_endpoint):
        monkeypatch.setenv("NETRA_API_KEY", "FAKEKEY123\n")

        config = Config()

        assert config.api_key == "FAKEKEY123"
        assert config.headers == {"x-api-key": "FAKEKEY123"}

    def test_whitespace_only_api_key_counts_as_missing(self, monkeypatch, netra_endpoint):
        monkeypatch.setenv("NETRA_API_KEY", "  \n")

        assert Config().api_key is None

    def test_api_key_with_an_embedded_control_character_is_dropped(self, monkeypatch, netra_endpoint):
        monkeypatch.setenv("NETRA_API_KEY", "FAKE\r\nKEY")

        with pytest.warns(UserWarning, match="NETRA_API_KEY") as record:
            config = Config()

        assert config.api_key is None
        assert "FAKE" not in str(record[0].message)

    def test_values_in_a_headers_dict_are_stripped(self, monkeypatch, netra_endpoint):
        monkeypatch.delenv("NETRA_API_KEY", raising=False)

        config = Config(headers={"x-tenant": " acme\n"})

        assert config.headers == {"x-tenant": "acme"}

    def test_a_header_with_a_control_character_is_skipped_and_named_without_its_value(
        self, monkeypatch, netra_endpoint
    ):
        monkeypatch.delenv("NETRA_API_KEY", raising=False)

        with pytest.warns(UserWarning) as record:
            config = Config(headers={"x-tenant": "acme", "x-secret": "abc\r\ndef"})

        assert config.headers == {"x-tenant": "acme"}
        message = str(record[0].message)
        assert "x-secret" in message
        assert "abc" not in message

    def test_the_callers_headers_dict_is_not_mutated(self, monkeypatch, netra_endpoint):
        monkeypatch.setenv("NETRA_API_KEY", "KEY")
        headers = {"x-tenant": "acme"}

        Config(headers=headers)

        assert headers == {"x-tenant": "acme"}

    def test_clean_values_produce_no_warning(self, monkeypatch, netra_endpoint):
        monkeypatch.setenv("NETRA_API_KEY", "KEY")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = Config(headers={"x-tenant": "acme\tcorp"})

        assert config.headers["x-tenant"] == "acme\tcorp"


class TestCredentialHeaders:
    """H-5: explicit credential headers and whole-segment matches are redacted."""

    @pytest.mark.parametrize(
        "header",
        [
            "x-goog-api-key",
            "x-amz-security-token",
            "ocp-apim-subscription-key",
            "x-csrf-token",
            "x-xsrf-token",
            "x-session-token",
            "x-client-secret",
            "x-webhook-signature",
            "x-db-password",
            "x-service-credential",
            "X-Custom-Api-Key",
        ],
    )
    def test_credential_headers_are_redacted_on_both_shapes(self, header):
        assert sanitize_header_mapping({header: "secret"})[header] == REDACTED
        assert sanitize_asgi_headers([(header.encode(), b"secret")])[header.lower()] == REDACTED

    @pytest.mark.parametrize(
        "header",
        [
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-limit-tokens",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-input-tokens-limit",
            "idempotency-key",
            "sec-websocket-key",
            "keep-alive",
            "access-control-allow-credentials",
            "x-request-id",
            "content-type",
        ],
    )
    def test_debugging_headers_pass_through(self, header):
        assert sanitize_header_mapping({header: "value"})[header] == "value"
        assert not is_sensitive_header(header)

    def test_configured_extra_headers_are_still_redacted(self, monkeypatch):
        monkeypatch.setenv("NETRA_REDACT_HEADERS", "x-tenant-id")
        config_module.set_active_config(Config())

        assert sanitize_header_mapping({"x-tenant-id": "acme"}) == {"x-tenant-id": REDACTED}

    def test_upstream_header_attributes_are_redacted_by_segment(self, exporter):
        with exporter.tracer.start_as_current_span("GET") as span:
            span.set_attribute("http.request.header.x_goog_api_key", ("AIzaFAKE",))
            span.set_attribute("http.response.header.x_ratelimit_remaining_tokens", ("9000",))

        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes["http.request.header.x_goog_api_key"] == (REDACTED,)
        assert attributes["http.response.header.x_ratelimit_remaining_tokens"] == ("9000",)


class TestUrlRedaction:
    """H-6: secret query parameters are redacted; everything else is byte-identical."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (
                "https://generativelanguage.googleapis.com/v1beta/models/g:generateContent?key=AIzaFAKE",
                "https://generativelanguage.googleapis.com/v1beta/models/g:generateContent?key=REDACTED",
            ),
            (
                "https://b.s3.amazonaws.com/o.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256"
                "&X-Amz-Credential=AKIAFAKE%2F20260101%2Fus-east-1%2Fs3%2Faws4_request"
                "&X-Amz-Date=20260101T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
                "&X-Amz-Security-Token=FwoFAKE&X-Amz-Signature=abc123",
                "https://b.s3.amazonaws.com/o.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256"
                "&X-Amz-Credential=REDACTED"
                "&X-Amz-Date=20260101T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
                "&X-Amz-Security-Token=REDACTED&X-Amz-Signature=REDACTED",
            ),
            (
                "https://acct.blob.core.windows.net/c/b?sv=2022-11-02&se=2026-01-01&sp=r&sig=FAKE%2Bsig%3D",
                "https://acct.blob.core.windows.net/c/b?sv=2022-11-02&se=2026-01-01&sp=r&sig=REDACTED",
            ),
            (
                "https://app.example.com/oauth/callback?code=AUTHCODE&state=xyz",
                "https://app.example.com/oauth/callback?code=REDACTED&state=xyz",
            ),
            (
                "https://api.example.com/me?access_token=TOKEN&fields=id",
                "https://api.example.com/me?access_token=REDACTED&fields=id",
            ),
            (
                "https://app.example.com/cb#access_token=TOKEN&token_type=bearer",
                "https://app.example.com/cb#access_token=REDACTED&token_type=bearer",
            ),
            (
                "https://storage.googleapis.com/b/o?X-Goog-Credential=FAKE&X-Goog-Signature=FAKE&x=1",
                "https://storage.googleapis.com/b/o?X-Goog-Credential=REDACTED&X-Goog-Signature=REDACTED&x=1",
            ),
        ],
    )
    def test_secret_parameters_are_redacted(self, url, expected):
        assert redact_url_secrets(url) == expected

    @pytest.mark.parametrize("name", ["KEY", "Api_Key", "ACCESS_TOKEN", "Sig", "x-amz-signature", "api%5Fkey"])
    def test_parameter_names_match_case_insensitively_and_decoded(self, name):
        assert redact_url_secrets(f"https://h/p?{name}=s3cret") == f"https://h/p?{name}={REDACTED_VALUE}"

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.example.com/v1/items?page=2&q=a%20b+c&empty=&flag&x=1",
            "https://api.example.com/v1/items?sort=name;desc",
            "https://api.example.com/v1/items",
            "https://api.example.com/v1/items?",
            "/relative/path?page=1",
            "https://api.example.com/v1/items?key=&page=1",
            "",
        ],
    )
    def test_urls_with_nothing_to_redact_are_byte_identical(self, url):
        assert redact_url_secrets(url) == url

    def test_userinfo_credentials_are_still_stripped(self):
        assert redact_url_secrets("https://user:pass@h/p?key=k") == "https://REDACTED:REDACTED@h/p?key=REDACTED"

    def test_blank_values_and_order_are_preserved_around_redactions(self):
        assert redact_query_secrets("a=&key=k&b&c=3") == "a=&key=REDACTED&b&c=3"

    def test_fastapi_inbound_oauth_code_is_redacted(self):
        scope = {
            "scheme": "https",
            "server": ("app.example.com", 443),
            "path": "/oauth/callback",
            "query_string": b"code=AUTHCODE&state=xyz",
        }

        assert build_request_url(scope) == "https://app.example.com/oauth/callback?code=REDACTED&state=xyz"

    def test_agno_inbound_query_is_redacted(self):
        scope = {"scheme": "http", "server": ("h", 80), "path": "/run", "query_string": b"token=T&x=1"}

        assert build_scope_url(scope) == "http://h/run?token=REDACTED&x=1"

    def test_upstream_url_attributes_are_redacted_on_end(self, exporter):
        with exporter.tracer.start_as_current_span(
            "GET", attributes={"url.full": "https://h/p?key=AIzaFAKE&page=1"}
        ) as span:
            span.set_attribute("http.url", "https://h/p?sig=FAKE")
            span.set_attribute("http.target", "/cb?code=AUTHCODE")
            span.set_attribute("url.query", "access_token=T&x=1")

        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes["url.full"] == "https://h/p?key=REDACTED&page=1"
        assert attributes["http.url"] == "https://h/p?sig=REDACTED"
        assert attributes["http.target"] == "/cb?code=REDACTED"
        assert attributes["url.query"] == "access_token=REDACTED&x=1"

    def test_httpx_span_records_the_redacted_url_and_sends_the_real_one(self, exporter):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"ok": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        request = client.build_request("GET", "https://generativelanguage.googleapis.com/v1/m?key=AIzaFAKE")

        send_wrapper(exporter.tracer)(client.send, client, (request,), {})

        assert seen == ["https://generativelanguage.googleapis.com/v1/m?key=AIzaFAKE"]
        span = exporter.get_finished_spans()[0]
        assert span.attributes["http.url"] == "https://generativelanguage.googleapis.com/v1/m?key=REDACTED"
        assert "AIzaFAKE" not in str(dict(span.attributes))
