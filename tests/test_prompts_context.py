import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.processors.session_span_processor import SessionSpanProcessor
from netra.prompts.context import (
    ATTR_PROMPT_ID,
    ATTR_PROMPT_LABEL,
    ATTR_PROMPT_NAME,
    ATTR_PROMPT_VERSION,
    get_active_prompt_attributes,
    use_prompt,
)

PROMPT = {
    "messages": [{"role": "user", "content": "hi"}],
    "promptId": "p1",
    "promptName": "welcome",
    "versionNumber": 3,
    "label": "production",
}


class TestUsePrompt:
    def test_no_active_prompt_yields_no_attributes(self) -> None:
        assert get_active_prompt_attributes() == {}

    def test_active_prompt_yields_identity_attributes(self) -> None:
        with use_prompt(PROMPT):
            assert get_active_prompt_attributes() == {
                ATTR_PROMPT_ID: "p1",
                ATTR_PROMPT_NAME: "welcome",
                ATTR_PROMPT_VERSION: 3,
                ATTR_PROMPT_LABEL: "production",
            }
        assert get_active_prompt_attributes() == {}

    @pytest.mark.parametrize("prompt", [None, {}, {"promptId": "p1"}, {"versionNumber": 1}, "not-a-mapping"])
    def test_untrackable_prompt_links_nothing(self, prompt: object) -> None:
        with use_prompt(prompt):  # type: ignore[arg-type]
            assert get_active_prompt_attributes() == {}

    def test_version_zero_is_kept(self) -> None:
        with use_prompt({"promptId": "p1", "versionNumber": 0}):
            assert get_active_prompt_attributes()[ATTR_PROMPT_VERSION] == 0

    def test_nested_blocks_restore_outer_prompt(self) -> None:
        inner = {**PROMPT, "promptId": "p2"}
        with use_prompt(PROMPT):
            with use_prompt(inner):
                assert get_active_prompt_attributes()[ATTR_PROMPT_ID] == "p2"
            assert get_active_prompt_attributes()[ATTR_PROMPT_ID] == "p1"

    def test_prompt_reset_when_block_raises(self) -> None:
        with pytest.raises(RuntimeError):
            with use_prompt(PROMPT):
                raise RuntimeError("boom")
        assert get_active_prompt_attributes() == {}


class TestSpanTagging:
    def test_spans_inside_block_are_tagged_and_outside_are_not(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SessionSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with use_prompt(PROMPT):
            with tracer.start_as_current_span("inside"):
                pass
        with tracer.start_as_current_span("outside"):
            pass

        spans = {s.name: s.attributes for s in exporter.get_finished_spans()}
        assert spans["inside"][ATTR_PROMPT_ID] == "p1"
        assert spans["inside"][ATTR_PROMPT_VERSION] == 3
        assert ATTR_PROMPT_ID not in spans["outside"]
