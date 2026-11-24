import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.processors.session_span_processor import SessionSpanProcessor
from netra.prompts.context import (
    ATTR_PROMPT,
    get_active_prompt_attributes,
    use_prompt,
)


def _prompt(attrs: dict) -> dict:
    return json.loads(attrs[ATTR_PROMPT])


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
            attrs = get_active_prompt_attributes()
            assert list(attrs) == [ATTR_PROMPT]
            assert _prompt(attrs) == {"id": "p1", "name": "welcome", "version": 3, "label": "production"}
        assert get_active_prompt_attributes() == {}

    @pytest.mark.parametrize("prompt", [None, {}, {"promptId": "p1"}, {"versionNumber": 1}, "not-a-mapping"])
    def test_untrackable_prompt_links_nothing(self, prompt: object) -> None:
        with use_prompt(prompt):  # type: ignore[arg-type]
            assert get_active_prompt_attributes() == {}

    def test_version_zero_is_kept(self) -> None:
        with use_prompt({"promptId": "p1", "versionNumber": 0}):
            assert _prompt(get_active_prompt_attributes())["version"] == 0

    def test_nested_blocks_restore_outer_prompt(self) -> None:
        inner = {**PROMPT, "promptId": "p2"}
        with use_prompt(PROMPT):
            with use_prompt(inner):
                assert _prompt(get_active_prompt_attributes())["id"] == "p2"
            assert _prompt(get_active_prompt_attributes())["id"] == "p1"

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
        assert _prompt(spans["inside"]) == {"id": "p1", "name": "welcome", "version": 3, "label": "production"}
        assert ATTR_PROMPT not in spans["outside"]


def test_prompt_propagates_into_threadpool_with_threading_instrumentation() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from opentelemetry.instrumentation.threading import ThreadingInstrumentor

    ThreadingInstrumentor().instrument()
    try:
        with use_prompt(PROMPT), ThreadPoolExecutor(1) as pool:
            attrs = pool.submit(get_active_prompt_attributes).result()
        assert _prompt(attrs)["id"] == "p1"

        import threading

        out: dict = {}
        with use_prompt(PROMPT):
            t = threading.Thread(target=lambda: out.update(get_active_prompt_attributes()))
            t.start()
            t.join()
        assert _prompt(out)["id"] == "p1"
    finally:
        ThreadingInstrumentor().uninstrument()


class TestRecordPrompt:
    def test_collects_and_tags_current_span(self) -> None:
        from netra.prompts.context import collect_prompts, record_prompt

        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("t")
        with collect_prompts() as collected:
            with tracer.start_as_current_span("agent"):
                record_prompt(PROMPT)
            with use_prompt({**PROMPT, "promptId": "p2"}):
                pass
        assert _prompt(exporter.get_finished_spans()[0].attributes)["id"] == "p1"
        assert _prompt(collected[-1])["id"] == "p2"
