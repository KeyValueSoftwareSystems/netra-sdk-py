"""Tests for the structured ``input``/``output`` message builders."""

import json

import pytest

from netra.instrumentation import message_builder
from netra.instrumentation.capture.bounded_capture import TRUNCATION_ELLIPSIS, TRUNCATION_MARKER_KEY
from netra.instrumentation.message_builder import (
    build_chat_input,
    build_completion_output,
    build_messages,
    build_response_api_input,
)


@pytest.fixture
def attribute_limit(monkeypatch: pytest.MonkeyPatch):
    """Set the attribute length limit the builder fits its output to."""

    def set_limit(limit: int) -> None:
        monkeypatch.setattr(message_builder, "get_attribute_max_len", lambda: limit)

    return set_limit


@pytest.fixture
def content_limit(monkeypatch: pytest.MonkeyPatch):
    """Set the per-message content length limit."""

    def set_limit(limit: int) -> None:
        monkeypatch.setattr(message_builder, "get_conversation_max_len", lambda: limit)

    return set_limit


class TestBuildMessages:
    def test_small_payload_is_serialised_unchanged(self) -> None:
        entries = [{"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [{"id": "1"}]}]

        assert build_messages(entries) == json.dumps(entries)

    def test_oversized_content_is_capped_and_marked(self, content_limit) -> None:
        content_limit(10)

        entries = json.loads(build_messages([{"role": "user", "content": "x" * 50}, {"role": "user", "content": "ok"}]))

        assert entries[0]["content"] == "x" * (10 - len(TRUNCATION_ELLIPSIS)) + TRUNCATION_ELLIPSIS
        assert entries[0][TRUNCATION_MARKER_KEY] is True
        assert entries[1] == {"role": "user", "content": "ok"}

    def test_oldest_messages_are_dropped_to_fit_the_attribute_limit(self, attribute_limit) -> None:
        attribute_limit(200)
        messages = [{"role": "user", "content": f"message {index} " + "x" * 40} for index in range(10)]

        serialised = build_messages(messages)
        entries = json.loads(serialised)

        assert len(serialised) <= 200
        assert 1 < len(entries) < len(messages)
        # Every kept entry but the oldest is an untouched suffix of the conversation.
        first_untouched = len(messages) - len(entries) + 1
        assert entries[1:] == messages[first_untouched:]

    def test_a_single_oversized_message_is_shortened_to_fit(self, attribute_limit) -> None:
        attribute_limit(100)

        serialised = build_messages([{"role": "user", "content": 'é\n"' * 100}])
        entries = json.loads(serialised)

        assert len(serialised) <= 100
        assert entries[0][TRUNCATION_MARKER_KEY] is True
        assert entries[0]["content"].endswith(TRUNCATION_ELLIPSIS)

    @pytest.mark.parametrize("limit", [2, 30, 75, 500, 2000])
    def test_output_is_always_valid_json_within_the_limit(self, attribute_limit, limit: int) -> None:
        attribute_limit(limit)
        entries = [{"role": "user", "content": '😀 \\ " é ' * 40}, {"role": "tool", "tool_calls": ["a"]}, "raw"] * 4

        serialised = build_messages(entries)

        assert len(serialised) <= limit
        assert isinstance(json.loads(serialised), list)


class TestBuilders:
    def test_empty_inputs_return_none(self) -> None:
        assert build_chat_input([]) is None
        assert build_chat_input([{"role": "user", "content": None}]) is None
        assert build_response_api_input({}) is None
        assert build_completion_output({"data": [{"embedding": [0.1]}]}) is None
        assert build_completion_output(None) is None

    def test_legacy_text_completion_is_captured(self) -> None:
        output = build_completion_output({"choices": [{"text": "hello", "finish_reason": "stop"}]})

        assert json.loads(output) == [{"role": "assistant", "content": "hello", "finish_reason": "stop"}]

    def test_finish_reason_is_recorded_on_the_last_entry_of_each_choice(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                },
                {"message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"},
            ]
        }

        entries = json.loads(build_completion_output(response))

        assert [entry.get("finish_reason") for entry in entries] == [None, "tool_calls", "stop"]
        assert entries[1]["tool_call_id"] == "call_1"

    def test_responses_api_input_skips_reasoning_items(self) -> None:
        kwargs = {
            "instructions": "be brief",
            "input": [
                {"type": "reasoning", "summary": []},
                {"type": "function_call_output", "output": "42"},
                {"role": "user", "content": "and?"},
            ],
        }

        assert json.loads(build_response_api_input(kwargs)) == [
            {"role": "system", "content": "be brief"},
            {"role": "tool", "content": "42"},
            {"role": "user", "content": "and?"},
        ]
