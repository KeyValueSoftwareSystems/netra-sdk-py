from unittest.mock import MagicMock

import pytest

from netra.config import Config
from netra.prompts.api import Prompts


@pytest.fixture
def prompts() -> Prompts:
    cfg = Config(cache_ttl_seconds=60)
    client = MagicMock()
    instance = Prompts(cfg)
    instance._client = client
    return instance


class TestPromptsGetPromptCaching:
    def test_use_cache_omitted_calls_http_every_time(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {"template": "v1"}

        prompts.get_prompt("my-prompt")
        prompts.get_prompt("my-prompt")

        assert prompts._client.get_prompt_version.call_count == 2

    def test_use_cache_true_second_call_skips_http(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {"template": "v1"}

        first = prompts.get_prompt("my-prompt", use_cache=True)
        second = prompts.get_prompt("my-prompt", use_cache=True)

        assert prompts._client.get_prompt_version.call_count == 1
        assert first == {"template": "v1"}
        assert second == {"template": "v1"}

    def test_use_cache_true_different_labels_use_separate_entries(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.side_effect = [
            {"template": "prod"},
            {"template": "staging"},
        ]

        prod = prompts.get_prompt("my-prompt", label="production", use_cache=True)
        staging = prompts.get_prompt("my-prompt", label="staging", use_cache=True)

        assert prompts._client.get_prompt_version.call_count == 2
        assert prod == {"template": "prod"}
        assert staging == {"template": "staging"}

    def test_api_failure_does_not_store_in_cache(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {}

        prompts.get_prompt("my-prompt", use_cache=True)
        prompts.get_prompt("my-prompt", use_cache=True)

        assert prompts._client.get_prompt_version.call_count == 2

    def test_use_cache_false_with_cache_ttl_ignores_cache(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {"template": "v1"}

        prompts.get_prompt("my-prompt", use_cache=False, cache_ttl=30)
        prompts.get_prompt("my-prompt", use_cache=False, cache_ttl=30)

        assert prompts._client.get_prompt_version.call_count == 2

    def test_clear_cache_forces_next_call_to_hit_http(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {"template": "v1"}

        prompts.get_prompt("my-prompt", use_cache=True)
        prompts.clear_cache()
        prompts.get_prompt("my-prompt", use_cache=True)

        assert prompts._client.get_prompt_version.call_count == 2
