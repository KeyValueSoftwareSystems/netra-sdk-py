from unittest.mock import MagicMock, patch

import pytest

from netra import Netra
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

    def test_api_none_response_does_not_store_in_cache(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = None

        prompts.get_prompt("my-prompt", use_cache=True)
        prompts.get_prompt("my-prompt", use_cache=True)

        assert prompts._client.get_prompt_version.call_count == 2

    def test_per_call_cache_ttl_expires_before_default(self, prompts: Prompts) -> None:
        with patch("netra.cache.time.monotonic", side_effect=[0.0, 0.0, 1.1, 1.1]):
            prompts._client.get_prompt_version.return_value = {"template": "v1"}

            prompts.get_prompt("my-prompt", use_cache=True, cache_ttl=1)
            assert prompts._client.get_prompt_version.call_count == 1

            prompts.get_prompt("my-prompt", use_cache=True, cache_ttl=1)
            assert prompts._client.get_prompt_version.call_count == 1

            prompts.get_prompt("my-prompt", use_cache=True, cache_ttl=1)
            assert prompts._client.get_prompt_version.call_count == 2

    def test_zero_cache_ttl_skips_cache_write(self, prompts: Prompts) -> None:
        prompts._client.get_prompt_version.return_value = {"template": "v1"}

        prompts.get_prompt("my-prompt", use_cache=True, cache_ttl=0)
        prompts.get_prompt("my-prompt", use_cache=True, cache_ttl=0)

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


class TestPromptsCacheShutdown:
    def setup_method(self) -> None:
        with Netra._init_lock:
            Netra._initialized = False

    def teardown_method(self) -> None:
        with Netra._init_lock:
            Netra._initialized = False

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.Config")
    def test_shutdown_clears_prompt_cache(
        self,
        mock_config: MagicMock,
        mock_tracer: MagicMock,
        mock_init_instrumentations: MagicMock,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.cache_ttl_seconds = 60
        mock_config.return_value = mock_cfg

        Netra.init()

        mock_client = MagicMock()
        mock_client.get_prompt_version.return_value = {"template": "v1"}
        Netra.prompts._client = mock_client

        Netra.prompts.get_prompt("my-prompt", use_cache=True)
        Netra.prompts.get_prompt("my-prompt", use_cache=True)
        assert mock_client.get_prompt_version.call_count == 1

        Netra.shutdown()

        Netra.prompts.get_prompt("my-prompt", use_cache=True)
        assert mock_client.get_prompt_version.call_count == 2
