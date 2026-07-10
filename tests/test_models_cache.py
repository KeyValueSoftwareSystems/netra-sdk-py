from unittest.mock import MagicMock, patch

import pytest

from netra import Netra
from netra.config import Config
from netra.models.api import MODEL_PRICING_CACHE_TTL_SECONDS, Models


@pytest.fixture
def models() -> Models:
    cfg = Config(cache_ttl_seconds=60)
    client = MagicMock()
    instance = Models(cfg)
    instance._client = client
    return instance


class TestModelsGetModelPricingCaching:
    def test_use_cache_omitted_calls_http_every_time(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        models.get_model_pricing()
        models.get_model_pricing()

        assert models._client.get_model_pricing.call_count == 2

    def test_use_cache_true_second_call_skips_http(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        first = models.get_model_pricing(use_cache=True)
        second = models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 1
        assert first == [{"name": "gpt-4o"}]
        assert second == [{"name": "gpt-4o"}]

    def test_use_cache_true_different_names_use_separate_entries(self, models: Models) -> None:
        models._client.get_model_pricing.side_effect = [
            {"data": [{"name": "gpt-4o"}]},
            {"data": [{"name": "claude-3"}]},
        ]

        gpt = models.get_model_pricing("gpt-4o", use_cache=True)
        claude = models.get_model_pricing("claude-3", use_cache=True)

        assert models._client.get_model_pricing.call_count == 2
        assert gpt == [{"name": "gpt-4o"}]
        assert claude == [{"name": "claude-3"}]

    def test_name_none_uses_all_cache_key(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        models.get_model_pricing(use_cache=True)
        models.get_model_pricing(name=None, use_cache=True)

        assert models._client.get_model_pricing.call_count == 1

    def test_empty_list_is_cached(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": []}

        first = models.get_model_pricing(use_cache=True)
        second = models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 1
        assert first == []
        assert second == []

    def test_api_failure_empty_dict_does_not_store_in_cache(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {}

        models.get_model_pricing(use_cache=True)
        models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 2

    def test_api_none_response_does_not_store_in_cache(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = None

        models.get_model_pricing(use_cache=True)
        models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 2

    def test_non_list_data_does_not_store_in_cache(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": {"name": "bad"}}

        models.get_model_pricing(use_cache=True)
        models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 2

    def test_per_call_cache_ttl_expires_before_default(self, models: Models) -> None:
        with patch("netra.cache.time.monotonic", side_effect=[0.0, 0.0, 1.1, 1.1]):
            models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

            models.get_model_pricing(use_cache=True, cache_ttl=1)
            assert models._client.get_model_pricing.call_count == 1

            models.get_model_pricing(use_cache=True, cache_ttl=1)
            assert models._client.get_model_pricing.call_count == 1

            models.get_model_pricing(use_cache=True, cache_ttl=1)
            assert models._client.get_model_pricing.call_count == 2

    def test_zero_cache_ttl_skips_cache_write(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        models.get_model_pricing(use_cache=True, cache_ttl=0)
        models.get_model_pricing(use_cache=True, cache_ttl=0)

        assert models._client.get_model_pricing.call_count == 2

    def test_use_cache_false_with_cache_ttl_ignores_cache(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        models.get_model_pricing(use_cache=False, cache_ttl=30)
        models.get_model_pricing(use_cache=False, cache_ttl=30)

        assert models._client.get_model_pricing.call_count == 2

    def test_clear_cache_forces_next_call_to_hit_http(self, models: Models) -> None:
        models._client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}

        models.get_model_pricing(use_cache=True)
        models.clear_cache()
        models.get_model_pricing(use_cache=True)

        assert models._client.get_model_pricing.call_count == 2

    def test_default_ttl_is_models_owned_constant(self, models: Models) -> None:
        assert MODEL_PRICING_CACHE_TTL_SECONDS == 300
        assert models._cache._default_ttl == MODEL_PRICING_CACHE_TTL_SECONDS
        assert models._cache._default_ttl != 60


class TestModelsCacheShutdown:
    def setup_method(self) -> None:
        with Netra._init_lock:
            Netra._initialized = False

    def teardown_method(self) -> None:
        with Netra._init_lock:
            Netra._initialized = False

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.Config")
    def test_shutdown_clears_models_cache(
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
        mock_client.get_model_pricing.return_value = {"data": [{"name": "gpt-4o"}]}
        Netra.models._client = mock_client

        Netra.models.get_model_pricing(use_cache=True)
        Netra.models.get_model_pricing(use_cache=True)
        assert mock_client.get_model_pricing.call_count == 1

        Netra.shutdown()

        Netra.models.get_model_pricing(use_cache=True)
        assert mock_client.get_model_pricing.call_count == 2
