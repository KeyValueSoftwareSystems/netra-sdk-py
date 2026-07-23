"""
Unit tests for the truncation-limit configuration.

These limits (``attribute_max_len``, ``conversation_max_len``,
``trial_block_duration_seconds``) were previously class attributes resolved at
*import* time, which ignored any env change applied after ``netra`` was imported
(e.g. a late ``load_dotenv()``). They are now instance attributes resolved at
``Config`` construction (init) time, exposed to global/static consumers via the
active-config getters. These tests pin that behavior.
"""

import pytest

from netra import config as config_module
from netra.config import (
    _DEFAULT_ATTRIBUTE_MAX_LEN,
    _DEFAULT_CONVERSATION_CONTENT_MAX_LEN,
    _DEFAULT_TRIAL_BLOCK_DURATION_SECONDS,
    Config,
    get_attribute_max_len,
    get_conversation_max_len,
    get_trial_block_duration_seconds,
    set_active_config,
)
from netra.utils import serialize_value

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


class TestConfigLimitResolution:
    """Config resolves the limits from env at construction time."""

    def test_resolves_env_set_before_construction(self, monkeypatch):
        # Env applied *after* import but *before* Config() — the exact late-load case.
        monkeypatch.setenv("NETRA_ATTRIBUTE_MAX_LEN", "1234")
        monkeypatch.setenv("NETRA_CONVERSATION_CONTENT_MAX_LEN", "1500")
        monkeypatch.setenv("TRIAL_BLOCK_DURATION_SECONDS", "42")

        cfg = Config()

        assert cfg.attribute_max_len == 1234
        assert cfg.conversation_max_len == 1500
        assert cfg.trial_block_duration_seconds == 42

    def test_defaults_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("NETRA_ATTRIBUTE_MAX_LEN", raising=False)
        monkeypatch.delenv("NETRA_CONVERSATION_CONTENT_MAX_LEN", raising=False)
        monkeypatch.delenv("TRIAL_BLOCK_DURATION_SECONDS", raising=False)

        cfg = Config()

        assert cfg.attribute_max_len == _DEFAULT_ATTRIBUTE_MAX_LEN
        assert cfg.conversation_max_len == _DEFAULT_CONVERSATION_CONTENT_MAX_LEN
        assert cfg.trial_block_duration_seconds == _DEFAULT_TRIAL_BLOCK_DURATION_SECONDS

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NETRA_ATTRIBUTE_MAX_LEN", "not-an-int")

        cfg = Config()

        assert cfg.attribute_max_len == _DEFAULT_ATTRIBUTE_MAX_LEN


class TestActiveConfigGetters:
    """Getters fall back to defaults pre-init and reflect the active config after."""

    def test_getters_default_before_activation(self):
        assert config_module._active_config is None
        assert get_attribute_max_len() == _DEFAULT_ATTRIBUTE_MAX_LEN
        assert get_conversation_max_len() == _DEFAULT_CONVERSATION_CONTENT_MAX_LEN
        assert get_trial_block_duration_seconds() == _DEFAULT_TRIAL_BLOCK_DURATION_SECONDS

    def test_getters_reflect_active_config(self, monkeypatch):
        monkeypatch.setenv("NETRA_ATTRIBUTE_MAX_LEN", "10")
        monkeypatch.setenv("NETRA_CONVERSATION_CONTENT_MAX_LEN", "20")
        monkeypatch.setenv("TRIAL_BLOCK_DURATION_SECONDS", "30")

        set_active_config(Config())

        assert get_attribute_max_len() == 10
        assert get_conversation_max_len() == 20
        assert get_trial_block_duration_seconds() == 30

    def test_serialize_value_uses_active_limit(self, monkeypatch):
        monkeypatch.setenv("NETRA_ATTRIBUTE_MAX_LEN", "5")
        set_active_config(Config())

        assert serialize_value("x" * 100) == "x" * 5
