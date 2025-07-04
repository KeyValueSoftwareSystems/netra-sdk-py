"""
Shared test configuration and fixtures for Netra SDK tests.

This module provides common fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import logging
import os
import threading
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)  # type: ignore
def reset_netra_state() -> Generator[None, None, None]:
    """
    Automatically reset Netra state before and after each test.

    This fixture ensures that each test starts with a clean state
    and doesn't interfere with other tests.
    """
    # Import here to avoid circular imports
    from netra import Netra

    # Reset state before test
    with Netra._init_lock:
        Netra._initialized = False

    yield

    # Reset state after test
    with Netra._init_lock:
        Netra._initialized = False


@pytest.fixture  # type: ignore
def mock_config() -> Generator[MagicMock, None, None]:
    """Mock Config class for testing."""
    with patch("netra.Config") as mock:
        yield mock


@pytest.fixture  # type: ignore
def mock_tracer() -> Generator[MagicMock, None, None]:
    """Mock Tracer class for testing."""
    with patch("netra.Tracer") as mock:
        yield mock


@pytest.fixture  # type: ignore
def mock_init_instrumentations() -> Generator[MagicMock, None, None]:
    """Mock init_instrumentations function for testing."""
    with patch("netra.init_instrumentations") as mock:
        yield mock


@pytest.fixture  # type: ignore
def mock_session_manager() -> Generator[MagicMock, None, None]:
    """Mock SessionManager for testing."""
    with patch("netra.SessionManager") as mock:
        yield mock


@pytest.fixture  # type: ignore
def sample_config_params() -> Dict[str, Any]:
    """Sample configuration parameters for testing."""
    return {
        "app_name": "test-app",
        "headers": "key1=value1,key2=value2",
        "disable_batch": True,
        "trace_content": False,
        "resource_attributes": {"env": "test", "version": "1.0.0"},
        "environment": "testing",
    }


@pytest.fixture  # type: ignore
def sample_session_data() -> Dict[str, str]:
    """Sample session data for testing."""
    return {
        "session_id": "test-session-123",
        "user_id": "user-456",
        "tenant_id": "tenant-789",
    }


@pytest.fixture  # type: ignore
def sample_custom_attributes() -> Dict[str, Any]:
    """Sample custom attributes for testing."""
    return {
        "string_attr": "value",
        "int_attr": 123,
        "float_attr": 45.67,
        "bool_attr": True,
        "list_attr": ["a", "b", "c"],
        "dict_attr": {"nested": "value"},
    }


@pytest.fixture(name="sample_event_data")  # type: ignore
def sample_event_data() -> Dict[str, Any]:
    """Sample event data for testing."""
    return {
        "event_name": "user_action",
        "attributes": {
            "action": "click",
            "element": "button",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }


@pytest.fixture  # type: ignore
def clean_environment() -> Generator[None, None, None]:
    """
    Provide a clean environment for testing by temporarily clearing
    relevant environment variables.
    """
    # Environment variables that might affect tests
    env_vars_to_clear: List[str] = [
        "OTEL_SERVICE_NAME",
        "NETRA_APP_NAME",
        "NETRA_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "NETRA_API_KEY",
        "NETRA_HEADERS",
    ]

    # Store original values
    original_values: Dict[str, Optional[str]] = {}
    for var in env_vars_to_clear:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture(autouse=True)  # type: ignore
def thread_barrier() -> threading.Barrier:
    """
    Create a threading barrier for synchronizing multiple threads in tests.

    Returns:
        threading.Barrier: A barrier for coordinating thread execution
    """
    return threading.Barrier(2)  # Default to 2 threads, can be customized


class MockLogger:
    """Mock logger for testing logging behavior."""

    def __init__(self) -> None:
        self.info_calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.warning_calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.error_calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.debug_calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.info_calls.append((message, args, kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_calls.append((message, args, kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((message, args, kwargs))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_calls.append((message, args, kwargs))


@pytest.fixture(name="mock_logger")  # type: ignore
def mock_logger() -> MockLogger:
    """Mock logger instance for testing."""
    return MockLogger()


# Test utilities
def assert_called_with_subset(mock_call: Any, expected_subset: Dict[str, Any]) -> None:
    """
    Assert that a mock was called with arguments that include the expected subset.

    Args:
        mock_call: The mock call to check
        expected_subset: Dictionary of expected key-value pairs
    """
    if mock_call.kwargs:
        actual_kwargs = mock_call.kwargs
    else:
        # If called with positional args, convert to dict for comparison
        actual_kwargs = mock_call.args[0] if mock_call.args else {}

    for key, expected_value in expected_subset.items():
        assert key in actual_kwargs, f"Expected key '{key}' not found in call arguments"
        assert actual_kwargs[key] == expected_value, f"Expected {key}={expected_value}, got {key}={actual_kwargs[key]}"


def create_thread_safe_counter() -> Tuple[Callable[[], int], Callable[[], int]]:
    """Create a thread-safe counter for testing concurrent operations."""
    counter: Dict[str, int] = {"value": 0}
    lock = threading.Lock()

    def increment() -> int:
        with lock:
            counter["value"] += 1
            return counter["value"]

    def get_value() -> int:
        with lock:
            return counter["value"]

    return increment, get_value


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "thread_safety: mark test as thread safety test")


# Custom assertions for better error messages
def assert_initialization_state(expected_state: bool, message: str = "") -> None:
    """Assert the initialization state of Netra SDK."""
    from netra import Netra

    actual_state = Netra.is_initialized()
    assert actual_state == expected_state, (
        f"Expected initialization state to be {expected_state}, " f"but got {actual_state}. {message}"
    )


def assert_mock_called_once_with_subset(mock_obj: MagicMock, expected_subset: Dict[str, Any]) -> None:
    """Assert that a mock was called once with arguments containing the expected subset."""
    assert mock_obj.call_count == 1, f"Expected 1 call, got {mock_obj.call_count}"
    assert_called_with_subset(mock_obj.call_args, expected_subset)
