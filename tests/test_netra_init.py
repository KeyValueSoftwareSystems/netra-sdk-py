"""
Unit tests for netra/__init__.py module.

This module contains comprehensive tests for the main Netra SDK class,
covering initialization, session management, and thread safety.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from netra import Netra


class TestNetraInitialization:
    """Test cases for Netra SDK initialization."""

    def setup_method(self) -> None:
        """Reset Netra state before each test."""
        # Reset the initialized state and clear any locks
        with Netra._init_lock:
            Netra._initialized = False

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset the initialized state
        with Netra._init_lock:
            Netra._initialized = False

    def test_is_initialized_returns_false_initially(self) -> None:
        """Test that is_initialized returns False before initialization."""
        assert Netra.is_initialized() is False

    def test_is_initialized_returns_true_after_init(self) -> None:
        """Test that is_initialized returns True after initialization."""
        with patch("netra.Tracer"), patch("netra.init_instrumentations"):
            Netra.init()
            assert Netra.is_initialized() is True

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.Config")
    def test_init_with_default_parameters(
        self, mock_config: MagicMock, mock_tracer: MagicMock, mock_init_instrumentations: MagicMock
    ) -> None:
        """Test initialization with default parameters."""
        Netra.init()

        # Verify Config was created with default values
        mock_config.assert_called_once_with(
            app_name=None,
            headers=None,
            disable_batch=None,
            trace_content=None,
            resource_attributes=None,
            environment=None,
            enable_root_span=None,
            debug_mode=None,
        )

        # Verify Tracer was initialized
        mock_tracer.assert_called_once()

        # Verify instrumentations were initialized
        mock_init_instrumentations.assert_called_once_with(
            should_enrich_metrics=True,
            base64_image_uploader=None,
            instruments=None,
            block_instruments=None,
        )

        assert Netra.is_initialized() is True

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.Config")
    def test_init_with_custom_parameters(
        self, mock_config: MagicMock, mock_tracer: MagicMock, mock_init_instrumentations: MagicMock
    ) -> None:
        """Test initialization with custom parameters."""
        custom_params = {
            "app_name": "test-app",
            "headers": "key1=value1,key2=value2",
            "disable_batch": True,
            "trace_content": False,
            "resource_attributes": {"env": "test", "version": "1.0.0"},
            "environment": "testing",
            "enable_root_span": False,
            "debug_mode": True,
        }

        app_name = "test-app"
        headers = "key1=value1,key2=value2"
        disable_batch = True
        trace_content = False
        resource_attributes = {"env": "test", "version": "1.0.0"}
        environment = "testing"

        Netra.init(
            app_name=app_name,
            headers=headers,
            disable_batch=disable_batch,
            trace_content=trace_content,
            resource_attributes=resource_attributes,
            environment=environment,
            enable_root_span=False,
            debug_mode=True,
        )

        # Verify Config was created with custom values
        mock_config.assert_called_once_with(**custom_params)

        # Verify other components were initialized
        mock_tracer.assert_called_once()
        mock_init_instrumentations.assert_called_once()

        assert Netra.is_initialized() is True

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.logger")
    def test_init_called_multiple_times_logs_warning(
        self, mock_logger: MagicMock, mock_tracer: MagicMock, mock_init_instrumentations: MagicMock
    ) -> None:
        """Test that calling init multiple times logs a warning and doesn't reinitialize."""
        # First initialization
        Netra.init()

        # Reset mocks to check second call behavior
        mock_tracer.reset_mock()
        mock_init_instrumentations.reset_mock()

        # Second initialization should log warning and not reinitialize
        Netra.init()

        mock_logger.warning.assert_called_once_with("Netra.init() called more than once; ignoring subsequent calls.")

        # Tracer and instrumentations should not be called again
        mock_tracer.assert_not_called()
        mock_init_instrumentations.assert_not_called()

    def test_init_thread_safety(self) -> None:
        """Test that initialization is thread-safe."""
        init_count = 0
        exceptions = []

        def init_worker() -> None:
            """Worker function for thread safety test."""
            nonlocal init_count
            try:
                with patch("netra.Tracer"), patch("netra.init_instrumentations"):
                    Netra.init()
                    init_count += 1
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads that try to initialize simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=init_worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no exceptions occurred
        assert len(exceptions) == 0

        # Verify initialization happened only once
        assert Netra.is_initialized() is True

    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    @patch("netra.logger")
    def test_init_logs_success_message(
        self, mock_logger: MagicMock, mock_tracer: MagicMock, mock_init_instrumentations: MagicMock
    ) -> None:
        """Test that successful initialization logs success message."""
        Netra.init()

        mock_logger.info.assert_called_once_with("Netra successfully initialized.")


class TestNetraSessionManagement:
    """Test cases for Netra session management methods."""

    def setup_method(self) -> None:
        """Reset Netra state before each test."""
        with Netra._init_lock:
            Netra._initialized = False

    @patch("netra.SessionManager.set_session_context")
    def test_set_session_id(self, mock_set_session_context: MagicMock) -> None:
        """Test setting session ID."""
        session_id = "test-session-123"
        Netra.set_session_id(session_id)

        mock_set_session_context.assert_called_once_with("session_id", session_id)

    @patch("netra.SessionManager.set_session_context")
    def test_set_user_id(self, mock_set_session_context: MagicMock) -> None:
        """Test setting user ID."""
        user_id = "user-456"
        Netra.set_user_id(user_id)

        mock_set_session_context.assert_called_once_with("user_id", user_id)

    @patch("netra.SessionManager.set_session_context")
    def test_set_tenant_id(self, mock_set_session_context: MagicMock) -> None:
        """Test setting tenant ID."""
        tenant_id = "tenant-789"
        Netra.set_tenant_id(tenant_id)

        mock_set_session_context.assert_called_once_with("tenant_id", tenant_id)

    @patch("netra.SessionManager.set_session_context")
    def test_set_custom_attributes(self, mock_set_session_context: MagicMock) -> None:
        """Test setting custom attributes."""
        key = "custom_key"
        value = "custom_value"
        Netra.set_custom_attributes(key, value)

        mock_set_session_context.assert_called_once_with("custom_attributes", {key: value})

    @patch("netra.SessionManager.set_session_context")
    def test_set_custom_attributes_with_various_types(self, mock_set_session_context: MagicMock) -> None:
        """Test setting custom attributes with different value types."""
        test_cases = [
            ("string_key", "string_value"),
            ("int_key", 42),
            ("float_key", 3.14),
            ("bool_key", True),
            ("list_key", [1, 2, 3]),
            ("dict_key", {"nested": "value"}),
        ]

        for key, value in test_cases:
            Netra.set_custom_attributes(key, value)

        # Verify all calls were made with correct parameters
        expected_calls = [("custom_attributes", {key: value}) for key, value in test_cases]

        actual_calls = [call.args for call in mock_set_session_context.call_args_list]
        assert actual_calls == expected_calls

    @patch("netra.SessionManager.set_custom_event")
    def test_set_custom_event(self, mock_set_custom_event: MagicMock) -> None:
        """Test setting custom event."""
        event_name = "user_action"
        attributes = {"action": "click", "element": "button"}
        Netra.set_custom_event(event_name, attributes)

        mock_set_custom_event.assert_called_once_with(event_name, attributes)

    @patch("netra.SessionManager.set_custom_event")
    def test_set_custom_event_with_various_attribute_types(self, mock_set_custom_event: MagicMock) -> None:
        """Test setting custom event with different attribute types."""
        event_name = "complex_event"
        attributes = {
            "string_attr": "value",
            "int_attr": 123,
            "float_attr": 45.67,
            "bool_attr": False,
            "list_attr": ["a", "b", "c"],
            "dict_attr": {"nested": {"deep": "value"}},
        }
        Netra.set_custom_event(event_name, attributes)

        mock_set_custom_event.assert_called_once_with(event_name, attributes)


class TestNetraClassProperties:
    """Test cases for Netra class properties and constants."""

    def test_class_has_correct_attributes(self) -> None:
        """Test that Netra class has expected attributes."""
        assert hasattr(Netra, "_initialized")
        assert hasattr(Netra, "_init_lock")
        assert isinstance(Netra._init_lock, type(threading.RLock()))

    def test_initial_state(self) -> None:
        """Test initial state of Netra class."""
        # Reset state first
        with Netra._init_lock:
            Netra._initialized = False

        assert Netra._initialized is False
        assert isinstance(Netra._init_lock, type(threading.RLock()))

    def test_lock_is_reentrant(self) -> None:
        """Test that the initialization lock is reentrant."""
        # This should not deadlock
        with Netra._init_lock:
            with Netra._init_lock:
                assert True  # If we reach here, the lock is reentrant


class TestNetraIntegration:
    """Integration tests for Netra SDK."""

    def setup_method(self) -> None:
        """Reset Netra state before each test."""
        with Netra._init_lock:
            Netra._initialized = False

    def teardown_method(self) -> None:
        """Clean up after each test."""
        with Netra._init_lock:
            Netra._initialized = False

    @patch("netra.SessionManager.set_session_context")
    @patch("netra.SessionManager.set_custom_event")
    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    def test_full_workflow(
        self,
        mock_tracer: MagicMock,
        mock_init_instrumentations: MagicMock,
        mock_set_custom_event: MagicMock,
        mock_set_session_context: MagicMock,
    ) -> None:
        """Test a complete workflow of initialization and session management."""
        # Initialize SDK
        Netra.init(app_name="test-app", environment="testing")
        assert Netra.is_initialized() is True

        # Set session context
        Netra.set_session_id("session-123")
        Netra.set_user_id("user-456")
        Netra.set_tenant_id("tenant-789")
        Netra.set_custom_attributes("feature_flag", "enabled")

        # Set custom event
        Netra.set_custom_event("user_login", {"timestamp": "2024-01-01T00:00:00Z"})

        # Verify all components were called correctly
        mock_tracer.assert_called_once()
        mock_init_instrumentations.assert_called_once()

        # Verify session management calls
        expected_session_calls = [
            ("session_id", "session-123"),
            ("user_id", "user-456"),
            ("tenant_id", "tenant-789"),
            ("custom_attributes", {"feature_flag": "enabled"}),
        ]
        actual_session_calls = [call.args for call in mock_set_session_context.call_args_list]
        assert actual_session_calls == expected_session_calls

        mock_set_custom_event.assert_called_once_with("user_login", {"timestamp": "2024-01-01T00:00:00Z"})

    def test_session_methods_work_without_initialization(self) -> None:
        """Test that session management methods work even without initialization."""
        # This tests that session methods don't depend on initialization state
        with patch("netra.SessionManager.set_session_context") as mock_set_context:
            with patch("netra.SessionManager.set_custom_event") as mock_set_event:
                Netra.set_session_id("test-session")
                Netra.set_custom_event("test-event", {"key": "value"})

                mock_set_context.assert_called_once_with("session_id", "test-session")
                mock_set_event.assert_called_once_with("test-event", {"key": "value"})


class TestNetraErrorHandling:
    """Test cases for error handling in Netra SDK."""

    def setup_method(self) -> None:
        """Reset Netra state before each test."""
        with Netra._init_lock:
            Netra._initialized = False

    @patch("netra.Tracer", side_effect=Exception("Tracer initialization failed"))
    @patch("netra.Config")
    def test_init_handles_tracer_exception(self, mock_config: MagicMock, mock_tracer: MagicMock) -> None:
        """Test that initialization handles Tracer exceptions properly."""
        with pytest.raises(Exception, match="Tracer initialization failed"):
            Netra.init()

        # Verify that initialization state remains False on failure
        assert Netra.is_initialized() is False

    @patch("netra.init_instrumentations", side_effect=Exception("Instrumentation failed"))
    @patch("netra.Tracer")
    @patch("netra.Config")
    def test_init_handles_instrumentation_exception(
        self, mock_config: MagicMock, mock_tracer: MagicMock, mock_init_instrumentations: MagicMock
    ) -> None:
        """Test that initialization handles instrumentation exceptions properly."""
        with pytest.raises(Exception, match="Instrumentation failed"):
            Netra.init()

        # Verify that initialization state remains False on failure
        assert Netra.is_initialized() is False

    @patch("netra.SessionManager.set_session_context", side_effect=Exception("Session context failed"))
    def test_session_methods_propagate_exceptions(self, mock_set_session_context: MagicMock) -> None:
        """Test that session management methods propagate exceptions."""
        with pytest.raises(Exception, match="Session context failed"):
            Netra.set_session_id("test-session")

    @patch("netra.SessionManager.set_custom_event", side_effect=Exception("Custom event failed"))
    def test_custom_event_propagates_exceptions(self, mock_set_custom_event: MagicMock) -> None:
        """Test that custom event method propagates exceptions."""
        with pytest.raises(Exception, match="Custom event failed"):
            Netra.set_custom_event("test-event", {"key": "value"})


if __name__ == "__main__":
    pytest.main([__file__])
