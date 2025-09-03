"""
Unit tests for Netra decorators module.
Tests the functionality of workflow, agent, and task decorators for both functions and classes.
"""

import asyncio
import inspect
import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from netra.config import Config
from netra.decorators import (
    _add_output_attributes,
    _add_span_attributes,
    _create_function_wrapper,
    _serialize_value,
    _wrap_class_methods,
    agent,
    task,
    workflow,
)


class TestSerializeValue:
    """Test value serialization functionality."""

    def test_serialize_primitive_types(self):
        """Test serialization of primitive types."""
        assert _serialize_value("string") == "string"
        assert _serialize_value(42) == "42"
        assert _serialize_value(3.14) == "3.14"
        assert _serialize_value(True) == "True"
        assert _serialize_value(None) == "None"

    def test_serialize_collections(self):
        """Test serialization of collections."""
        # List
        result = _serialize_value([1, 2, 3])
        assert result == "[1, 2, 3]"

        # Dict
        result = _serialize_value({"key": "value"})
        assert result == '{"key": "value"}'

        # Tuple
        result = _serialize_value((1, 2, 3))
        assert result == "[1, 2, 3]"

    def test_serialize_large_collections(self):
        """Test serialization with size limit."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = _serialize_value(large_dict)
        assert len(result) <= 1000

    def test_serialize_custom_object(self):
        """Test serialization of custom objects."""

        class CustomObject:
            def __str__(self):
                return "custom_object"

        obj = CustomObject()
        result = _serialize_value(obj)
        assert result == "custom_object"

    def test_serialize_exception_handling(self):
        """Test serialization with exception handling."""

        class ProblematicObject:
            def __str__(self):
                raise ValueError("Cannot serialize")

        obj = ProblematicObject()
        result = _serialize_value(obj)
        assert result == "ProblematicObject"


class TestAddSpanAttributes:
    """Test span attribute addition functionality."""

    def test_add_span_attributes_basic_function(self):
        """Test adding span attributes for basic function."""
        mock_span = Mock()

        def test_func(arg1: str, arg2: int):
            return f"{arg1}_{arg2}"

        _add_span_attributes(mock_span, test_func, ("hello", 42), {}, "workflow")

        # Check entity type attribute
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "workflow")

        # Check input attributes
        expected_input = json.dumps({"arg1": "hello", "arg2": "42"})
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.input", expected_input)

    def test_add_span_attributes_with_kwargs(self):
        """Test adding span attributes with keyword arguments."""
        mock_span = Mock()

        def test_func(arg1: str, arg2: int = 10):
            return f"{arg1}_{arg2}"

        _add_span_attributes(mock_span, test_func, ("hello",), {"arg2": 42}, "agent")

        # Check entity type attribute
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "agent")

        # Check input attributes include both args and kwargs
        expected_input = json.dumps({"arg1": "hello", "arg2": "42"})
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.input", expected_input)

    def test_add_span_attributes_with_self_parameter(self):
        """Test adding span attributes ignoring self parameter."""
        mock_span = Mock()

        def test_method(self, arg1: str):
            return arg1

        _add_span_attributes(mock_span, test_method, ("self_instance", "hello"), {}, "task")

        # Check that self parameter is ignored
        expected_input = json.dumps({"arg1": "hello"})
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.input", expected_input)

    def test_add_span_attributes_with_cls_parameter(self):
        """Test adding span attributes ignoring cls parameter."""
        mock_span = Mock()

        def test_classmethod(cls, arg1: str):
            return arg1

        _add_span_attributes(mock_span, test_classmethod, ("cls_instance", "hello"), {}, "workflow")

        # Check that cls parameter is ignored
        expected_input = json.dumps({"arg1": "hello"})
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.input", expected_input)

    def test_add_span_attributes_exception_handling(self):
        """Test span attribute addition with exception handling."""
        mock_span = Mock()

        def problematic_func():
            pass

        # Mock inspect.signature to raise an exception
        with patch("netra.decorators.inspect.signature", side_effect=ValueError("Signature error")):
            _add_span_attributes(mock_span, problematic_func, (), {}, "workflow")

        # Check that error is recorded
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.input_error", "Signature error")


class TestAddOutputAttributes:
    """Test output attribute addition functionality."""

    def test_add_output_attributes_simple_result(self):
        """Test adding output attributes for simple result."""
        mock_span = Mock()
        result = "test_result"

        _add_output_attributes(mock_span, result)

        mock_span.set_attribute.assert_called_once_with(f"{Config.LIBRARY_NAME}.entity.output", "test_result")

    def test_add_output_attributes_complex_result(self):
        """Test adding output attributes for complex result."""
        mock_span = Mock()
        result = {"status": "success", "data": [1, 2, 3]}

        _add_output_attributes(mock_span, result)

        expected_output = '{"status": "success", "data": [1, 2, 3]}'
        mock_span.set_attribute.assert_called_once_with(f"{Config.LIBRARY_NAME}.entity.output", expected_output)

    def test_add_output_attributes_exception_handling(self):
        """Test output attribute addition with exception handling."""
        mock_span = Mock()

        class ProblematicResult:
            def __str__(self):
                raise ValueError("Cannot serialize")

        result = ProblematicResult()

        # Mock _serialize_value to raise an exception
        with patch("netra.decorators._serialize_value", side_effect=ValueError("Cannot serialize")):
            _add_output_attributes(mock_span, result)

        mock_span.set_attribute.assert_called_once_with(
            f"{Config.LIBRARY_NAME}.entity.output_error", "Cannot serialize"
        )


class TestCreateFunctionWrapper:
    """Test function wrapper creation functionality."""

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_sync_function_wrapper(self, mock_trace, mock_session_manager):
        """Test wrapping synchronous function."""
        # Arrange
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        def original_func(x: int, y: int) -> int:
            return x + y

        # Act
        wrapped_func = _create_function_wrapper(original_func, "workflow", "test_span")
        result = wrapped_func(3, 5)

        # Assert
        assert result == 8
        mock_session_manager.push_entity.assert_called_once_with("workflow", "test_span")
        mock_session_manager.pop_entity.assert_called_once_with("workflow")
        mock_trace.get_tracer.assert_called_once_with("original_func")
        mock_tracer.start_span.assert_called_once_with("test_span")

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_async_function_wrapper(self, mock_trace, mock_session_manager):
        """Test wrapping asynchronous function."""
        # Arrange
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        async def original_async_func(x: int, y: int) -> int:
            return x * y

        # Act
        wrapped_func = _create_function_wrapper(original_async_func, "agent", "async_test_span")

        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(wrapped_func(4, 6))
        finally:
            loop.close()

        # Assert
        assert result == 24
        mock_session_manager.push_entity.assert_called_once_with("agent", "async_test_span")
        mock_session_manager.pop_entity.assert_called_once_with("agent")
        mock_trace.get_tracer.assert_called_once_with("original_async_func")
        mock_tracer.start_span.assert_called_once_with("async_test_span")

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_function_wrapper_with_exception(self, mock_trace, mock_session_manager):
        """Test function wrapper handling exceptions."""
        # Arrange
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        def failing_func():
            raise ValueError("Test error")

        # Act & Assert
        wrapped_func = _create_function_wrapper(failing_func, "task", "failing_span")

        with pytest.raises(ValueError, match="Test error"):
            wrapped_func()

        # Verify error was recorded in span
        mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.error", "Test error")
        mock_session_manager.push_entity.assert_called_once_with("task", "failing_span")
        mock_session_manager.pop_entity.assert_called_once_with("task")

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_function_wrapper_default_name(self, mock_trace, mock_session_manager):
        """Test function wrapper with default name."""
        # Arrange
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        def test_function():
            return "success"

        # Act
        wrapped_func = _create_function_wrapper(test_function, "workflow")
        result = wrapped_func()

        # Assert
        assert result == "success"
        mock_tracer.start_span.assert_called_once_with("test_function")


class TestWrapClassMethods:
    """Test class method wrapping functionality."""

    def test_wrap_class_methods_basic(self):
        """Test wrapping basic class methods."""

        # Arrange
        class TestClass:
            def public_method(self, x: int) -> int:
                return x * 2

            def _private_method(self, x: int) -> int:
                return x * 3

            @property
            def some_property(self) -> str:
                return "property_value"

        # Act
        wrapped_class = _wrap_class_methods(TestClass, "agent", "TestAgent")

        # Assert
        assert wrapped_class is TestClass

        # Check that public method was wrapped
        assert hasattr(TestClass.public_method, "__wrapped__")

        # Check that private method was not wrapped (starts with _)
        assert not hasattr(TestClass._private_method, "__wrapped__")

    def test_wrap_class_methods_with_custom_name(self):
        """Test wrapping class methods with custom name."""

        # Arrange
        class CustomClass:
            def method_one(self) -> str:
                return "one"

            def method_two(self) -> str:
                return "two"

        # Act
        wrapped_class = _wrap_class_methods(CustomClass, "task", "CustomTask")

        # Assert
        assert wrapped_class is CustomClass
        assert hasattr(CustomClass.method_one, "__wrapped__")
        assert hasattr(CustomClass.method_two, "__wrapped__")

    def test_wrap_class_methods_default_name(self):
        """Test wrapping class methods with default name."""

        # Arrange
        class DefaultClass:
            def test_method(self) -> str:
                return "test"

        # Act
        wrapped_class = _wrap_class_methods(DefaultClass, "workflow")

        # Assert
        assert wrapped_class is DefaultClass
        assert hasattr(DefaultClass.test_method, "__wrapped__")


class TestWorkflowDecorator:
    """Test workflow decorator functionality."""

    def test_workflow_decorator_on_function(self):
        """Test workflow decorator applied to function."""

        @workflow
        def test_function(x: int) -> int:
            return x + 1

        # Check that function was wrapped
        assert hasattr(test_function, "__wrapped__")
        assert test_function.__name__ == "test_function"

    def test_workflow_decorator_on_function_with_name(self):
        """Test workflow decorator with custom name."""

        @workflow(name="custom_workflow")
        def test_function(x: int) -> int:
            return x + 1

        # Check that function was wrapped
        assert hasattr(test_function, "__wrapped__")

    def test_workflow_decorator_on_class(self):
        """Test workflow decorator applied to class."""

        @workflow
        class TestWorkflowClass:
            def process(self, data: str) -> str:
                return f"processed_{data}"

        # Check that class methods were wrapped
        assert hasattr(TestWorkflowClass.process, "__wrapped__")

    def test_workflow_decorator_on_class_with_name(self):
        """Test workflow decorator on class with custom name."""

        @workflow(name="CustomWorkflow")
        class TestClass:
            def execute(self) -> str:
                return "executed"

        # Check that class methods were wrapped
        assert hasattr(TestClass.execute, "__wrapped__")

    def test_workflow_decorator_as_function_call(self):
        """Test workflow decorator used as function call."""

        def original_function(x: int) -> int:
            return x * 2

        decorated_function = workflow(original_function)

        # Check that function was wrapped
        assert hasattr(decorated_function, "__wrapped__")
        assert decorated_function.__name__ == "original_function"


class TestAgentDecorator:
    """Test agent decorator functionality."""

    def test_agent_decorator_on_function(self):
        """Test agent decorator applied to function."""

        @agent
        def test_agent_function(query: str) -> str:
            return f"agent_response_{query}"

        # Check that function was wrapped
        assert hasattr(test_agent_function, "__wrapped__")
        assert test_agent_function.__name__ == "test_agent_function"

    def test_agent_decorator_on_function_with_name(self):
        """Test agent decorator with custom name."""

        @agent(name="custom_agent")
        def test_function(query: str) -> str:
            return f"response_{query}"

        # Check that function was wrapped
        assert hasattr(test_function, "__wrapped__")

    def test_agent_decorator_on_class(self):
        """Test agent decorator applied to class."""

        @agent
        class TestAgentClass:
            def think(self, problem: str) -> str:
                return f"thinking_about_{problem}"

            def act(self, action: str) -> str:
                return f"executing_{action}"

        # Check that class methods were wrapped
        assert hasattr(TestAgentClass.think, "__wrapped__")
        assert hasattr(TestAgentClass.act, "__wrapped__")

    def test_agent_decorator_on_class_with_name(self):
        """Test agent decorator on class with custom name."""

        @agent(name="SmartAgent")
        class IntelligentAgent:
            def reason(self, data: str) -> str:
                return f"reasoning_{data}"

        # Check that class methods were wrapped
        assert hasattr(IntelligentAgent.reason, "__wrapped__")


class TestTaskDecorator:
    """Test task decorator functionality."""

    def test_task_decorator_on_function(self):
        """Test task decorator applied to function."""

        @task
        def test_task_function(input_data: str) -> str:
            return f"task_result_{input_data}"

        # Check that function was wrapped
        assert hasattr(test_task_function, "__wrapped__")
        assert test_task_function.__name__ == "test_task_function"

    def test_task_decorator_on_function_with_name(self):
        """Test task decorator with custom name."""

        @task(name="data_processing_task")
        def process_data(data: str) -> str:
            return f"processed_{data}"

        # Check that function was wrapped
        assert hasattr(process_data, "__wrapped__")

    def test_task_decorator_on_class(self):
        """Test task decorator applied to class."""

        @task
        class TestTaskClass:
            def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "executed", "params": params}

            def validate(self, data: str) -> bool:
                return len(data) > 0

        # Check that class methods were wrapped
        assert hasattr(TestTaskClass.execute, "__wrapped__")
        assert hasattr(TestTaskClass.validate, "__wrapped__")

    def test_task_decorator_on_class_with_name(self):
        """Test task decorator on class with custom name."""

        @task(name="DataProcessor")
        class ProcessorTask:
            def transform(self, input_data: str) -> str:
                return input_data.upper()

        # Check that class methods were wrapped
        assert hasattr(ProcessorTask.transform, "__wrapped__")


class TestDecoratorIntegration:
    """Test decorator integration scenarios."""

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_nested_decorators(self, mock_trace, mock_session_manager):
        """Test using multiple decorators together."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        @workflow
        def main_workflow(data: str) -> str:
            return process_task(data)

        @task
        def process_task(data: str) -> str:
            return f"processed_{data}"

        # Execute the workflow
        result = main_workflow("test_data")

        # Verify result
        assert result == "processed_test_data"

        # Verify both decorators were applied
        assert hasattr(main_workflow, "__wrapped__")
        assert hasattr(process_task, "__wrapped__")

    def test_decorator_preserves_function_metadata(self):
        """Test that decorators preserve function metadata."""

        @workflow
        def documented_function(x: int, y: str = "default") -> str:
            """This is a documented function."""
            return f"{y}_{x}"

        # Check that metadata is preserved
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."

        # Check that signature is accessible
        sig = inspect.signature(documented_function)
        assert len(sig.parameters) == 2
        assert "x" in sig.parameters
        assert "y" in sig.parameters

    @patch("netra.decorators.SessionManager")
    @patch("netra.decorators.trace")
    def test_async_decorator_integration(self, mock_trace, mock_session_manager):
        """Test decorator integration with async functions."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=None)
        mock_context_manager.__exit__ = Mock(return_value=None)

        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_span.return_value = mock_span
        mock_trace.use_span.return_value = mock_context_manager

        @agent
        async def async_agent(query: str) -> str:
            return f"async_response_{query}"

        # Check that async function was wrapped
        assert hasattr(async_agent, "__wrapped__")
        assert inspect.iscoroutinefunction(async_agent)

        # Test execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_agent("test_query"))
            assert result == "async_response_test_query"
        finally:
            loop.close()

    def test_decorator_error_handling_integration(self):
        """Test error handling across decorated functions."""

        @task
        def failing_task(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Task failed")
            return "success"

        # Test successful execution
        result = failing_task(False)
        assert result == "success"

        # Test error propagation
        with pytest.raises(ValueError, match="Task failed"):
            failing_task(True)
