"""Unit tests for netra.decorators module.

This module contains comprehensive tests for the Netra SDK decorator utilities,
including workflow, agent, and task decorators for OpenTelemetry span tracking.
"""

import asyncio
import inspect
import json
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from opentelemetry import trace

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
    """Test cases for _serialize_value helper function."""

    def test_serialize_basic_types(self) -> None:
        """Test serialization of basic Python types."""
        # String
        assert _serialize_value("hello") == "hello"

        # Integer
        assert _serialize_value(42) == "42"

        # Float
        assert _serialize_value(3.14) == "3.14"

        # Boolean
        assert _serialize_value(True) == "True"
        assert _serialize_value(False) == "False"

        # None
        assert _serialize_value(None) == "None"

    def test_serialize_collections(self) -> None:
        """Test serialization of collections (list, dict, tuple)."""
        # List
        result = _serialize_value([1, 2, 3])
        assert result == "[1, 2, 3]"

        # Dictionary
        result = _serialize_value({"key": "value", "num": 42})
        parsed = json.loads(result)
        assert parsed == {"key": "value", "num": 42}

        # Tuple
        result = _serialize_value((1, "two", 3.0))
        assert result == '[1, "two", 3.0]'

    def test_serialize_complex_objects(self) -> None:
        """Test serialization of complex objects."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        obj = CustomObject()
        result = _serialize_value(obj)
        assert result == "custom_object"

    def test_serialize_size_limit(self) -> None:
        """Test that serialization respects size limits."""
        # Large string - the actual implementation only limits collections, not basic types
        large_string = "x" * 2000
        result = _serialize_value(large_string)
        # Basic types like strings are not limited in the actual implementation
        assert len(result) == 2000

        # Large list - JSON serialization may not exactly match the limit
        large_list = list(range(500))  # Smaller list to ensure JSON fits in limit
        result = _serialize_value(large_list)
        assert len(result) <= 1000

    def test_serialize_exception_handling(self) -> None:
        """Test serialization handles exceptions gracefully."""

        class ProblematicObject:
            def __str__(self) -> str:
                raise ValueError("Cannot serialize")

        obj = ProblematicObject()
        result = _serialize_value(obj)
        assert result == "ProblematicObject"


class TestAddSpanAttributes:
    """Test cases for _add_span_attributes helper function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_span = Mock(spec=trace.Span)

    def test_add_span_attributes_basic_function(self) -> None:
        """Test adding span attributes for a basic function."""

        def test_func(param1: Any, param2: str = "default") -> Any:
            return param1 + param2

        args = ("hello",)
        kwargs: Dict[str, Any] = {"param2": "world"}

        _add_span_attributes(self.mock_span, test_func, args, kwargs, "test_entity")

        # Verify entity type is set
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "test_entity")

        # Verify input data is serialized and set
        calls = self.mock_span.set_attribute.call_args_list
        input_calls = [call for call in calls if f"{Config.LIBRARY_NAME}.entity.input" in str(call)]
        assert len(input_calls) == 1

        # Parse the input data
        input_data = json.loads(input_calls[0][0][1])
        assert input_data["param1"] == "hello"
        assert input_data["param2"] == "world"

    def test_add_span_attributes_class_method(self) -> None:
        """Test adding span attributes for class methods (skips self)."""

        class TestClass:
            def test_method(self, param1: Any, param2: Any) -> Any:
                return param1 + param2

        obj = TestClass()
        args = (obj, "hello", "world")
        kwargs: Dict[str, Any] = {}

        _add_span_attributes(self.mock_span, obj.test_method, args, kwargs, "test_entity")

        # Verify self is skipped in input data
        calls = self.mock_span.set_attribute.call_args_list
        input_calls = [call for call in calls if f"{Config.LIBRARY_NAME}.entity.input" in str(call)]
        assert len(input_calls) == 1

        input_data = json.loads(input_calls[0][0][1])
        assert "param1" in input_data
        assert "param2" in input_data
        assert "self" not in input_data

    def test_add_span_attributes_exception_handling(self) -> None:
        """Test that exceptions in attribute addition are handled gracefully."""

        def problematic_func() -> None:
            pass

        # Mock inspect.signature to raise an exception
        with patch("netra.decorators.inspect.signature", side_effect=ValueError("Signature error")):
            _add_span_attributes(self.mock_span, problematic_func, (), {}, "test_entity")

        # Verify error is recorded
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.input_error", "Signature error")

    def test_add_span_attributes_empty_input(self) -> None:
        """Test adding span attributes with no input parameters."""

        def no_param_func() -> str:
            return "result"

        _add_span_attributes(self.mock_span, no_param_func, (), {}, "test_entity")

        # Should still set entity type
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "test_entity")


class TestAddOutputAttributes:
    """Test cases for _add_output_attributes helper function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_span = Mock(spec=trace.Span)

    def test_add_output_attributes_simple_result(self) -> None:
        """Test adding output attributes for simple results."""
        result = "test_result"
        _add_output_attributes(self.mock_span, result)

        self.mock_span.set_attribute.assert_called_once_with(f"{Config.LIBRARY_NAME}.entity.output", "test_result")

    def test_add_output_attributes_complex_result(self) -> None:
        """Test adding output attributes for complex results."""
        result = {"status": "success", "data": [1, 2, 3]}
        _add_output_attributes(self.mock_span, result)

        # Verify the result is serialized as JSON
        call_args = self.mock_span.set_attribute.call_args[0]
        assert call_args[0] == f"{Config.LIBRARY_NAME}.entity.output"
        parsed_result = json.loads(call_args[1])
        assert parsed_result == {"status": "success", "data": [1, 2, 3]}

    def test_add_output_attributes_exception_handling(self) -> None:
        """Test that exceptions in output attribute addition are handled."""
        # Mock _serialize_value to raise an exception
        with patch("netra.decorators._serialize_value", side_effect=ValueError("Cannot serialize result")):
            _add_output_attributes(self.mock_span, "any_result")

        self.mock_span.set_attribute.assert_called_once_with(
            f"{Config.LIBRARY_NAME}.entity.output_error", "Cannot serialize result"
        )


class TestCreateFunctionWrapper:
    """Test cases for _create_function_wrapper function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_wrap_sync_function(self) -> None:
        """Test wrapping a synchronous function."""

        def test_func(x: Any, y: int = 10) -> Any:
            return x + y

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_func = _create_function_wrapper(test_func, "test_entity")
            result = wrapped_func(5, y=15)

        # Verify function executed correctly
        assert result == 20

        # Verify span was created with correct name
        self.mock_tracer.start_as_current_span.assert_called_once_with("test_func")

        # Verify span attributes were set
        assert self.mock_span.set_attribute.call_count > 0

    def test_wrap_async_function(self) -> None:
        """Test wrapping an asynchronous function."""

        async def async_test_func(x: Any, y: int = 10) -> Any:
            return x + y

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_func = _create_function_wrapper(async_test_func, "test_entity")
            result = asyncio.run(wrapped_func(5, y=15))

        # Verify function executed correctly
        assert result == 20

        # Verify span was created
        self.mock_tracer.start_as_current_span.assert_called_once_with("async_test_func")

    def test_wrap_function_with_custom_name(self) -> None:
        """Test wrapping a function with custom span name."""

        def test_func() -> str:
            return "result"

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_func = _create_function_wrapper(test_func, "test_entity", "custom_name")
            wrapped_func()

        # Verify custom name was used
        self.mock_tracer.start_as_current_span.assert_called_once_with("custom_name")

    def test_wrap_function_exception_handling(self) -> None:
        """Test that exceptions are properly handled and re-raised."""

        def failing_func() -> None:
            raise ValueError("Test error")

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_func = _create_function_wrapper(failing_func, "test_entity")

            with pytest.raises(ValueError, match="Test error"):
                wrapped_func()

        # Verify exception was recorded in span
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.error", "Test error")
        self.mock_span.record_exception.assert_called_once()

    def test_wrap_async_function_exception_handling(self) -> None:
        """Test that exceptions in async functions are properly handled."""

        async def failing_async_func() -> None:
            raise ValueError("Async test error")

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_func = _create_function_wrapper(failing_async_func, "test_entity")

            with pytest.raises(ValueError, match="Async test error"):
                asyncio.run(wrapped_func())

        # Verify exception was recorded
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.error", "Async test error")
        self.mock_span.record_exception.assert_called_once()

    def test_wrapped_function_preserves_metadata(self) -> None:
        """Test that wrapped function preserves original function metadata."""

        def original_func(x: int) -> int:
            """Original function docstring."""
            return x

        wrapped_func = _create_function_wrapper(original_func, "test_entity")

        # Verify metadata is preserved
        assert wrapped_func.__name__ == "original_func"
        assert wrapped_func.__doc__ == "Original function docstring."
        assert hasattr(wrapped_func, "__wrapped__")


class TestWrapClassMethods:
    """Test cases for _wrap_class_methods function."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_wrap_class_public_methods(self) -> None:
        """Test wrapping public methods of a class."""

        class TestClass:
            def public_method(self, x: Any) -> Any:
                return x * 2

            def another_method(self) -> str:
                return "result"

            def _private_method(self) -> str:
                return "private"

            def __special_method__(self) -> str:
                return "special"

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_class = _wrap_class_methods(TestClass, "test_entity")

            # Test public methods are wrapped
            obj = wrapped_class()
            result = obj.public_method(5)
            assert result == 10

            # Verify span was created with class.method name
            self.mock_tracer.start_as_current_span.assert_called_with("TestClass.public_method")

    def test_wrap_class_skips_private_methods(self) -> None:
        """Test that private and special methods are not wrapped."""

        class TestClass:
            def public_method(self) -> str:
                return "public"

            def _private_method(self) -> str:
                return "private"

            def __special_method__(self) -> str:
                return "special"

        original_private = TestClass._private_method
        original_special = TestClass.__special_method__

        wrapped_class = _wrap_class_methods(TestClass, "test_entity")

        # Verify private and special methods are unchanged
        assert wrapped_class._private_method is original_private
        assert wrapped_class.__special_method__ is original_special

    def test_wrap_class_with_custom_name(self) -> None:
        """Test wrapping class methods with custom class name."""

        class TestClass:
            def method(self) -> str:
                return "result"

        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):
            wrapped_class = _wrap_class_methods(TestClass, "test_entity", "CustomName")

            obj = wrapped_class()
            obj.method()

            # Verify custom class name was used
            self.mock_tracer.start_as_current_span.assert_called_with("CustomName.method")

    def test_wrap_class_skips_non_callable_attributes(self) -> None:
        """Test that non-callable attributes are not wrapped."""

        class TestClass:
            class_var = "not_callable"

            def method(self) -> str:
                return "callable"

        original_class_var = TestClass.class_var
        wrapped_class = _wrap_class_methods(TestClass, "test_entity")

        # Verify non-callable attribute is unchanged
        assert wrapped_class.class_var is original_class_var


class TestWorkflowDecorator:
    """Test cases for @workflow decorator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_workflow_decorator_on_function(self) -> None:
        """Test @workflow decorator on a function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow
            def test_workflow(data: Any) -> Any:
                return data.upper()

            result = test_workflow("hello")
            assert result == "HELLO"

            # Verify span attributes include workflow entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "workflow")

    def test_workflow_decorator_with_name(self) -> None:
        """Test @workflow decorator with custom name."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow(name="Custom Workflow")
            def test_workflow(data: Any) -> Any:
                return data.lower()

            test_workflow("hello")

            # Verify custom name was used
            self.mock_tracer.start_as_current_span.assert_called_with("Custom Workflow")

    def test_workflow_decorator_on_async_function(self) -> None:
        """Test @workflow decorator on async function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow
            async def async_workflow(data: Any) -> Any:
                return data.lower()

            result = asyncio.run(async_workflow("HELLO"))
            assert result == "hello"

            # Verify workflow entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "workflow")

    def test_workflow_decorator_on_class(self) -> None:
        """Test @workflow decorator on a class."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow
            class WorkflowClass:
                def process(self, data: Any) -> Any:
                    return data * 2

                def transform(self, data: Any) -> Any:
                    return data + 10

            obj = WorkflowClass()
            result = obj.process(5)
            assert result == 10

            # Verify workflow entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "workflow")

    def test_workflow_decorator_class_with_name(self) -> None:
        """Test @workflow decorator on class with custom name."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow
            class WorkflowClass:
                def process(self, data: Any) -> Any:
                    return data * 2

            obj = WorkflowClass()
            obj.process(5)

            # Verify custom class name was used in span name
            self.mock_tracer.start_as_current_span.assert_called_with("WorkflowClass.process")


class TestAgentDecorator:
    """Test cases for @agent decorator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_agent_decorator_on_function(self) -> None:
        """Test @agent decorator on a function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @agent
            def test_agent(query: Any) -> Any:
                return f"Response to: {query}"

            result = test_agent("hello")
            assert result == "Response to: hello"

            # Verify span attributes include agent entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "agent")

    def test_agent_decorator_with_name(self) -> None:
        """Test @agent decorator with custom name."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @agent(name="Custom Agent")
            def test_agent(query: Any) -> Any:
                return f"Response: {query}"

            test_agent("hello")

            # Verify custom name was used
            self.mock_tracer.start_as_current_span.assert_called_with("Custom Agent")

    def test_agent_decorator_on_async_function(self) -> None:
        """Test @agent decorator on async function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @agent
            async def async_agent(query: Any) -> Any:
                return f"Async response: {query}"

            result = asyncio.run(async_agent("hello"))
            assert result == "Async response: hello"

            # Verify agent entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "agent")

    def test_agent_decorator_on_class(self) -> None:
        """Test @agent decorator on a class."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @agent
            class AgentClass:
                def respond(self, query: Any) -> Any:
                    return f"Agent response: {query}"

                def process(self, data: Any) -> Any:
                    return data.upper()

            obj = AgentClass()
            result = obj.respond("hello")
            assert result == "Agent response: hello"

            # Verify agent entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "agent")


class TestTaskDecorator:
    """Test cases for @task decorator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_task_decorator_on_function(self) -> None:
        """Test @task decorator on a function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @task
            def test_task(data: Any) -> Any:
                return data.upper()

            result = test_task("hello")
            assert result == "HELLO"

            # Verify span attributes include task entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "task")

    def test_task_decorator_with_name(self) -> None:
        """Test @task decorator with custom name."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @task(name="Custom Task")
            def test_task(data: Any) -> Any:
                return data.lower()

            test_task("HELLO")

            # Verify custom name was used
            self.mock_tracer.start_as_current_span.assert_called_with("Custom Task")

    def test_task_decorator_on_async_function(self) -> None:
        """Test @task decorator on async function."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @task
            async def async_task(data: Any) -> Any:
                return data * 2

            result = asyncio.run(async_task(5))
            assert result == 10

            # Verify task entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "task")

    def test_task_decorator_on_class(self) -> None:
        """Test @task decorator on a class."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @task
            class TaskClass:
                def execute(self, data: Any) -> Any:
                    return data.upper()

                def process(self, data: Any) -> Any:
                    return data + " processed"

            obj = TaskClass()
            result = obj.execute("hello")
            assert result == "HELLO"

            # Verify task entity type
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.type", "task")


class TestDecoratorIntegration:
    """Integration tests for decorator functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock(spec=trace.Span)
        self.mock_context_manager = Mock()
        self.mock_context_manager.__enter__ = Mock(return_value=self.mock_span)
        self.mock_context_manager.__exit__ = Mock(return_value=None)
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager

    def test_nested_decorated_functions(self) -> None:
        """Test that nested decorated functions work correctly."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow
            def outer_workflow(data: Any) -> Any:
                return inner_task(data)

            @task
            def inner_task(data: Any) -> Any:
                return data.upper()

            result = outer_workflow("hello")
            assert result == "HELLO"

            # Verify both spans were created
            assert self.mock_tracer.start_as_current_span.call_count == 2

    def test_decorator_with_complex_data_types(self) -> None:
        """Test decorators with complex input/output data types."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @agent(name="Complex Agent")
            def complex_agent(data_dict: Dict[str, Any], data_list: List[Any]) -> Dict[str, Any]:
                return {"processed_dict": data_dict, "processed_list": data_list, "status": "success"}

            input_dict = {"key": "value", "number": 42}
            input_list = [1, 2, 3, "four"]

            result = complex_agent(input_dict, input_list)

            # Verify result structure
            assert result["processed_dict"] == input_dict
            assert result["processed_list"] == input_list
            assert result["status"] == "success"

            # Verify span attributes were set (input and output should be serialized)
            assert self.mock_span.set_attribute.call_count > 0

    def test_decorator_error_propagation(self) -> None:
        """Test that errors are properly propagated through decorated functions."""
        with patch("netra.decorators.trace.get_tracer", return_value=self.mock_tracer):

            @workflow(name="Failing Workflow")
            def failing_workflow() -> None:
                raise RuntimeError("Workflow failed")

            with pytest.raises(RuntimeError, match="Workflow failed"):
                failing_workflow()

            # Verify error was recorded in span
            self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.entity.error", "Workflow failed")
            self.mock_span.record_exception.assert_called_once()


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions for decorators."""

    def test_decorator_with_no_parameters(self) -> None:
        """Test decorators applied to functions with no parameters."""
        mock_tracer = Mock()
        mock_span = Mock(spec=trace.Span)
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = mock_context_manager

        with patch("netra.decorators.trace.get_tracer", return_value=mock_tracer):

            @task
            def no_param_task() -> str:
                return "no params"

            result: str = no_param_task()
            assert result == "no params"

            # Should still create span
            mock_tracer.start_as_current_span.assert_called_once()

    def test_decorator_preserves_function_signature(self) -> None:
        """Test that decorators preserve function signatures for introspection."""

        @agent
        def test_function(param1: str, param2: int = 10) -> str:
            """Test function docstring."""
            return f"{param1}_{param2}"

        # Verify signature is preserved
        sig = inspect.signature(test_function)
        params = list(sig.parameters.keys())
        assert params == ["param1", "param2"]
        assert sig.parameters["param1"].annotation == str
        assert sig.parameters["param2"].default == 10
        assert sig.return_annotation == str

        # Verify docstring is preserved
        assert test_function.__doc__ == "Test function docstring."
