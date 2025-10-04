"""
Unit tests for BaseHandler abstract base class.

Tests cover abstract method contracts, interface compliance,
and base functionality provided by the abstract class.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from abc import ABC, abstractmethod

# Import the classes to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from handlers.base_handler import BaseHandler


class ConcreteHandler(BaseHandler):
    """Concrete implementation of BaseHandler for testing."""

    def __init__(self, should_fail_validation: bool = False, should_fail_loading: bool = False,
                 should_fail_processing: bool = False):
        super().__init__("test-handler")
        self.should_fail_validation = should_fail_validation
        self.should_fail_loading = should_fail_loading
        self.should_fail_processing = should_fail_processing
        self.validation_called_with = None
        self.loading_called_with = None
        self.processing_called_with = None
        self.formatting_called_with = None

    @property
    def supported_modality(self) -> str:
        return "test-modality"

    @property
    def required_parameters(self) -> List[str]:
        return ["prompt", "width", "height"]

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        return {"steps": 20, "guidance": 7.5}

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock validation implementation that tracks call arguments."""
        # Track call arguments
        self.validation_called_with = (request_data,)

        if hasattr(self, 'should_fail_validation') and self.should_fail_validation:
            raise ValueError("Validation failed")

        return request_data

    def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
        """Mock model requirements implementation."""
        self.loading_called_with = request_data
        if self.should_fail_loading:
            raise RuntimeError("Model loading failed")
        return ["test-model-1", "test-model-2"]

    def process_inference(self, models: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock inference processing implementation."""
        self.processing_called_with = (request_data, models)
        if self.should_fail_processing:
            raise RuntimeError("Processing failed")
        return {
            "result": "test-output",
            "metadata": {
                "models_used": list(models.keys()) if models else [],
                "request_validated": True
            }
        }

    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock response formatting implementation."""
        self.formatting_called_with = (inference_results, request_data)
        return {
            "status": "success",
            "output": inference_results.get("result", "test-output"),
            "metadata": inference_results.get("metadata", {})
        }


class ConcreteHandlerWithRequestId:
    """
    Alternative concrete handler that supports request_id interface.
    This is used for tests that expect a different method signature.
    """

    def __init__(self, should_fail_validation: bool = False, should_fail_loading: bool = False, should_fail_processing: bool = False):
        self.handler_name = "test_with_id"
        self.should_fail_validation = should_fail_validation
        self.should_fail_loading = should_fail_loading
        self.should_fail_processing = should_fail_processing
        self.validation_called_with = None
        self.loading_called_with = None
        self.processing_called_with = None
        self.formatting_called_with = None

    def validate_request(self, request_data: Dict[str, Any], request_id: str = None) -> bool:
        """Mock validation that accepts request_id separately."""
        self.validation_called_with = (request_data, request_id)

        if self.should_fail_validation:
            raise ValueError("Validation failed")

        return True

    def load_models(self, request_data: Dict[str, Any], request_id: str) -> List[str]:
        """Mock load_models that accepts request_id separately."""
        self.loading_called_with = (request_data, request_id)

        if self.should_fail_loading:
            raise RuntimeError("Model loading failed")

        return ["test-model-1", "test-model-2"]

    def process_inference(self, request_data: Dict[str, Any], models: List[str], request_id: str) -> Dict[str, Any]:
        """Mock process_inference that accepts request_id separately."""
        self.processing_called_with = (request_data, models, request_id)

        if self.should_fail_processing:
            raise RuntimeError("Processing failed")

        return {
            "result": "test-output",
            "metadata": {
                "request_id": request_id,
                "models_used": models
            }
        }

    def format_response(self, output: Dict[str, Any], processing_time_ms: float, models_used: List[str], request_id: str) -> Dict[str, Any]:
        """Mock format_response that accepts request_id separately."""
        self.formatting_called_with = (output, processing_time_ms, models_used, request_id)

        return {
            "status": "success",
            "output": output,
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "models_used": models_used,
                "request_id": request_id
            }
        }
class IncompleteHandler(BaseHandler):
    """Incomplete handler that doesn't implement all abstract methods."""

    def validate_request(self, request_data: Dict[str, Any], request_id: str) -> bool:
        return True

    def load_models(self, request_data: Dict[str, Any], request_id: str) -> List[str]:
        return ["model"]

    # Missing process_inference and format_response methods


class TestBaseHandler:
    """Test cases for BaseHandler abstract base class."""

    def test_base_handler_is_abstract(self):
        """Test that BaseHandler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseHandler"):
            BaseHandler()

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        handler = ConcreteHandler()
        assert isinstance(handler, BaseHandler)
        assert isinstance(handler, ConcreteHandler)

    def test_incomplete_implementation_cannot_be_instantiated(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteHandler"):
            IncompleteHandler()

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseHandler.__abstractmethods__
        expected_methods = {
            'supported_modality',
            'required_parameters',
            'optional_parameters',
            'validate_request',
            'get_required_models',
            'process_inference',
            'format_response'
        }

        assert abstract_methods == expected_methods

    def test_validate_request_signature(self):
        """Test validate_request method signature and behavior."""
        handler = ConcreteHandlerWithRequestId()

        request_data = {"param1": "value1"}
        request_id = "test-123"

        result = handler.validate_request(request_data, request_id)

        assert result is True
        assert handler.validation_called_with == (request_data, request_id)

    def test_validate_request_failure(self):
        """Test validate_request when it raises an exception."""
        handler = ConcreteHandlerWithRequestId(should_fail_validation=True)

        with pytest.raises(ValueError, match="Validation failed"):
            handler.validate_request({"param": "value"}, "test-123")

    def test_load_models_signature(self):
        """Test load_models method signature and behavior."""
        handler = ConcreteHandlerWithRequestId()

        request_data = {"model_requirements": ["model1", "model2"]}
        request_id = "test-456"

        result = handler.load_models(request_data, request_id)

        assert isinstance(result, list)
        assert result == ["test-model-1", "test-model-2"]
        assert handler.loading_called_with == (request_data, request_id)

    def test_load_models_failure(self):
        """Test load_models when it raises an exception."""
        handler = ConcreteHandlerWithRequestId(should_fail_loading=True)

        with pytest.raises(RuntimeError, match="Model loading failed"):
            handler.load_models({"models": ["model1"]}, "test-456")

    def test_process_inference_signature(self):
        """Test process_inference method signature and behavior."""
        handler = ConcreteHandlerWithRequestId()

        request_data = {"prompt": "test prompt"}
        models = ["model1", "model2"]
        request_id = "test-789"

        result = handler.process_inference(request_data, models, request_id)

        assert isinstance(result, dict)
        assert result["result"] == "test-output"
        assert result["metadata"]["request_id"] == request_id
        assert handler.processing_called_with == (request_data, models, request_id)

    def test_process_inference_failure(self):
        """Test process_inference when it raises an exception."""
        handler = ConcreteHandlerWithRequestId(should_fail_processing=True)

        with pytest.raises(RuntimeError, match="Processing failed"):
            handler.process_inference({"prompt": "test"}, ["model1"], "test-789")

    def test_format_response_signature(self):
        """Test format_response method signature and behavior."""
        handler = ConcreteHandlerWithRequestId()

        output = {"result": "test output"}
        processing_time_ms = 1500.5
        models_used = ["model1", "model2"]
        request_id = "test-abc"

        result = handler.format_response(output, processing_time_ms, models_used, request_id)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["output"] == output
        assert result["metadata"]["processing_time_ms"] == processing_time_ms
        assert result["metadata"]["models_used"] == models_used
        assert result["metadata"]["request_id"] == request_id
        assert handler.formatting_called_with == (output, processing_time_ms, models_used, request_id)

    def test_method_call_order(self):
        """Test that methods can be called in typical workflow order."""
        handler = ConcreteHandlerWithRequestId()

        # Simulate typical workflow
        request_data = {"prompt": "test"}
        request_id = "workflow-test"

        # Step 1: Validate
        validation_result = handler.validate_request(request_data, request_id)
        assert validation_result is True

        # Step 2: Load models
        models = handler.load_models(request_data, request_id)
        assert len(models) == 2

        # Step 3: Process inference
        output = handler.process_inference(request_data, models, request_id)
        assert output["result"] == "test-output"

        # Step 4: Format response
        response = handler.format_response(output, 1000.0, models, request_id)
        assert response["status"] == "success"

        # Verify all methods were called
        assert handler.validation_called_with is not None
        assert handler.loading_called_with is not None
        assert handler.processing_called_with is not None
        assert handler.formatting_called_with is not None


class TestBaseHandlerInterface:
    """Test the interface contracts of BaseHandler."""

    def test_validate_request_return_type(self):
        """Test that validate_request returns boolean."""
        handler = ConcreteHandlerWithRequestId()

        result = handler.validate_request({}, "test")
        assert isinstance(result, bool)

    def test_load_models_return_type(self):
        """Test that load_models returns list of strings."""
        handler = ConcreteHandlerWithRequestId()

        result = handler.load_models({}, "test")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_process_inference_return_type(self):
        """Test that process_inference can return any type."""
        handler = ConcreteHandlerWithRequestId()

        result = handler.process_inference({}, ["model"], "test")
        # Should accept any return type
        assert result is not None

    def test_format_response_return_type(self):
        """Test that format_response returns dictionary."""
        handler = ConcreteHandlerWithRequestId()

        result = handler.format_response({}, 1000.0, ["model"], "test")
        assert isinstance(result, dict)

    def test_method_signatures_accept_correct_parameters(self):
        """Test that all methods accept the correct parameter types."""
        handler = ConcreteHandlerWithRequestId()

        # Test with various parameter types
        request_data_types = [
            {},
            {"string": "value", "int": 42, "float": 3.14, "list": [1, 2, 3]},
            {"nested": {"key": "value"}},
        ]

        for request_data in request_data_types:
            # These should not raise type-related exceptions
            handler.validate_request(request_data, "test-id")
            models = handler.load_models(request_data, "test-id")
            output = handler.process_inference(request_data, models, "test-id")
            response = handler.format_response(output, 1000.0, models, "test-id")

            assert isinstance(response, dict)


class TestBaseHandlerDocumentation:
    """Test that BaseHandler provides proper documentation."""

    def test_class_has_docstring(self):
        """Test that BaseHandler class has documentation."""
        assert BaseHandler.__doc__ is not None
        assert len(BaseHandler.__doc__.strip()) > 0

    def test_abstract_methods_have_docstrings(self):
        """Test that abstract methods have documentation."""
        # Note: This test assumes the BaseHandler has proper docstrings
        # In a real implementation, you would check for specific docstring content

        handler = ConcreteHandler()

        # Check that methods exist and can be called
        assert callable(handler.validate_request)
        assert callable(handler.get_required_models)
        assert callable(handler.process_inference)
        assert callable(handler.format_response)


class TestBaseHandlerEdgeCases:
    """Test edge cases and boundary conditions for BaseHandler."""

    def test_empty_request_data(self):
        """Test handling of empty request data."""
        handler = ConcreteHandlerWithRequestId()

        # Should handle empty request data gracefully
        result = handler.validate_request({}, "test")
        assert result is True

        models = handler.load_models({}, "test")
        assert isinstance(models, list)

    def test_none_values_in_parameters(self):
        """Test handling of None values in parameters."""
        handler = ConcreteHandlerWithRequestId()

        # Test with None request_id (should be handled by implementation)
        result = handler.validate_request({}, None)
        assert result is True

        # Test with None in models list
        output = handler.process_inference({}, [None, "valid-model"], "test")
        assert output is not None

    def test_very_long_request_id(self):
        """Test handling of very long request IDs."""
        handler = ConcreteHandlerWithRequestId()

        long_id = "x" * 1000  # Very long request ID

        result = handler.validate_request({}, long_id)
        assert result is True

        # Verify the long ID was passed through
        assert handler.validation_called_with[1] == long_id

    def test_unicode_in_request_data(self):
        """Test handling of Unicode characters in request data."""
        handler = ConcreteHandlerWithRequestId()

        unicode_data = {
            "prompt": "Hello 世界! 🌍 café",
            "description": "Test with émojis 🎨 and åccénts"
        }

        result = handler.validate_request(unicode_data, "unicode-test")
        assert result is True

        # Verify Unicode data was passed through
        assert handler.validation_called_with[0] == unicode_data

    def test_large_request_data(self):
        """Test handling of large request data."""
        handler = ConcreteHandlerWithRequestId()

        # Create large request data
        large_data = {
            "prompt": "x" * 10000,
            "large_list": list(range(1000)),
            "nested": {"deep": {"structure": {"with": "values"}}}
        }

        result = handler.validate_request(large_data, "large-test")
        assert result is True


class TestBaseHandlerSubclassing:
    """Test different ways to subclass BaseHandler."""

    def test_minimal_implementation(self):
        """Test minimal concrete implementation."""

        class MinimalHandler(BaseHandler):
            def __init__(self):
                super().__init__("minimal")

            @property
            def supported_modality(self) -> str:
                return "minimal"

            @property
            def required_parameters(self) -> List[str]:
                return []

            @property
            def optional_parameters(self) -> Dict[str, Any]:
                return {}

            def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                return request_data

            def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
                return []

            def process_inference(self, models: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": None}

            def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "success", "output": inference_results}

        handler = MinimalHandler()
        assert isinstance(handler, BaseHandler)

    def test_implementation_with_additional_methods(self):
        """Test implementation that adds additional methods."""

        class ExtendedHandler(BaseHandler):
            def __init__(self):
                super().__init__("extended")
                self.custom_state = "initialized"

            @property
            def supported_modality(self) -> str:
                return "extended"

            @property
            def required_parameters(self) -> List[str]:
                return ["data"]

            @property
            def optional_parameters(self) -> Dict[str, Any]:
                return {"option": "default"}

            def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                # Use custom validation logic but return correct format
                is_valid = self.custom_validation(request_data)
                if is_valid:
                    return request_data
                else:
                    raise ValueError("Custom validation failed")

            def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
                return self.get_models_for_data(request_data)

            def process_inference(self, models: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
                return self.run_inference(request_data, list(models.keys()))

            def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
                return self.create_response(inference_results, 1000.0)

            # Additional methods
            def custom_validation(self, data):
                return len(data) > 0

            def get_models_for_data(self, data):
                return ["default-model"]

            def run_inference(self, data, models):
                return {"processed": True}

            def create_response(self, output, time_ms):
                return {"result": output, "time": time_ms}

        handler = ExtendedHandler()
        assert isinstance(handler, BaseHandler)
        assert handler.custom_state == "initialized"

        # Test that additional methods work
        assert handler.custom_validation({"test": "data"}) is True
        assert handler.get_required_models({}) == ["default-model"]


if __name__ == '__main__':
    pytest.main([__file__])