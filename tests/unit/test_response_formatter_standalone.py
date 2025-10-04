"""
Standalone tests for MMI-004 Response Formatter functionality.

These tests validate response formatting and error handling
without requiring the full application stack.
"""

import pytest
from enum import Enum
from typing import Dict, Any, Optional
import time


# Inline ResponseFormatter implementation for testing
class ErrorType(Enum):
    """Enumeration of error types for consistent error handling."""
    VALIDATION_ERROR = "validation_error"
    INFERENCE_ERROR = "inference_error"
    MODEL_LOAD_ERROR = "model_load_error"
    INTERNAL_ERROR = "internal_error"
    UNSUPPORTED_MODALITY_ERROR = "unsupported_modality_error"


class ResponseFormatter:
    """Formats responses for RunPod compatibility and consistency."""

    @staticmethod
    def format_success(output: Any, processing_time_ms: float,
                      models_used: list, request_id: str) -> Dict[str, Any]:
        """
        Format a successful inference response.

        Args:
            output: The inference result
            processing_time_ms: Time taken for processing
            models_used: List of models used in inference
            request_id: Unique request identifier

        Returns:
            Formatted success response
        """
        return {
            "status": "success",
            "output": output,
            "metadata": {
                "request_id": request_id,
                "processing_time_ms": round(processing_time_ms, 2),
                "models_used": models_used,
                "timestamp": time.time()
            }
        }

    @staticmethod
    def format_error(error_type: ErrorType, message: str,
                    request_id: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error_type: Type of error from ErrorType enum
            message: Human-readable error message
            request_id: Optional request identifier
            details: Optional additional error details

        Returns:
            Formatted error response
        """
        error_response = {
            "status": "error",
            "error": {
                "type": error_type.value,
                "message": message,
                "timestamp": time.time()
            }
        }

        if request_id:
            error_response["error"]["request_id"] = request_id

        if details:
            error_response["error"]["details"] = details

        return error_response

    @staticmethod
    def format_validation_error(validation_errors: Dict[str, Any],
                              request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Format a validation error response.

        Args:
            validation_errors: Dictionary containing validation error details
            request_id: Optional request identifier

        Returns:
            Formatted validation error response
        """
        field = validation_errors.get("field", "unknown")
        value = validation_errors.get("value", "unknown")
        message = validation_errors.get("message", "Validation failed")

        full_message = f"Validation failed for field '{field}': {message}"

        return ResponseFormatter.format_error(
            ErrorType.VALIDATION_ERROR,
            full_message,
            request_id,
            details={
                "field": field,
                "invalid_value": value,
                "validation_message": message
            }
        )

    @staticmethod
    def format_health_check() -> Dict[str, Any]:
        """
        Format a health check response.

        Returns:
            Formatted health check response
        """
        return {
            "status": "success",
            "output": {
                "system_status": "healthy",
                "service": "multi-modal-inference-worker",
                "timestamp": time.time()
            }
        }


class TestResponseFormatter:
    """Test response formatting functionality."""

    def test_format_success_response(self):
        """Test formatting of successful responses."""
        output = {"image_url": "https://example.com/result.jpg"}
        processing_time = 1500.75
        models_used = ["flux-1-schnell"]
        request_id = "test-123"

        response = ResponseFormatter.format_success(
            output, processing_time, models_used, request_id
        )

        assert response["status"] == "success"
        assert response["output"] == output
        assert response["metadata"]["request_id"] == request_id
        assert response["metadata"]["processing_time_ms"] == 1500.75
        assert response["metadata"]["models_used"] == models_used
        assert "timestamp" in response["metadata"]

    def test_format_error_response(self):
        """Test formatting of error responses."""
        error_type = ErrorType.INFERENCE_ERROR
        message = "Model inference failed"
        request_id = "test-456"
        details = {"model": "flux-1-schnell", "step": 3}

        response = ResponseFormatter.format_error(
            error_type, message, request_id, details
        )

        assert response["status"] == "error"
        assert response["error"]["type"] == error_type.value
        assert response["error"]["message"] == message
        assert response["error"]["request_id"] == request_id
        assert response["error"]["details"] == details
        assert "timestamp" in response["error"]

    def test_format_error_without_optional_params(self):
        """Test formatting error response without optional parameters."""
        error_type = ErrorType.VALIDATION_ERROR
        message = "Invalid parameters"

        response = ResponseFormatter.format_error(error_type, message)

        assert response["status"] == "error"
        assert response["error"]["type"] == error_type.value
        assert response["error"]["message"] == message
        assert "request_id" not in response["error"]
        assert "details" not in response["error"]
        assert "timestamp" in response["error"]

    def test_format_validation_error_response(self):
        """Test formatting of validation error responses."""
        validation_errors = {
            "field": "steps",
            "value": 100,
            "message": "Steps must be between 1 and 50"
        }
        request_id = "test-789"

        response = ResponseFormatter.format_validation_error(
            validation_errors, request_id
        )

        assert response["status"] == "error"
        assert response["error"]["type"] == ErrorType.VALIDATION_ERROR.value
        assert "Validation failed for field 'steps'" in response["error"]["message"]
        assert response["error"]["request_id"] == request_id
        assert response["error"]["details"]["field"] == "steps"
        assert response["error"]["details"]["invalid_value"] == 100

    def test_format_validation_error_without_request_id(self):
        """Test formatting validation error without request ID."""
        validation_errors = {
            "field": "prompt",
            "value": "",
            "message": "Prompt is required"
        }

        response = ResponseFormatter.format_validation_error(validation_errors)

        assert response["status"] == "error"
        assert response["error"]["type"] == ErrorType.VALIDATION_ERROR.value
        assert "request_id" not in response["error"]

    def test_format_health_check_response(self):
        """Test formatting of health check responses."""
        response = ResponseFormatter.format_health_check()

        assert response["status"] == "success"
        assert response["output"]["system_status"] == "healthy"
        assert response["output"]["service"] == "multi-modal-inference-worker"
        assert "timestamp" in response["output"]

    def test_processing_time_rounding(self):
        """Test that processing time is properly rounded."""
        output = {"result": "test"}
        processing_time = 1234.56789  # Many decimal places

        response = ResponseFormatter.format_success(
            output, processing_time, ["model"], "test-id"
        )

        assert response["metadata"]["processing_time_ms"] == 1234.57

    def test_error_type_values(self):
        """Test that all error types have correct string values."""
        expected_values = {
            ErrorType.VALIDATION_ERROR: "validation_error",
            ErrorType.INFERENCE_ERROR: "inference_error",
            ErrorType.MODEL_LOAD_ERROR: "model_load_error",
            ErrorType.INTERNAL_ERROR: "internal_error",
            ErrorType.UNSUPPORTED_MODALITY_ERROR: "unsupported_modality_error"
        }

        for error_type, expected_value in expected_values.items():
            assert error_type.value == expected_value

    def test_complex_output_handling(self):
        """Test formatting responses with complex output structures."""
        complex_output = {
            "images": [
                {"url": "https://example.com/img1.jpg", "size": "1024x1024"},
                {"url": "https://example.com/img2.jpg", "size": "512x512"}
            ],
            "metadata": {
                "prompt": "Complex test",
                "generation_params": {
                    "steps": 4,
                    "guidance_scale": 1.0
                }
            }
        }

        response = ResponseFormatter.format_success(
            complex_output, 2000.0, ["model1", "model2"], "complex-test"
        )

        assert response["output"] == complex_output
        assert len(response["metadata"]["models_used"]) == 2

    def test_empty_models_list(self):
        """Test handling of empty models list."""
        response = ResponseFormatter.format_success(
            {"result": "test"}, 1000.0, [], "test-id"
        )

        assert response["metadata"]["models_used"] == []

    def test_none_output_handling(self):
        """Test handling of None output."""
        response = ResponseFormatter.format_success(
            None, 500.0, ["model"], "test-id"
        )

        assert response["output"] is None
        assert response["status"] == "success"


class TestResponseFormatterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_error_message(self):
        """Test handling of very long error messages."""
        long_message = "Error: " + "x" * 10000

        response = ResponseFormatter.format_error(
            ErrorType.INTERNAL_ERROR, long_message
        )

        assert response["error"]["message"] == long_message
        assert len(response["error"]["message"]) > 10000

    def test_unicode_in_responses(self):
        """Test handling of Unicode characters in responses."""
        unicode_output = {
            "text": "Hello 世界! 🌍",
            "description": "Test with émojis 🎨 and åccénts"
        }

        response = ResponseFormatter.format_success(
            unicode_output, 1000.0, ["model"], "unicode-test"
        )

        assert response["output"]["text"] == "Hello 世界! 🌍"
        assert response["output"]["description"] == "Test with émojis 🎨 and åccénts"

    def test_very_small_processing_time(self):
        """Test handling of very small processing times."""
        response = ResponseFormatter.format_success(
            {"result": "fast"}, 0.123456, ["model"], "fast-test"
        )

        assert response["metadata"]["processing_time_ms"] == 0.12

    def test_zero_processing_time(self):
        """Test handling of zero processing time."""
        response = ResponseFormatter.format_success(
            {"result": "instant"}, 0.0, ["model"], "instant-test"
        )

        assert response["metadata"]["processing_time_ms"] == 0.0

    def test_negative_processing_time(self):
        """Test handling of negative processing time (edge case)."""
        response = ResponseFormatter.format_success(
            {"result": "negative"}, -100.0, ["model"], "negative-test"
        )

        # Should still work (implementation dependent behavior)
        assert "processing_time_ms" in response["metadata"]

    def test_validation_error_with_missing_fields(self):
        """Test validation error formatting with missing fields."""
        incomplete_validation = {}  # Missing all expected fields

        response = ResponseFormatter.format_validation_error(incomplete_validation)

        assert response["status"] == "error"
        assert "unknown" in response["error"]["message"]

    def test_large_details_object(self):
        """Test error formatting with large details object."""
        large_details = {
            f"field_{i}": f"value_{i}" for i in range(1000)
        }

        response = ResponseFormatter.format_error(
            ErrorType.INTERNAL_ERROR,
            "Large details test",
            details=large_details
        )

        assert len(response["error"]["details"]) == 1000
        assert response["error"]["details"]["field_999"] == "value_999"

    def test_nested_error_details(self):
        """Test error formatting with nested details structure."""
        nested_details = {
            "validation": {
                "errors": [
                    {"field": "prompt", "issue": "empty"},
                    {"field": "steps", "issue": "out_of_range"}
                ]
            },
            "context": {
                "user_id": "test123",
                "session": "abc456"
            }
        }

        response = ResponseFormatter.format_error(
            ErrorType.VALIDATION_ERROR,
            "Multiple validation errors",
            details=nested_details
        )

        assert response["error"]["details"]["validation"]["errors"][0]["field"] == "prompt"
        assert response["error"]["details"]["context"]["user_id"] == "test123"


class TestErrorTypes:
    """Test ErrorType enumeration."""

    def test_all_error_types_exist(self):
        """Test that all expected error types are defined."""
        expected_types = [
            "VALIDATION_ERROR",
            "INFERENCE_ERROR",
            "MODEL_LOAD_ERROR",
            "INTERNAL_ERROR",
            "UNSUPPORTED_MODALITY_ERROR"
        ]

        for error_type_name in expected_types:
            assert hasattr(ErrorType, error_type_name)

    def test_error_type_string_values(self):
        """Test that error type string values follow convention."""
        for error_type in ErrorType:
            # Should be lowercase with underscores
            assert error_type.value.islower()
            assert " " not in error_type.value
            if "_" in error_type.value:
                parts = error_type.value.split("_")
                assert all(part.isalpha() for part in parts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])