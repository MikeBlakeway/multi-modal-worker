"""
Unit tests for ResponseFormatter class.

Tests cover response formatting, error handling, RunPod compatibility,
and various response scenarios for the multi-modal inference worker.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import the classes to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.response_formatter import ResponseFormatter, ResponseStatus, ErrorType


class TestResponseFormatter:
    """Test cases for ResponseFormatter utility class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.formatter = ResponseFormatter()
        self.test_request_id = "test-request-123"
        self.test_modality = "text-to-image"
        self.test_models = ["flux-schnell", "clip-vit"]

    def test_format_success_response_basic(self):
        """Test formatting of basic success response."""
        output_data = {
            "image_urls": ["https://example.com/image1.png", "https://example.com/image2.png"],
            "generation_info": {"seed": 42, "steps": 4}
        }
        processing_time = 1500.0

        response = self.formatter.format_success_response(
            output_data=output_data,
            request_id=self.test_request_id,
            modality=self.test_modality,
            processing_time_ms=processing_time,
            models_used=self.test_models
        )

        # Validate response structure
        assert response['status'] == ResponseStatus.SUCCESS.value
        assert response['output'] == output_data
        assert 'metadata' in response

        # Validate metadata
        metadata = response['metadata']
        assert metadata['request_id'] == self.test_request_id
        assert metadata['modality'] == self.test_modality
        assert metadata['processing_time_ms'] == 1500.0
        assert metadata['models_used'] == self.test_models
        assert 'timestamp' in metadata
        assert metadata['status'] == ResponseStatus.SUCCESS.value

    def test_format_success_response_with_additional_metadata(self):
        """Test success response formatting with additional metadata."""
        output_data = {"result": "success"}
        additional_metadata = {
            "gpu_memory_used": 8.5,
            "inference_backend": "diffusers",
            "model_precision": "fp16"
        }

        response = self.formatter.format_success_response(
            output_data=output_data,
            request_id=self.test_request_id,
            modality=self.test_modality,
            processing_time_ms=2000.0,
            models_used=self.test_models,
            additional_metadata=additional_metadata
        )

        metadata = response['metadata']
        assert metadata['gpu_memory_used'] == 8.5
        assert metadata['inference_backend'] == "diffusers"
        assert metadata['model_precision'] == "fp16"

    def test_format_error_response_basic(self):
        """Test formatting of basic error response."""
        error_message = "Invalid input parameters"
        error_type = ErrorType.VALIDATION_ERROR

        response = self.formatter.format_error_response(
            error_message=error_message,
            error_type=error_type,
            request_id=self.test_request_id,
            modality=self.test_modality
        )

        # Validate response structure
        assert response['status'] == ResponseStatus.ERROR.value
        assert 'error' in response

        # Validate error information
        error_info = response['error']
        assert error_info['message'] == error_message
        assert error_info['type'] == error_type.value
        assert error_info['request_id'] == self.test_request_id
        assert error_info['modality'] == self.test_modality
        assert 'timestamp' in error_info

    def test_format_error_response_with_details_and_suggestions(self):
        """Test error response formatting with details and suggestions."""
        error_message = "Model loading failed"
        error_type = ErrorType.MODEL_ERROR
        details = {
            "model_name": "flux-schnell",
            "error_code": "CUDA_OUT_OF_MEMORY",
            "available_vram": "6GB"
        }
        suggestions = [
            "Try reducing batch size",
            "Use a smaller model variant",
            "Clear GPU cache and retry"
        ]

        response = self.formatter.format_error_response(
            error_message=error_message,
            error_type=error_type,
            request_id=self.test_request_id,
            modality=self.test_modality,
            details=details,
            suggestions=suggestions
        )

        error_info = response['error']
        assert error_info['details'] == details
        assert error_info['suggestions'] == suggestions

    def test_format_validation_error(self):
        """Test formatting of validation-specific error."""
        field_name = "steps"
        field_value = 100
        validation_message = "Value must be between 1 and 50"

        response = self.formatter.format_validation_error(
            field_name=field_name,
            field_value=field_value,
            validation_message=validation_message,
            request_id=self.test_request_id
        )

        # Validate response structure
        assert response['status'] == ResponseStatus.ERROR.value
        error_info = response['error']
        assert error_info['type'] == ErrorType.VALIDATION_ERROR.value
        assert field_name in error_info['message']
        assert validation_message in error_info['message']

        # Validate details
        details = error_info['details']
        assert details['field'] == field_name
        assert details['provided_value'] == str(field_value)
        assert details['validation_rule'] == validation_message

        # Validate suggestions are present
        assert 'suggestions' in error_info
        assert len(error_info['suggestions']) > 0

    def test_format_modality_not_supported_error(self):
        """Test formatting of unsupported modality error."""
        requested_modality = "unsupported-modality"
        supported_modalities = ["text-to-image", "image-to-video", "text-to-video"]

        response = self.formatter.format_modality_not_supported_error(
            requested_modality=requested_modality,
            supported_modalities=supported_modalities,
            request_id=self.test_request_id
        )

        error_info = response['error']
        assert error_info['type'] == ErrorType.VALIDATION_ERROR.value
        assert requested_modality in error_info['message']

        # Validate details
        details = error_info['details']
        assert details['requested_modality'] == requested_modality
        assert details['supported_modalities'] == supported_modalities

        # Validate suggestions contain supported modalities
        suggestions_text = ' '.join(error_info['suggestions'])
        for modality in supported_modalities:
            assert modality in suggestions_text

    def test_format_model_loading_error(self):
        """Test formatting of model loading error."""
        model_name = "flux-schnell"
        error_details = "CUDA out of memory: tried to allocate 2.0GB"

        response = self.formatter.format_model_loading_error(
            model_name=model_name,
            error_details=error_details,
            request_id=self.test_request_id,
            modality=self.test_modality
        )

        error_info = response['error']
        assert error_info['type'] == ErrorType.MODEL_ERROR.value
        assert model_name in error_info['message']
        assert error_details in error_info['message']
        assert error_info['modality'] == self.test_modality

        # Validate details
        details = error_info['details']
        assert details['model_name'] == model_name
        assert details['loading_error'] == error_details
        assert details['modality'] == self.test_modality

    def test_format_inference_error(self):
        """Test formatting of inference processing error."""
        error_details = "Inference timeout after 30 seconds"
        models_attempted = ["flux-schnell", "clip-vit"]

        response = self.formatter.format_inference_error(
            error_details=error_details,
            request_id=self.test_request_id,
            modality=self.test_modality,
            models_attempted=models_attempted
        )

        error_info = response['error']
        assert error_info['type'] == ErrorType.INFERENCE_ERROR.value
        assert error_details in error_info['message']
        assert error_info['modality'] == self.test_modality

        # Validate details
        details = error_info['details']
        assert details['inference_error'] == error_details
        assert details['models_used'] == models_attempted

    def test_add_runpod_compatibility_success(self):
        """Test RunPod compatibility formatting for success responses."""
        original_response = {
            'status': ResponseStatus.SUCCESS.value,
            'output': {'result': 'success'},
            'metadata': {'processing_time': 1000}
        }

        runpod_response = self.formatter.add_runpod_compatibility(original_response)

        # RunPod format should have specific structure
        assert 'output' in runpod_response
        assert runpod_response['output'] == original_response['output']
        assert runpod_response['metadata'] == original_response['metadata']
        assert runpod_response['status'] == 'success'

    def test_add_runpod_compatibility_error(self):
        """Test RunPod compatibility formatting for error responses."""
        original_response = {
            'status': ResponseStatus.ERROR.value,
            'error': {
                'message': 'Test error',
                'type': ErrorType.VALIDATION_ERROR.value
            }
        }

        runpod_response = self.formatter.add_runpod_compatibility(original_response)

        # RunPod error format
        assert 'error' in runpod_response
        assert runpod_response['error'] == original_response['error']
        assert runpod_response['status'] == 'error'

    def test_format_system_status_response(self):
        """Test formatting of system status response."""
        system_stats = {
            'gpu_memory_total': 24.0,
            'gpu_memory_used': 8.5,
            'loaded_models': 3,
            'uptime_seconds': 3600
        }
        supported_modalities = ["text-to-image", "image-to-video"]

        response = self.formatter.format_system_status_response(
            system_stats=system_stats,
            supported_modalities=supported_modalities,
            request_id=self.test_request_id
        )

        # Validate response structure
        assert response['status'] == ResponseStatus.SUCCESS.value
        output = response['output']
        assert output['system_status'] == 'healthy'
        assert output['supported_modalities'] == supported_modalities
        assert output['system_stats'] == system_stats

        # Validate capabilities
        capabilities = output['capabilities']
        assert 'concurrent_requests' in capabilities
        assert 'model_management' in capabilities
        assert 'automatic_eviction' in capabilities

    def test_processing_time_rounding(self):
        """Test that processing times are properly rounded."""
        output_data = {"result": "test"}
        processing_time = 1234.56789  # High precision

        response = self.formatter.format_success_response(
            output_data=output_data,
            request_id=self.test_request_id,
            modality=self.test_modality,
            processing_time_ms=processing_time,
            models_used=self.test_models
        )

        # Should be rounded to 2 decimal places
        assert response['metadata']['processing_time_ms'] == 1234.57

    def test_timestamp_generation(self):
        """Test that timestamps are generated correctly."""
        before_time = time.time()

        response = self.formatter.format_success_response(
            output_data={"result": "test"},
            request_id=self.test_request_id,
            modality=self.test_modality,
            processing_time_ms=1000.0,
            models_used=self.test_models
        )

        after_time = time.time()
        timestamp = response['metadata']['timestamp']

        # Timestamp should be between before and after
        assert before_time <= timestamp <= after_time

    def test_error_response_without_optional_fields(self):
        """Test error response formatting with minimal required fields."""
        response = self.formatter.format_error_response(
            error_message="Simple error",
            error_type=ErrorType.INTERNAL_ERROR,
            request_id=self.test_request_id
        )

        error_info = response['error']
        assert error_info['message'] == "Simple error"
        assert error_info['type'] == ErrorType.INTERNAL_ERROR.value
        assert error_info['request_id'] == self.test_request_id
        assert 'modality' not in error_info  # Should not be present when not provided
        assert 'details' not in error_info
        assert 'suggestions' not in error_info


class TestResponseFormatterEdgeCases:
    """Test edge cases and boundary conditions for ResponseFormatter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.formatter = ResponseFormatter()

    def test_empty_output_data(self):
        """Test handling of empty output data."""
        response = self.formatter.format_success_response(
            output_data={},  # Empty output
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=1000.0,
            models_used=[]
        )

        assert response['output'] == {}
        assert response['status'] == ResponseStatus.SUCCESS.value

    def test_none_output_data(self):
        """Test handling of None output data."""
        response = self.formatter.format_success_response(
            output_data=None,
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=1000.0,
            models_used=[]
        )

        assert response['output'] is None
        assert response['status'] == ResponseStatus.SUCCESS.value

    def test_very_long_error_message(self):
        """Test handling of very long error messages."""
        long_message = "x" * 10000  # Very long message

        response = self.formatter.format_error_response(
            error_message=long_message,
            error_type=ErrorType.INTERNAL_ERROR,
            request_id="test-123"
        )

        assert response['error']['message'] == long_message
        assert len(response['error']['message']) == 10000

    def test_unicode_in_error_messages(self):
        """Test handling of Unicode characters in error messages."""
        unicode_message = "Error with émojis 🚫 and 中文 characters"

        response = self.formatter.format_error_response(
            error_message=unicode_message,
            error_type=ErrorType.VALIDATION_ERROR,
            request_id="test-123"
        )

        assert response['error']['message'] == unicode_message

    def test_nested_complex_output_data(self):
        """Test handling of complex nested output data."""
        complex_output = {
            "images": [
                {
                    "url": "https://example.com/image1.png",
                    "metadata": {"width": 1024, "height": 1024, "format": "PNG"},
                    "generation_params": {"seed": 42, "steps": 4}
                }
            ],
            "processing_info": {
                "model_versions": {"diffusion": "1.0", "vae": "2.0"},
                "performance_metrics": {"inference_time": 2.5, "memory_peak": 8.2}
            }
        }

        response = self.formatter.format_success_response(
            output_data=complex_output,
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=2500.0,
            models_used=["model1", "model2"]
        )

        assert response['output'] == complex_output

    def test_zero_processing_time(self):
        """Test handling of zero processing time."""
        response = self.formatter.format_success_response(
            output_data={"result": "instant"},
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=0.0,
            models_used=[]
        )

        assert response['metadata']['processing_time_ms'] == 0.0

    def test_negative_processing_time(self):
        """Test handling of negative processing time (edge case)."""
        response = self.formatter.format_success_response(
            output_data={"result": "test"},
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=-100.0,  # Negative time (shouldn't happen in practice)
            models_used=[]
        )

        # Should still format the response even with negative time
        assert response['metadata']['processing_time_ms'] == -100.0


class TestErrorTypes:
    """Test cases for different error types and their formatting."""

    def setup_method(self):
        """Setup test fixtures."""
        self.formatter = ResponseFormatter()

    def test_all_error_types_formatting(self):
        """Test that all error types format correctly."""
        error_types = [
            ErrorType.VALIDATION_ERROR,
            ErrorType.MODEL_ERROR,
            ErrorType.INFERENCE_ERROR,
            ErrorType.RESOURCE_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.INTERNAL_ERROR
        ]

        for error_type in error_types:
            response = self.formatter.format_error_response(
                error_message=f"Test {error_type.value} message",
                error_type=error_type,
                request_id="test-123"
            )

            assert response['error']['type'] == error_type.value
            assert response['status'] == ResponseStatus.ERROR.value

    def test_response_status_enum_values(self):
        """Test that ResponseStatus enum values are used correctly."""
        # Test success status
        success_response = self.formatter.format_success_response(
            output_data={"result": "success"},
            request_id="test-123",
            modality="text-to-image",
            processing_time_ms=1000.0,
            models_used=[]
        )
        assert success_response['status'] == ResponseStatus.SUCCESS.value

        # Test error status
        error_response = self.formatter.format_error_response(
            error_message="Test error",
            error_type=ErrorType.VALIDATION_ERROR,
            request_id="test-123"
        )
        assert error_response['status'] == ResponseStatus.ERROR.value


if __name__ == '__main__':
    pytest.main([__file__])