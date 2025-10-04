"""
Unit tests for MultiModalHandler class.

Tests cover request routing, handler registration, system monitoring,
and integration with validation and response formatting systems.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import uuid

# Import the classes to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from handlers.multi_modal_handler import MultiModalHandler
from handlers.base_handler import BaseHandler
from models.model_manager import ModelManager
from utils.response_formatter import ResponseFormatter, ErrorType
from utils.request_validator import RequestValidator
from utils.exceptions import ValidationError, InferenceError


class MockBaseHandler(BaseHandler):
    """Mock implementation of BaseHandler for testing."""

    def __init__(self, modality: str):
        # Call parent constructor with handler name
        super().__init__(f"{modality}-handler")
        self.modality = modality
        self.validate_called = False
        self.load_models_called = False  # Match test expectations
        self.process_inference_called = False
        self.format_response_called = False

    @property
    def supported_modality(self) -> str:
        """Return the modality type this handler supports."""
        return self.modality

    @property
    def required_parameters(self) -> List[str]:
        """Return list of required parameters for this modality."""
        return ["prompt"]

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        return {
            "width": 512,
            "height": 512,
            "steps": 20
        }

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request data."""
        self.validate_called = True
        return request_data

    def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
        """Get required models."""
        self.load_models_called = True  # Match test expectations
        return [f"{self.modality}-model"]

    def process_inference(self, models: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference."""
        self.process_inference_called = True
        return {"result": f"{self.modality}-output"}

    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format response."""
        self.format_response_called = True
        return {
            "status": "success",
            "output": inference_results,
            "metadata": {
                "modality": self.modality,
                "request_data": request_data
            }
        }

    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override handle_request to track calls properly."""
        # Track that validation was called
        validated_data = self.validate_request(request_data)

        # Track that model loading was called
        required_models = self.get_required_models(validated_data)

        # Track that inference was called (mock successful inference)
        mock_models = {model: Mock() for model in required_models}
        inference_results = self.process_inference(mock_models, validated_data)

        # Track that response formatting was called
        return self.format_response(inference_results, validated_data)


class TestMultiModalHandler:
    """Test cases for MultiModalHandler routing system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model_manager = Mock(spec=ModelManager)
        # Configure realistic manager status response
        self.mock_model_manager.get_manager_status.return_value = {
            "loaded_models": [],
            "loaded_count": 0,
            "registered_count": 2,
            "max_models": 3,
            "memory_summary": {
                "stats": {
                    "gpu_total_mb": 24000,
                    "gpu_allocated_mb": 8000,
                    "gpu_utilization_percent": 33.3
                },
                "thresholds": {
                    "warning_percent": 80.0,
                    "eviction_percent": 90.0,
                    "warning_exceeded": False,
                    "eviction_needed": False
                },
                "available_memory_mb": 16000,
                "monitoring_active": True,
                "cuda_available": True
            }
        }

        # Enable auto-initialization for main testing (edge cases use controlled testing)
        self.handler = MultiModalHandler(self.mock_model_manager, auto_initialize=True)

        # Setup mock handlers for testing
        self.mock_text_to_image_handler = MockBaseHandler("text-to-image")
        self.mock_image_to_video_handler = MockBaseHandler("image-to-video")

    def test_initialization(self):
        """Test MultiModalHandler initialization."""
        assert self.handler.model_manager == self.mock_model_manager
        assert isinstance(self.handler.request_validator, RequestValidator)
        assert isinstance(self.handler.response_formatter, ResponseFormatter)

        # With auto_initialize=True, should start with auto-initialized handlers
        assert len(self.handler.handlers) == 3
        assert 'image-to-video' in self.handler.handlers

        # Supported modalities should match registered handlers
        assert set(self.handler.supported_modalities) == {'text-to-image', 'controlnet', 'image-to-video'}

        assert self.handler.request_count == 0
        assert self.handler.total_processing_time == 0.0

    def test_register_handler(self):
        """Test handler registration functionality."""
        # Register a handler
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        assert "text-to-image" in self.handler.handlers
        assert self.handler.handlers["text-to-image"] == self.mock_text_to_image_handler
        assert "text-to-image" in self.handler.supported_modalities

    def test_register_multiple_handlers(self):
        """Test registration of multiple handlers."""
        # Register additional handlers (text-to-image and image-to-video will replace auto-initialized ones)
        initial_count = len(self.handler.handlers)  # Should be 3 (auto-initialized)

        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)
        self.handler.register_handler("image-to-video", self.mock_image_to_video_handler)

        # Handler count should remain the same (replaced existing handlers)
        assert len(self.handler.handlers) == initial_count
        assert len(self.handler.supported_modalities) == initial_count
        assert "text-to-image" in self.handler.supported_modalities
        assert "image-to-video" in self.handler.supported_modalities

    def test_get_supported_modalities(self):
        """Test getting list of supported modalities."""
        # Should have auto-initialized handlers
        modalities = self.handler.get_supported_modalities()
        assert len(modalities) == 3  # text-to-image, controlnet, image-to-video
        assert set(modalities) == {'text-to-image', 'controlnet', 'image-to-video'}

        # After registering additional mock handlers
        self.handler.register_handler("test-modality", self.mock_text_to_image_handler)

        modalities = self.handler.get_supported_modalities()
        assert len(modalities) == 4
        assert "test-modality" in modalities
        assert "image-to-video" in modalities

        # Should return a copy, not the original list
        modalities.append("extra-test")
        assert len(self.handler.get_supported_modalities()) == 4  # Still 4 - doesn't modify original

    @patch('uuid.uuid4')
    def test_process_request_with_explicit_modality(self, mock_uuid):
        """Test processing request with explicit modality specification."""
        mock_uuid.return_value = "test-request-123"

        # Register handler
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        # Mock validation to pass
        with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
            request_data = {
                "modality": "text-to-image",
                "prompt": "A beautiful sunset",
                "steps": 4
            }

            response = self.handler.process_request(request_data)

            # Verify handler was called
            assert self.mock_text_to_image_handler.validate_called
            assert self.mock_text_to_image_handler.load_models_called
            assert self.mock_text_to_image_handler.process_inference_called
            assert self.mock_text_to_image_handler.format_response_called

            # Verify request count updated
            assert self.handler.request_count == 1

    @patch('uuid.uuid4')
    def test_process_request_with_auto_detection(self, mock_uuid):
        """Test processing request with automatic modality detection."""
        mock_uuid.return_value = "test-request-456"

        # Register handler
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        # Mock modality detection and validation
        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
                request_data = {
                    "prompt": "A beautiful sunset",
                    "steps": 4,
                    "guidance_scale": 1.0
                }

                response = self.handler.process_request(request_data)

                # Verify handler was called
                assert self.mock_text_to_image_handler.validate_called

    @patch('uuid.uuid4')
    def test_process_request_modality_detection_failure(self, mock_uuid):
        """Test handling when modality detection fails."""
        mock_uuid.return_value = "test-request-789"

        # Mock modality detection to fail
        with patch.object(self.handler.modality_detector, 'detect_modality', return_value=None):
            request_data = {
                "unknown_param": "value"
            }

            response = self.handler.process_request(request_data)

            # Should return error response
            assert response['status'] == 'error'
            assert 'error' in response
            assert 'Could not determine request modality' in response['error']['message']

    @patch('uuid.uuid4')
    def test_process_request_unsupported_modality(self, mock_uuid):
        """Test handling of requests for unsupported modalities."""
        mock_uuid.return_value = "test-request-unsupported"

        # Don't register any handlers, but mock detection to return unsupported modality
        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="unsupported-modality"):
            request_data = {
                "modality": "unsupported-modality",
                "param": "value"
            }

            response = self.handler.process_request(request_data)

            # Should return modality not supported error
            assert response['status'] == 'error'
            assert 'Unsupported modality' in response['error']['message']

    @patch('uuid.uuid4')
    def test_process_request_validation_failure(self, mock_uuid):
        """Test handling of validation failures."""
        mock_uuid.return_value = "test-request-validation-fail"

        # Register handler
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        # Mock validation to fail
        validation_error = {
            'field': 'steps',
            'value': 100,
            'message': 'Value must be between 1 and 50'
        }

        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=validation_error):
                request_data = {
                    "prompt": "Test",
                    "steps": 100
                }

                response = self.handler.process_request(request_data)

                # Should return validation error
                assert response['status'] == 'error'
                assert 'Validation failed' in response['error']['message']
                assert response['error']['type'] == ErrorType.VALIDATION_ERROR.value

    @patch('uuid.uuid4')
    def test_process_request_handler_exception(self, mock_uuid):
        """Test handling of exceptions raised by handlers."""
        mock_uuid.return_value = "test-request-exception"

        # Create handler that raises exception
        failing_handler = MockBaseHandler("text-to-image")
        failing_handler.process_inference = Mock(side_effect=Exception("Handler failed"))

        self.handler.register_handler("text-to-image", failing_handler)

        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
                request_data = {
                    "prompt": "Test",
                    "steps": 4
                }

                response = self.handler.process_request(request_data)

                # Should return error response
                assert response['status'] == 'error'
                assert 'Processing failed' in response['error']['message']
                assert response['error']['type'] == ErrorType.INFERENCE_ERROR.value

    @patch('uuid.uuid4')
    def test_process_request_unexpected_exception(self, mock_uuid):
        """Test handling of unexpected exceptions during processing."""
        mock_uuid.return_value = "test-request-unexpected"

        # Mock modality detection to raise exception
        with patch.object(self.handler.modality_detector, 'detect_modality', side_effect=Exception("Unexpected error")):
            request_data = {
                "prompt": "Test"
            }

            response = self.handler.process_request(request_data)

            # Should return modality detection error (exception is caught in _detect_modality)
            assert response['status'] == 'error'
            assert 'Could not determine request modality' in response['error']['message']
            assert response['error']['type'] == ErrorType.VALIDATION_ERROR.value

    def test_get_system_status(self):
        """Test system status reporting."""
        # Register some handlers
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)
        self.handler.register_handler("image-to-video", self.mock_image_to_video_handler)

        # Simulate some processing history
        self.handler.request_count = 10
        self.handler.total_processing_time = 15000.0  # 15 seconds total

        # Update mock model manager to return 2 loaded models
        self.mock_model_manager.get_manager_status.return_value.update({
            "loaded_count": 2,
            "memory_summary": {
                "stats": {
                    "gpu_total_mb": 24000,
                    "gpu_free_mb": 16000,
                    "gpu_allocated_mb": 8000
                }
            }
        })

        status = self.handler.get_system_status()

        # Validate status structure
        assert status['service'] == 'multi-modal-inference-worker'
        assert status['status'] == 'healthy'
        assert set(status['supported_modalities']) == {"text-to-image", "controlnet", "image-to-video"}

        # Validate statistics
        stats = status['statistics']
        assert stats['total_requests'] == 10
        assert stats['average_processing_time_ms'] == 1500.0  # 15000 / 10

        # Validate system info
        system_info = status['system']
        assert 'memory' in system_info
        assert 'models' in system_info
        assert system_info['models']['loaded_models'] == 2

    def test_health_check(self):
        """Test health check functionality."""
        with patch('time.time', return_value=1234567890):
            response = self.handler.health_check()

            assert response['status'] == 'success'
            assert 'output' in response
            output = response['output']
            assert output['system_status'] == 'healthy'
            assert 'supported_modalities' in output
            assert 'system_stats' in output

    def test_performance_tracking(self):
        """Test that performance metrics are tracked correctly."""
        initial_count = self.handler.request_count
        initial_time = self.handler.total_processing_time

        # Register handler
        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        # Process a request
        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
                with patch('time.time', side_effect=[0, 1.5]):  # 1.5 second processing time
                    request_data = {"prompt": "Test"}
                    self.handler.process_request(request_data)

        # Verify metrics updated
        assert self.handler.request_count == initial_count + 1
        assert self.handler.total_processing_time > initial_time

    def test_concurrent_request_independence(self):
        """Test that concurrent requests are handled independently."""
        # This test verifies that request processing doesn't interfere with each other
        # In practice, this would require threading, but we can test the structure

        self.handler.register_handler("text-to-image", self.mock_text_to_image_handler)

        # Process multiple requests
        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
                request1 = {"prompt": "Request 1"}
                request2 = {"prompt": "Request 2"}

                response1 = self.handler.process_request(request1)
                response2 = self.handler.process_request(request2)

                # Both should succeed independently
                assert response1['status'] == 'success'
                assert response2['status'] == 'success'
                assert self.handler.request_count == 2


class TestMultiModalHandlerEdgeCases:
    """Test edge cases and boundary conditions for MultiModalHandler."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model_manager = Mock(spec=ModelManager)

        # Configure the mock to return proper numeric values for system status
        self.mock_model_manager.get_manager_status.return_value = {
            'loaded_count': 0,
            'memory_summary': {
                'stats': {
                    'gpu_free_mb': 8192,
                    'gpu_total_mb': 16384
                }
            }
        }

        # Disable auto-initialization for testing to prevent interference
        self.handler = MultiModalHandler(self.mock_model_manager, auto_initialize=False)

    def test_empty_request_data(self):
        """Test handling of empty request data."""
        response = self.handler.process_request({})

        assert response['status'] == 'error'
        assert 'Could not determine request modality' in response['error']['message']

    def test_none_request_data(self):
        """Test handling of None request data."""
        # This should be handled gracefully
        with pytest.raises(Exception):
            self.handler.process_request(None)

    def test_very_large_request_data(self):
        """Test handling of very large request data."""
        large_request = {
            "prompt": "x" * 10000,  # Very long prompt
            "modality": "text-to-image",
            "large_data": list(range(1000))  # Large data structure
        }

        # Should still process (though may fail validation)
        response = self.handler.process_request(large_request)
        assert 'status' in response

    def test_system_status_with_no_requests(self):
        """Test system status when no requests have been processed."""
        status = self.handler.get_system_status()

        stats = status['statistics']
        assert stats['total_requests'] == 0
        assert stats['average_processing_time_ms'] == 0.0

    def test_overwrite_handler_registration(self):
        """Test behavior when overwriting an existing handler registration."""
        handler1 = MockBaseHandler("text-to-image")
        handler2 = MockBaseHandler("text-to-image")

        # Register first handler
        self.handler.register_handler("text-to-image", handler1)
        assert self.handler.handlers["text-to-image"] == handler1

        # Register second handler (should overwrite)
        self.handler.register_handler("text-to-image", handler2)
        assert self.handler.handlers["text-to-image"] == handler2

        # Should still have only one modality in supported list
        assert len(self.handler.supported_modalities) == 1

    @patch('uuid.uuid4')
    def test_request_id_generation(self, mock_uuid):
        """Test that unique request IDs are generated."""
        mock_uuid.side_effect = ["request-1", "request-2", "request-3"]

        self.handler.register_handler("text-to-image", MockBaseHandler("text-to-image"))

        with patch.object(self.handler.modality_detector, 'detect_modality', return_value="text-to-image"):
            with patch.object(self.handler.request_validator, 'validate_full_request', return_value=None):
                # Process multiple requests
                self.handler.process_request({"prompt": "Test 1"})
                self.handler.process_request({"prompt": "Test 2"})
                self.handler.process_request({"prompt": "Test 3"})

                # Verify UUID was called for each request
                assert mock_uuid.call_count == 3


class TestMultiModalHandlerIntegration:
    """Integration tests for MultiModalHandler with real components."""

    def setup_method(self):
        """Setup test fixtures with real components."""
        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.get_manager_status.return_value = {
            "loaded_models": [],
            "loaded_count": 0,
            "registered_count": 3,
            "max_models": 5,
            "memory_summary": {
                "stats": {
                    "gpu_free_mb": 16000,
                    "gpu_total_mb": 24000,
                    "gpu_utilization": 0.0
                },
                "thresholds": {
                    "warning_percent": 80,
                    "eviction_percent": 90,
                    "warning_exceeded": False,
                    "eviction_needed": False
                },
                "available_memory_mb": 16000
            },
            "statistics": {},
            "configuration": {
                "max_models": 5,
                "model_timeout_seconds": 300,
                "protect_duration_minutes": 5
            }
        }
        self.handler = MultiModalHandler(self.mock_model_manager)

    def test_full_request_flow_with_real_validator(self):
        """Test complete request flow with actual RequestValidator."""
        # Register mock handler
        mock_handler = MockBaseHandler("text-to-image")
        self.handler.register_handler("text-to-image", mock_handler)

        # Process request with valid text-to-image parameters
        request_data = {
            "prompt": "A beautiful sunset over mountains",
            "steps": 4,
            "guidance_scale": 1.0
        }

        response = self.handler.process_request(request_data)

        # Should successfully process
        assert response['status'] == 'success'
        assert mock_handler.validate_called
        assert mock_handler.process_inference_called

    def test_full_request_flow_with_validation_error(self):
        """Test complete request flow with validation error."""
        mock_handler = MockBaseHandler("text-to-image")
        self.handler.register_handler("text-to-image", mock_handler)

        # Process request with invalid parameters
        request_data = {
            "prompt": "",  # Empty prompt should fail validation
            "steps": 100,  # Out of range
            "guidance_scale": 1.0
        }

        response = self.handler.process_request(request_data)

        # Should fail validation
        assert response['status'] == 'error'
        assert response['error']['type'] == ErrorType.VALIDATION_ERROR.value


if __name__ == '__main__':
    pytest.main([__file__])