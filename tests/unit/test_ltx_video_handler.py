"""
Unit tests for LTX-Video text-to-video handler.

Tests the LTX-Video handler's parameter validation, request processing,
response formatting, and integration with model management system.
Following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import base64
from datetime import datetime
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.handlers.ltx_video_handler import LTXVideoHandler
from src.schemas.text_to_video_schema import TextToVideoRequest, TextToVideoResponse, VideoInfo
from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
from src.models.ltx_video_model import LTXVideoModel


class TestLTXVideoHandler(unittest.TestCase):
    """Test cases for LTXVideoHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = LTXVideoHandler()

        # Valid test request with dimensions divisible by 32
        self.valid_request = {
            'prompt': 'A majestic eagle soaring through mountain peaks at sunset, slow motion cinematic shot',
            'width': 704,   # 704 is divisible by 32 (22 * 32)
            'height': 1280, # 1280 is divisible by 32 (40 * 32)
            'num_frames': 25,  # (8*3)+1 for LTX-Video
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'fps': 8
        }

        # Mock video response
        self.mock_video_b64 = "mock_base64_video_data"

    def test_handler_initialization(self):
        """Test handler initialization."""
        self.assertEqual(self.handler.HANDLER_NAME, "ltx-video-text-to-video")
        self.assertEqual(self.handler.SUPPORTED_MODALITY, "text-to-video")
        self.assertIn('prompt', self.handler.required_parameters)
        self.assertIsInstance(self.handler.model, LTXVideoModel)

    def test_validate_request_valid(self):
        """Test request validation with valid parameters."""
        result = self.handler.validate_request(self.valid_request)

        self.assertEqual(result['prompt'], self.valid_request['prompt'])
        self.assertEqual(result['width'], 704)
        self.assertEqual(result['height'], 1280)
        self.assertEqual(result['num_frames'], 25)

    def test_validate_request_missing_required_param(self):
        """Test request validation with missing required parameter."""
        invalid_request = self.valid_request.copy()
        del invalid_request['prompt']

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_empty_prompt(self):
        """Test request validation with empty prompt."""
        invalid_request = self.valid_request.copy()
        invalid_request['prompt'] = ""

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_dimensions(self):
        """Test request validation with invalid dimensions (not divisible by 32)."""
        invalid_request = self.valid_request.copy()
        invalid_request['width'] = 511  # Not divisible by 32

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_frame_count(self):
        """Test request validation with invalid frame count (not 8*n+1)."""
        invalid_request = self.valid_request.copy()
        invalid_request['num_frames'] = 26  # Not (8*n)+1 pattern

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_default_values(self):
        """Test request validation applies default values correctly."""
        minimal_request = {'prompt': 'Test video generation'}
        result = self.handler.validate_request(minimal_request)

        self.assertEqual(result['width'], 704)
        self.assertEqual(result['height'], 1280)
        self.assertEqual(result['num_frames'], 25)
        self.assertEqual(result['num_inference_steps'], 20)

    @patch('handlers.ltx_video_handler.LTXVideoModel')
    def test_process_request_success(self, mock_model_class):
        """Test successful request processing."""
        # Mock model instance
        mock_model = Mock()
        mock_model.generate_video.return_value = self.mock_video_b64
        mock_model.is_loaded = True
        mock_model_class.return_value = mock_model

        # Create handler with mocked model
        handler = LTXVideoHandler()
        handler.model = mock_model

        result = handler.process_request(self.valid_request)

        # Verify model method was called correctly
        mock_model.generate_video.assert_called_once()
        call_args = mock_model.generate_video.call_args[1]
        self.assertEqual(call_args['prompt'], self.valid_request['prompt'])
        self.assertEqual(call_args['width'], self.valid_request['width'])
        self.assertEqual(call_args['height'], self.valid_request['height'])

        # Verify response format
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertIn('data', result)
        self.assertIn('video_data', result['data'])

    @patch('handlers.ltx_video_handler.LTXVideoModel')
    def test_process_request_model_not_loaded(self, mock_model_class):
        """Test request processing when model is not loaded."""
        mock_model = Mock()
        mock_model.is_loaded = False
        mock_model.load_model.side_effect = ModelLoadError("Model loading failed")
        mock_model_class.return_value = mock_model

        handler = LTXVideoHandler()
        handler.model = mock_model

        with self.assertRaises(ModelLoadError):
            handler.process_request(self.valid_request)

    @patch('handlers.ltx_video_handler.LTXVideoModel')
    def test_process_request_inference_error(self, mock_model_class):
        """Test request processing with inference error."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.generate_video.side_effect = InferenceError("Video generation failed")
        mock_model_class.return_value = mock_model

        handler = LTXVideoHandler()
        handler.model = mock_model

        with self.assertRaises(InferenceError):
            handler.process_request(self.valid_request)

    def test_format_response_success(self):
        """Test successful response formatting."""
        video_data = self.mock_video_b64

        response = self.handler.format_response(
            success=True,
            video_data=video_data,
            request_params=self.valid_request,
            processing_time=2.5
        )

        self.assertTrue(response['success'])
        self.assertIn('data', response)
        self.assertEqual(response['data']['video_data'], video_data)
        self.assertIn('video_info', response['data'])
        self.assertEqual(response['data']['video_info']['width'], 720)
        self.assertEqual(response['data']['video_info']['height'], 1280)
        self.assertEqual(response['data']['video_info']['num_frames'], 25)

    def test_format_response_error(self):
        """Test error response formatting."""
        error_message = "Video generation failed"

        response = self.handler.format_response(
            success=False,
            error_message=error_message,
            request_params=self.valid_request
        )

        self.assertFalse(response['success'])
        self.assertIn('error', response)
        self.assertEqual(response['error']['message'], error_message)
        self.assertIn('timestamp', response['error'])

    def test_get_capabilities(self):
        """Test handler capabilities reporting."""
        capabilities = self.handler.get_capabilities()

        self.assertIn('modality', capabilities)
        self.assertEqual(capabilities['modality'], 'text-to-video')
        self.assertIn('max_duration', capabilities)
        self.assertIn('resolution_support', capabilities)
        self.assertIn('frame_rates', capabilities)

    def test_performance_metrics_tracking(self):
        """Test performance metrics are tracked correctly."""
        initial_count = self.handler.request_count
        initial_time = self.handler.total_processing_time

        # Mock successful processing
        with patch.object(self.handler.model, 'generate_video', return_value=self.mock_video_b64):
            with patch.object(self.handler.model, 'is_loaded', True):
                self.handler.process_request(self.valid_request)

        self.assertEqual(self.handler.request_count, initial_count + 1)
        self.assertGreater(self.handler.total_processing_time, initial_time)
        self.assertEqual(self.handler.successful_requests, 1)

    def test_prompt_preprocessing(self):
        """Test prompt preprocessing for optimal LTX-Video results."""
        short_prompt = "eagle flying"
        processed = self.handler.preprocess_prompt(short_prompt)

        # Should enhance short prompts with detailed descriptions
        self.assertGreater(len(processed), len(short_prompt))
        self.assertIn("eagle", processed.lower())

    def test_parameter_optimization(self):
        """Test parameter optimization for LTX-Video constraints."""
        params = {
            'prompt': 'Test video',
            'width': 700,  # Not optimal, should be adjusted
            'height': 1200,  # Not optimal, should be adjusted
            'num_frames': 24  # Invalid, should be corrected
        }

        optimized = self.handler.optimize_parameters(params)

        self.assertEqual(optimized['width'], 704)  # Closest to 700, divisible by 32
        self.assertEqual(optimized['height'], 1216)  # Closest to 1200, divisible by 32
        self.assertEqual(optimized['num_frames'], 25)  # Corrected to (8*3)+1

    def test_memory_management_integration(self):
        """Test integration with memory management system."""
        # Mock memory monitor
        with patch('handlers.ltx_video_handler.MemoryMonitor') as mock_monitor:
            mock_instance = Mock()
            mock_monitor.return_value = mock_instance

            handler = LTXVideoHandler()

            # Verify memory monitoring is integrated
            self.assertIsNotNone(handler.memory_monitor)

    def test_error_handling_chain(self):
        """Test comprehensive error handling throughout processing chain."""
        test_cases = [
            (ValidationError("Invalid parameters"), ValidationError),
            (ModelLoadError("Model loading failed"), ModelLoadError),
            (InferenceError("Generation failed"), InferenceError),
            (Exception("Unexpected error"), Exception)
        ]

        for exception, expected_type in test_cases:
            with patch.object(self.handler.model, 'generate_video', side_effect=exception):
                with patch.object(self.handler.model, 'is_loaded', True):
                    with self.assertRaises(expected_type):
                        self.handler.process_request(self.valid_request)

    def test_modality_detection_integration(self):
        """Test integration with modality detection system."""
        # Test parameters that should be detected as text-to-video
        text_to_video_params = {'prompt': 'Generate video of dancing'}

        is_supported = self.handler.supports_request(text_to_video_params)
        self.assertTrue(is_supported)

        # Test parameters that should NOT be detected as text-to-video
        image_to_video_params = {'image': 'base64_image', 'prompt': 'Add motion'}
        is_supported = self.handler.supports_request(image_to_video_params)
        self.assertFalse(is_supported)

    @patch('handlers.ltx_video_handler.datetime')
    def test_response_metadata(self, mock_datetime):
        """Test response includes proper metadata."""
        mock_now = datetime(2025, 10, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = mock_now

        response = self.handler.format_response(
            success=True,
            video_data=self.mock_video_b64,
            request_params=self.valid_request,
            processing_time=3.2
        )

        metadata = response['data']['metadata']
        self.assertEqual(metadata['model_type'], 'ltx-video-2b')
        self.assertEqual(metadata['processing_time'], 3.2)
        self.assertEqual(metadata['handler_version'], '1.0.0')

    def test_resource_cleanup(self):
        """Test proper resource cleanup after processing."""
        with patch.object(self.handler.model, 'cleanup_resources') as mock_cleanup:
            with patch.object(self.handler.model, 'generate_video', return_value=self.mock_video_b64):
                with patch.object(self.handler.model, 'is_loaded', True):
                    self.handler.process_request(self.valid_request)

                    # Verify cleanup was called
                    mock_cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()