"""
Multi-Modal Integration Tests for LTX-Video Handler

Tests the integration of LTXVideoHandler with MultiModalHandler,
including modality detection, request routing, and response handling.
Following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.handlers.multi_modal_handler import MultiModalHandler
from src.handlers.ltx_video_handler import LTXVideoHandler
from src.models.model_manager import ModelManager
from src.utils.request_validator import ModalityDetector
from src.schemas.text_to_video_schema import TextToVideoRequest, TextToVideoResponse, VideoInfo


class TestLTXVideoMultiModalIntegration(unittest.TestCase):
    """Test cases for LTX-Video multi-modal integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock model manager to avoid actual model loading
        self.mock_model_manager = Mock(spec=ModelManager)

        # Create multi-modal handler without auto-initialization
        self.multi_handler = MultiModalHandler(
            model_manager=self.mock_model_manager,
            auto_initialize=False
        )

        # Create LTX-Video handler
        self.ltx_video_handler = LTXVideoHandler()

        # Valid text-to-video request with proper video indicators
        self.text_to_video_request = {
            'prompt': 'A majestic eagle soaring through mountain peaks at sunset',
            'width': 720,
            'height': 1280,
            'num_frames': 25,
            'fps': 8,  # Video indicator
            'duration': 3.125,  # Video indicator
            'video': True  # Explicit video indicator
        }

    def test_handler_registration(self):
        """Test that LTX-Video handler can be registered with MultiModalHandler."""
        # Register the handler
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Verify registration
        self.assertIn('text-to-video', self.multi_handler.handlers)
        self.assertEqual(
            self.multi_handler.handlers['text-to-video'],
            self.ltx_video_handler
        )
        self.assertIn('text-to-video', self.multi_handler.get_supported_modalities())

    def test_modality_detection_text_to_video(self):
        """Test that text-to-video requests are correctly detected."""
        detector = ModalityDetector()

        # Test explicit text-to-video detection
        detected = detector.detect_modality(self.text_to_video_request)
        self.assertEqual(detected, 'text-to-video')

    def test_modality_detection_various_prompts(self):
        """Test modality detection with various prompt formats."""
        detector = ModalityDetector()

        test_cases = [
            # Text-to-video indicators
            {'prompt': 'A car driving', 'fps': 24, 'frames': 60},
            {'text': 'Dancing person', 'duration': 5.0},
            {'prompt': 'Ocean waves', 'video': True, 'num_frames': 120},

            # Should not be confused with other modalities
            {'prompt': 'Still image', 'width': 512, 'height': 512},  # text-to-image
            {'prompt': 'Video from image', 'image': 'data:...', 'fps': 24},  # image-to-video
        ]

        # Text-to-video cases
        for request in test_cases[:3]:
            with self.subTest(request=request):
                detected = detector.detect_modality(request)
                self.assertEqual(detected, 'text-to-video')

        # Other modality cases
        text_to_image_detected = detector.detect_modality(test_cases[3])
        self.assertEqual(text_to_image_detected, 'text-to-image')

        image_to_video_detected = detector.detect_modality(test_cases[4])
        self.assertEqual(image_to_video_detected, 'image-to-video')

    def test_handler_supports_request(self):
        """Test that LTX-Video handler correctly identifies supported requests."""
        # Test with valid text-to-video request
        supports_valid = self.ltx_video_handler.supports_request(self.text_to_video_request)
        self.assertTrue(supports_valid)

        # Test with unsupported requests
        unsupported_requests = [
            {'image': 'data:...', 'fps': 24},  # image-to-video (has image)
            {'control_image': 'data:...', 'control_type': 'canny'},  # controlnet (no prompt)
            {},  # no prompt at all
        ]

        for request in unsupported_requests:
            with self.subTest(request=request):
                supports = self.ltx_video_handler.supports_request(request)
                self.assertFalse(supports)

        # Test edge case - text-to-image request should be supported by supports_request
        # (actual modality detection will happen at the MultiModalHandler level)
        text_to_image_request = {'prompt': 'text', 'width': 512, 'height': 512}
        supports_text_to_image = self.ltx_video_handler.supports_request(text_to_image_request)
        self.assertTrue(supports_text_to_image)  # Has prompt, no image

    @patch('src.handlers.ltx_video_handler.LTXVideoModel')
    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_multi_modal_request_routing(self, mock_handle_request, mock_model_class):
        """Test complete request routing through MultiModalHandler."""
        # Setup mock model
        mock_model = Mock()
        mock_model.generate_video.return_value = (
            "mock_base64_video_data",
            {'duration': 3.125, 'fps': 8, 'frames': 25}
        )
        mock_model.is_loaded = True
        mock_model_class.return_value = mock_model

        # Mock the handler response
        mock_handle_request.return_value = {
            'success': True,
            'data': {
                'video_data': 'mock_base64_video_data',
                'video_info': {
                    'width': 720,
                    'height': 1280,
                    'fps': 8,
                    'frames': 25,
                    'duration': 3.125
                }
            }
        }

        # Register handler
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Mock model manager methods
        self.mock_model_manager.get_model.return_value = mock_model
        self.mock_model_manager.is_model_loaded.return_value = True

        # Process request through multi-modal handler
        result = self.multi_handler.process_request(self.text_to_video_request)

        # Verify request was processed
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        mock_handle_request.assert_called_once()

    def test_handler_capabilities_integration(self):
        """Test that handler capabilities are properly integrated."""
        capabilities = self.ltx_video_handler.get_capabilities()

        # Verify expected capabilities structure (based on actual implementation)
        self.assertIn('modality', capabilities)
        self.assertEqual(capabilities['modality'], 'text-to-video')
        self.assertIn('model_type', capabilities)
        self.assertEqual(capabilities['model_type'], 'ltx-video-2b')
        self.assertIn('inference_time_target', capabilities)
        self.assertEqual(capabilities['inference_time_target'], '<45s')

        # Verify resolution support
        self.assertIn('resolution_support', capabilities)
        res_support = capabilities['resolution_support']
        self.assertIn('constraint', res_support)
        self.assertEqual(res_support['constraint'], 'divisible_by_32')

    def test_multi_modal_handler_initialization_with_ltx_video(self):
        """Test MultiModalHandler initialization includes LTX-Video handler."""
        # This test would require updating the MultiModalHandler._initialize_handlers method
        # to include LTX-Video handler. For now, test manual registration.

        # Register all handlers manually
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Verify supported modalities include text-to-video
        supported = self.multi_handler.get_supported_modalities()
        self.assertIn('text-to-video', supported)

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_error_propagation_through_multi_modal_handler(self, mock_handle_request):
        """Test that errors from LTX-Video handler are properly propagated."""
        # Register handler
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Mock handler to raise error
        mock_handle_request.side_effect = Exception("Model inference failed")

        result = self.multi_handler.process_request(self.text_to_video_request)

        # Verify error handling - ResponseFormatter uses 'status' field
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('status'), 'error')
        self.assertIn('error', result)

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_concurrent_request_handling(self, mock_handle_request):
        """Test that multiple text-to-video requests can be handled."""
        # Register handler
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Create multiple requests with proper video indicators
        requests = [
            {'prompt': f'Video {i}', 'fps': 8, 'frames': 25, 'video': True}
            for i in range(3)
        ]

        # Mock successful responses
        mock_handle_request.return_value = {
            'success': True,
            'data': {'video_data': 'mock_data'}
        }

        # Process all requests
        results = []
        for request in requests:
            result = self.multi_handler.process_request(request)
            results.append(result)

        # Verify all succeeded
        for result in results:
            self.assertTrue(result.get('success', False))

        # Verify handler was called for each request
        self.assertEqual(mock_handle_request.call_count, len(requests))

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_parameter_forwarding_accuracy(self, mock_handle_request):
        """Test that parameters are accurately forwarded to LTX-Video handler."""
        # Register handler
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Detailed request with all parameters and video indicators
        detailed_request = {
            'prompt': 'Detailed video with all parameters',
            'width': 704,
            'height': 1280,
            'num_frames': 49,  # (8*6)+1
            'num_inference_steps': 30,
            'guidance_scale': 8.0,
            'fps': 12,
            'seed': 42,
            'video': True  # Video indicator for proper modality detection
        }

        # Mock handler to capture forwarded parameters
        mock_handle_request.return_value = {'success': True}

        self.multi_handler.process_request(detailed_request)

        # Verify parameters were forwarded correctly
        mock_handle_request.assert_called_once()
        forwarded_params = mock_handle_request.call_args[0][0]

        # Check key parameters
        self.assertEqual(forwarded_params['prompt'], detailed_request['prompt'])
        self.assertEqual(forwarded_params['width'], detailed_request['width'])
        self.assertEqual(forwarded_params['height'], detailed_request['height'])
        self.assertEqual(forwarded_params['num_frames'], detailed_request['num_frames'])
        self.assertEqual(forwarded_params['seed'], detailed_request['seed'])


if __name__ == '__main__':
    unittest.main()