"""
Unit tests for ControlNet handler.

Tests the ControlNet handler's parameter validation, request processing,
and integration with model management system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import base64
from PIL import Image
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from handlers.controlnet_handler import ControlNetHandler
from schemas.controlnet_schema import ControlNetRequest, ControlNetOutput
from utils.exceptions import ValidationError, InferenceError, ModelLoadError
from models.controlnet_model import ControlNetModel


class TestControlNetHandler(unittest.TestCase):
    """Test cases for ControlNetHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = ControlNetHandler(['canny', 'depth'])

        # Create test image as base64
        test_image = Image.new('RGB', (256, 256), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        self.test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Valid test request
        self.valid_request = {
            'prompt': 'A beautiful landscape',
            'control_image': self.test_image_b64,
            'control_type': 'canny',
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'control_strength': 1.0
        }

    def test_handler_initialization(self):
        """Test handler initialization."""
        self.assertEqual(self.handler.HANDLER_NAME, "controlnet-guided-generation")
        self.assertEqual(self.handler.SUPPORTED_MODALITY, "controlnet")
        self.assertEqual(self.handler.control_types, ['canny', 'depth'])
        self.assertIn('prompt', self.handler.required_parameters)
        self.assertIn('control_image', self.handler.required_parameters)
        self.assertIn('control_type', self.handler.required_parameters)

    def test_handler_initialization_invalid_control_type(self):
        """Test handler initialization with invalid control type."""
        with self.assertRaises(ValueError):
            ControlNetHandler(['invalid_type'])

    def test_validate_request_valid(self):
        """Test request validation with valid parameters."""
        result = self.handler.validate_request(self.valid_request)

        self.assertEqual(result['prompt'], 'A beautiful landscape')
        self.assertEqual(result['control_type'], 'canny')
        self.assertEqual(result['width'], 512)
        self.assertEqual(result['height'], 512)

    def test_validate_request_missing_required_param(self):
        """Test request validation with missing required parameter."""
        invalid_request = self.valid_request.copy()
        del invalid_request['prompt']

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_control_type(self):
        """Test request validation with invalid control type."""
        invalid_request = self.valid_request.copy()
        invalid_request['control_type'] = 'invalid'

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_control_image(self):
        """Test request validation with invalid control image."""
        invalid_request = self.valid_request.copy()
        invalid_request['control_image'] = 'invalid_base64'

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_dimensions(self):
        """Test request validation with invalid dimensions."""
        invalid_request = self.valid_request.copy()
        invalid_request['width'] = 100  # Below minimum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_guidance_range(self):
        """Test request validation with invalid guidance range."""
        invalid_request = self.valid_request.copy()
        invalid_request['control_guidance_start'] = 0.8
        invalid_request['control_guidance_end'] = 0.5  # End before start

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_canny_thresholds(self):
        """Test request validation with invalid Canny thresholds."""
        invalid_request = self.valid_request.copy()
        invalid_request['canny_low_threshold'] = 200
        invalid_request['canny_high_threshold'] = 100  # High < Low

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    @patch('handlers.controlnet_handler.ControlNetModel')
    def test_process_request_success(self, mock_controlnet_model_class):
        """Test successful request processing."""
        # Mock model manager
        mock_model_manager = Mock()
        mock_controlnet_model = Mock(spec=ControlNetModel)

        # Mock image generation
        test_output_image = Image.new('RGB', (512, 512), color='blue')
        generation_info = {
            'control_type': 'canny',
            'control_info': {
                'original_width': 256,
                'original_height': 256,
                'processed_width': 512,
                'processed_height': 512,
                'control_type': 'canny',
                'processing_time_ms': 150.0
            },
            'inference_time_s': 12.5,
            'model_memory_mb': 15000.0
        }

        mock_controlnet_model.generate_image.return_value = (test_output_image, generation_info)
        mock_model_manager.get_model.return_value = mock_controlnet_model

        # Process request
        result = self.handler.process_request(self.valid_request, mock_model_manager)

        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['images']), 1)
        self.assertIn('inference_time_ms', result)
        self.assertIn('preprocessing_time_ms', result)

        # Verify model was called correctly
        mock_controlnet_model.generate_image.assert_called_once()
        call_args = mock_controlnet_model.generate_image.call_args
        self.assertEqual(call_args[1]['prompt'], 'A beautiful landscape')
        self.assertEqual(call_args[1]['control_type'], 'canny')

    @patch('handlers.controlnet_handler.ControlNetModel')
    def test_process_request_model_load_failure(self, mock_controlnet_model_class):
        """Test request processing with model load failure."""
        # Mock model manager that fails to load model
        mock_model_manager = Mock()
        mock_model_manager.get_model.side_effect = ModelLoadError("test-model", "Failed to load model")

        # Process request
        result = self.handler.process_request(self.valid_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error_type'], 'modelloaderror')

    @patch('handlers.controlnet_handler.ControlNetModel')
    def test_process_request_inference_failure(self, mock_controlnet_model_class):
        """Test request processing with inference failure."""
        # Mock model manager
        mock_model_manager = Mock()
        mock_controlnet_model = Mock(spec=ControlNetModel)
        mock_controlnet_model.generate_image.side_effect = InferenceError("Inference failed")
        mock_model_manager.get_model.return_value = mock_controlnet_model

        # Process request
        result = self.handler.process_request(self.valid_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error_type'], 'inferenceerror')

    def test_extract_control_params_canny(self):
        """Test extraction of Canny-specific parameters."""
        request_data = self.valid_request.copy()
        request_data.update({
            'canny_low_threshold': 150,
            'canny_high_threshold': 250
        })

        params = self.handler._extract_control_params(request_data)

        self.assertEqual(params['low_threshold'], 150)
        self.assertEqual(params['high_threshold'], 250)

    def test_extract_control_params_depth(self):
        """Test extraction of depth-specific parameters."""
        request_data = self.valid_request.copy()
        request_data['control_type'] = 'depth'

        params = self.handler._extract_control_params(request_data)

        self.assertTrue(params['normalize_depth'])
        self.assertTrue(params['invert_depth'])

    def test_get_handler_info(self):
        """Test handler information retrieval."""
        info = self.handler.get_handler_info()

        self.assertEqual(info['name'], self.handler.HANDLER_NAME)
        self.assertEqual(info['supported_modality'], self.handler.SUPPORTED_MODALITY)
        self.assertEqual(info['control_types'], ['canny', 'depth'])
        self.assertIn('required_parameters', info)
        self.assertIn('optional_parameters', info)
        self.assertIn('performance_stats', info)

    def test_get_performance_stats_no_inferences(self):
        """Test performance statistics with no inferences."""
        stats = self.handler.get_performance_stats()

        self.assertEqual(stats['successful_inferences'], 0)
        self.assertEqual(stats['failed_inferences'], 0)
        self.assertEqual(stats['total_inferences'], 0)
        self.assertEqual(stats['success_rate'], 0.0)

    def test_get_performance_stats_with_inferences(self):
        """Test performance statistics with inference data."""
        # Simulate some inferences
        self.handler.successful_inferences = 8
        self.handler.failed_inferences = 2
        self.handler.total_processing_time = 120.0
        self.handler.control_type_stats['canny']['count'] = 5
        self.handler.control_type_stats['canny']['total_time'] = 60.0

        stats = self.handler.get_performance_stats()

        self.assertEqual(stats['successful_inferences'], 8)
        self.assertEqual(stats['failed_inferences'], 2)
        self.assertEqual(stats['total_inferences'], 10)
        self.assertEqual(stats['success_rate'], 0.8)
        self.assertEqual(stats['avg_processing_time_s'], 15.0)
        self.assertEqual(stats['control_type_stats']['canny']['inference_count'], 5)
        self.assertEqual(stats['control_type_stats']['canny']['avg_time_s'], 12.0)

    def test_supports_modality(self):
        """Test modality support checking."""
        self.assertTrue(self.handler.supports_modality('controlnet'))
        self.assertFalse(self.handler.supports_modality('text-to-image'))
        self.assertFalse(self.handler.supports_modality('invalid'))

    def test_get_required_models(self):
        """Test required models list."""
        models = self.handler.get_required_models()
        self.assertEqual(models, ['controlnet-canny-depth'])

    def test_estimate_processing_time_canny(self):
        """Test processing time estimation for Canny."""
        estimated_time = self.handler.estimate_processing_time(self.valid_request)

        # Should be around 15s base + 2.5s overhead = ~17.5s for 512x512, 20 steps
        self.assertGreater(estimated_time, 15.0)
        self.assertLess(estimated_time, 25.0)

    def test_estimate_processing_time_depth(self):
        """Test processing time estimation for depth."""
        depth_request = self.valid_request.copy()
        depth_request['control_type'] = 'depth'

        estimated_time = self.handler.estimate_processing_time(depth_request)

        # Depth should be slightly slower than Canny
        canny_time = self.handler.estimate_processing_time(self.valid_request)
        self.assertGreater(estimated_time, canny_time)

    def test_estimate_processing_time_larger_image(self):
        """Test processing time estimation for larger images."""
        large_request = self.valid_request.copy()
        large_request.update({'width': 1024, 'height': 1024})

        large_time = self.handler.estimate_processing_time(large_request)
        normal_time = self.handler.estimate_processing_time(self.valid_request)

        # Larger image should take longer (4x pixels = ~4x time)
        self.assertGreater(large_time, normal_time * 2)

    def test_get_supported_output_formats(self):
        """Test supported output formats."""
        formats = self.handler.get_supported_output_formats()
        self.assertEqual(formats, ['png', 'jpg', 'webp'])

    def test_get_parameter_constraints(self):
        """Test parameter constraints retrieval."""
        constraints = self.handler.get_parameter_constraints()

        self.assertIn('width', constraints)
        self.assertIn('height', constraints)
        self.assertIn('control_strength', constraints)

        # Check specific constraints
        self.assertEqual(constraints['width']['min'], 256)
        self.assertEqual(constraints['width']['max'], 2048)
        self.assertEqual(constraints['control_strength']['min'], 0.0)
        self.assertEqual(constraints['control_strength']['max'], 2.0)

    def test_validate_parameters_delegation(self):
        """Test that validate_parameters delegates to validate_request."""
        # Should not raise for valid request
        result = self.handler.validate_parameters(self.valid_request)
        self.assertIsNotNone(result)

        # Should raise for invalid request
        invalid_request = self.valid_request.copy()
        del invalid_request['prompt']

        with self.assertRaises(ValidationError):
            self.handler.validate_parameters(invalid_request)


if __name__ == '__main__':
    unittest.main()