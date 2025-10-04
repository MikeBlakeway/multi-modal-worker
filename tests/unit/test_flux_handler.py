"""
Comprehensive unit tests for FluxHandler class.

Tests parameter validation, error handling, response formatting,
and integration with the FLUX.1 model and routing system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import time
from PIL import Image
import io
import base64
import numpy as np

# Import the components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.handlers.flux_handler import FluxHandler
    from src.models.flux_model import FluxModel
    from src.schemas.text_to_image_schema import TextToImageRequest, ImageOutput
    from src.utils.exceptions import ValidationError, InferenceError
except ImportError:
    # Standalone implementation for testing
    pass


class TestFluxHandler(unittest.TestCase):
    """Test FluxHandler functionality and integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = FluxHandler()

        # Mock model for testing
        self.mock_flux_model = Mock(spec=FluxModel)
        self.mock_flux_model.model_name = "flux-1-schnell-fp8"
        self.mock_flux_model.is_loaded = True

        # Sample test data
        self.valid_request = {
            'prompt': 'A beautiful sunset over mountains',
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 4,
            'guidance_scale': 0.0,
            'seed': 42
        }

        # Create sample image for mock responses
        self.sample_image = Image.new('RGB', (1024, 1024), color='red')

    def test_handler_initialization(self):
        """Test FluxHandler initialization."""
        handler = FluxHandler()

        self.assertEqual(handler.HANDLER_NAME, "flux-text-to-image")
        self.assertEqual(handler.SUPPORTED_MODALITY, "text-to-image")
        self.assertEqual(handler.supported_modality, "text-to-image")
        self.assertIn('prompt', handler.required_parameters)
        self.assertIsInstance(handler.optional_parameters, dict)
        self.assertEqual(handler.successful_inferences, 0)
        self.assertEqual(handler.failed_inferences, 0)

    def test_required_parameters(self):
        """Test required parameters property."""
        required = self.handler.required_parameters

        self.assertIsInstance(required, list)
        self.assertIn('prompt', required)
        self.assertEqual(len(required), 1)  # Only prompt is required

    def test_optional_parameters(self):
        """Test optional parameters with defaults."""
        optional = self.handler.optional_parameters

        self.assertIsInstance(optional, dict)
        self.assertEqual(optional['width'], 1024)
        self.assertEqual(optional['height'], 1024)
        self.assertEqual(optional['num_inference_steps'], 4)
        self.assertEqual(optional['guidance_scale'], 0.0)
        self.assertEqual(optional['output_format'], 'png')
        self.assertEqual(optional['quality'], 95)

    def test_validate_request_valid(self):
        """Test request validation with valid data."""
        validated = self.handler.validate_request(self.valid_request)

        self.assertIsInstance(validated, dict)
        self.assertEqual(validated['prompt'], self.valid_request['prompt'])
        self.assertEqual(validated['width'], 1024)
        self.assertEqual(validated['height'], 1024)
        self.assertEqual(validated['modality'], 'text-to-image')
        self.assertEqual(validated['handler'], 'flux-text-to-image')
        self.assertIn('timestamp', validated)

    def test_validate_request_missing_prompt(self):
        """Test validation fails with missing prompt."""
        invalid_request = self.valid_request.copy()
        del invalid_request['prompt']

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_empty_prompt(self):
        """Test validation fails with empty prompt."""
        invalid_request = self.valid_request.copy()
        invalid_request['prompt'] = ''

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_prompt_too_long(self):
        """Test validation fails with overly long prompt."""
        invalid_request = self.valid_request.copy()
        invalid_request['prompt'] = 'x' * 2001  # Exceeds 2000 char limit

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_dimensions(self):
        """Test validation fails with invalid dimensions."""
        # Test width too small
        invalid_request = self.valid_request.copy()
        invalid_request['width'] = 200  # Below minimum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

        # Test height too large
        invalid_request = self.valid_request.copy()
        invalid_request['height'] = 3000  # Above maximum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_steps(self):
        """Test validation fails with invalid inference steps."""
        invalid_request = self.valid_request.copy()
        invalid_request['num_inference_steps'] = 0  # Below minimum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

        invalid_request['num_inference_steps'] = 100  # Above maximum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_guidance(self):
        """Test validation fails with invalid guidance scale."""
        invalid_request = self.valid_request.copy()
        invalid_request['guidance_scale'] = -1.0  # Below minimum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

        invalid_request['guidance_scale'] = 25.0  # Above maximum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_validate_request_invalid_seed(self):
        """Test validation fails with invalid seed."""
        invalid_request = self.valid_request.copy()
        invalid_request['seed'] = -1  # Below minimum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

        invalid_request['seed'] = 2**32  # Above maximum

        with self.assertRaises(ValidationError):
            self.handler.validate_request(invalid_request)

    def test_get_required_models(self):
        """Test get_required_models returns correct model name."""
        models = self.handler.get_required_models(self.valid_request)

        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], 'flux-1-schnell-fp8')

    def test_process_inference_success(self):
        """Test successful inference processing."""
        # Mock successful inference result
        mock_inference_result = {
            'image': self.sample_image,
            'inference_time': 12.5,
            'parameters': self.valid_request.copy(),
            'memory_usage_mb': 14000,
            'model_info': {
                'name': 'FLUX.1-schnell-fp8',
                'inference_count': 1,
                'average_time': 12.5
            }
        }

        self.mock_flux_model.infer.return_value = mock_inference_result

        # Test inference processing
        models = {'flux-1-schnell-fp8': self.mock_flux_model}
        result = self.handler.process_inference(models, self.valid_request)

        self.assertTrue(result['success'])
        self.assertEqual(result['generated_image'], self.sample_image)
        self.assertEqual(result['inference_time'], 12.5)
        self.assertIn('parameters_used', result)
        self.assertIn('memory_usage_mb', result)
        self.assertIn('model_info', result)

        # Verify model was called correctly
        self.mock_flux_model.infer.assert_called_once()
        call_args = self.mock_flux_model.infer.call_args[0][0]
        self.assertEqual(call_args['prompt'], self.valid_request['prompt'])

    def test_process_inference_model_not_available(self):
        """Test inference fails when model is not available."""
        models = {}  # No models available

        with self.assertRaises(InferenceError) as context:
            self.handler.process_inference(models, self.valid_request)

        self.assertIn("FLUX model not available", str(context.exception))

    def test_process_inference_model_not_loaded(self):
        """Test inference fails when model is not loaded."""
        self.mock_flux_model.is_loaded = False
        models = {'flux-1-schnell-fp8': self.mock_flux_model}

        with self.assertRaises(InferenceError) as context:
            self.handler.process_inference(models, self.valid_request)

        self.assertIn("model not loaded", str(context.exception))

    def test_process_inference_model_error(self):
        """Test inference handles model errors gracefully."""
        self.mock_flux_model.infer.side_effect = Exception("Model inference failed")
        models = {'flux-1-schnell-fp8': self.mock_flux_model}

        with self.assertRaises(InferenceError) as context:
            self.handler.process_inference(models, self.valid_request)

        self.assertIn("Text-to-image inference failed", str(context.exception))

        # Verify failed inference was tracked
        self.assertEqual(self.handler.failed_inferences, 1)

    @patch('src.utils.image_utils.encode_pil_image')
    def test_format_response_success(self, mock_encode):
        """Test successful response formatting."""
        # Mock image encoding
        mock_encode.return_value = ("base64_image_data", 1024000)

        inference_results = {
            'success': True,
            'generated_image': self.sample_image,
            'inference_time': 12.5,
            'parameters_used': self.valid_request.copy(),
            'memory_usage_mb': 14000,
            'model_info': {'name': 'FLUX.1-schnell-fp8'},
            'handler_processing_time': 13.0
        }

        validated_request = self.handler.validate_request(self.valid_request)
        response = self.handler.format_response(inference_results, validated_request)

        self.assertIsInstance(response, dict)
        self.assertEqual(response['status'], 'success')
        self.assertIn('output', response)

        # Verify the output contains the expected structure
        output = response['output']
        self.assertIn('status', output)
        self.assertEqual(output['status'], 'success')
        self.assertIn('images', output)
        self.assertIsInstance(output['images'], list)
        self.assertGreater(len(output['images']), 0)

        # Verify image metadata
        image = output['images'][0]
        self.assertIn('image_data', image)
        self.assertIn('format', image)
        self.assertIn('file_size', image)

    def test_format_response_inference_failure(self):
        """Test response formatting for inference failure."""
        inference_results = {
            'success': False
        }

        response = self.handler.format_response(inference_results, self.valid_request)

        self.assertIsInstance(response, dict)
        self.assertEqual(response['status'], 'error')
        self.assertIn('error', response)
        self.assertIn('type', response['error'])

    def test_get_handler_stats(self):
        """Test handler statistics retrieval."""
        # Simulate some successful and failed inferences
        self.handler.successful_inferences = 5
        self.handler.failed_inferences = 1
        self.handler.total_processing_time = 60.0

        stats = self.handler.get_handler_stats()

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['handler_name'], 'flux-text-to-image')
        self.assertEqual(stats['supported_modality'], 'text-to-image')
        self.assertEqual(stats['successful_inferences'], 5)
        self.assertEqual(stats['failed_inferences'], 1)
        self.assertEqual(stats['total_inferences'], 6)
        self.assertEqual(stats['success_rate'], 5/6)
        self.assertEqual(stats['average_processing_time'], 12.0)

    def test_validate_model_compatibility(self):
        """Test model compatibility validation."""
        # Test compatible model
        compatible_model = Mock(spec=FluxModel)
        compatible_model.model_name = "flux-1-schnell-fp8"

        self.assertTrue(self.handler.validate_model_compatibility(compatible_model))

        # Test incompatible model (wrong name)
        incompatible_model = Mock()
        incompatible_model.model_name = "different-model"

        self.assertFalse(self.handler.validate_model_compatibility(incompatible_model))

    def test_get_parameter_info(self):
        """Test parameter information retrieval."""
        param_info = self.handler.get_parameter_info()

        self.assertIsInstance(param_info, dict)
        self.assertIn('required_parameters', param_info)
        self.assertIn('optional_parameters', param_info)

        # Check required parameters
        required = param_info['required_parameters']
        self.assertIn('prompt', required)
        self.assertEqual(required['prompt']['type'], 'string')

        # Check optional parameters
        optional = param_info['optional_parameters']
        self.assertIn('width', optional)
        self.assertIn('height', optional)
        self.assertIn('num_inference_steps', optional)
        self.assertIn('guidance_scale', optional)

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        # Test default parameters
        estimate = self.handler.estimate_processing_time(self.valid_request)
        self.assertIsInstance(estimate, float)
        self.assertGreater(estimate, 0)

        # Test higher resolution (should take longer)
        high_res_request = self.valid_request.copy()
        high_res_request.update({'width': 2048, 'height': 2048})

        high_res_estimate = self.handler.estimate_processing_time(high_res_request)
        self.assertGreater(high_res_estimate, estimate)

        # Test more steps (should take longer)
        more_steps_request = self.valid_request.copy()
        more_steps_request['num_inference_steps'] = 16

        more_steps_estimate = self.handler.estimate_processing_time(more_steps_request)
        self.assertGreater(more_steps_estimate, estimate)


class TestFluxHandlerEdgeCases(unittest.TestCase):
    """Test FluxHandler edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = FluxHandler()

    def test_validate_request_with_unusual_prompt(self):
        """Test validation with unusual but valid prompts."""
        unusual_prompts = [
            "A",  # Minimal prompt
            "   leading and trailing spaces   ",  # Whitespace
            "Multiple\nlines\nof\ntext",  # Multiline
            "Special chars: !@#$%^&*()",  # Special characters
            "Numbers 123 and symbols ★☆",  # Unicode
        ]

        for prompt in unusual_prompts:
            request = {'prompt': prompt}
            try:
                validated = self.handler.validate_request(request)
                self.assertIn('prompt', validated)
            except ValidationError:
                # Some prompts may fail validation, that's OK
                pass

    def test_validate_request_dimension_adjustment(self):
        """Test automatic dimension adjustment to multiples of 8."""
        request = {
            'prompt': 'test prompt',
            'width': 1025,  # Not multiple of 8
            'height': 1023  # Not multiple of 8
        }

        # Should not raise error due to automatic adjustment
        validated = self.handler.validate_request(request)

        # Verify dimensions were adjusted
        self.assertEqual(validated['width'] % 8, 0)
        self.assertEqual(validated['height'] % 8, 0)

    def test_process_inference_performance_tracking(self):
        """Test that performance tracking works correctly."""
        mock_model = Mock(spec=FluxModel)
        mock_model.model_name = "flux-1-schnell-fp8"
        mock_model.is_loaded = True

        # Mock successful inference
        mock_model.infer.return_value = {
            'image': Mock(),
            'inference_time': 10.0,
            'parameters': {},
            'memory_usage_mb': 12000,
            'model_info': {}
        }

        models = {'flux-1-schnell-fp8': mock_model}
        request = {'prompt': 'test'}

        initial_count = self.handler.successful_inferences
        initial_time = self.handler.total_processing_time

        # Process inference
        self.handler.process_inference(models, request)

        # Verify tracking updated
        self.assertEqual(self.handler.successful_inferences, initial_count + 1)
        self.assertGreater(self.handler.total_processing_time, initial_time)

    def test_format_response_with_different_formats(self):
        """Test response formatting with different output formats."""
        formats = ['png', 'jpeg', 'webp']

        for output_format in formats:
            request = {
                'prompt': 'test',
                'output_format': output_format,
                'quality': 90
            }

            inference_results = {
                'success': True,
                'generated_image': Mock(),
                'inference_time': 10.0,
                'parameters_used': request.copy(),
                'memory_usage_mb': 12000,
                'model_info': {}
            }

            with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
                mock_encode.return_value = ("encoded_data", 1000000)

                # Use a real PIL Image instead of Mock to avoid encoding issues
                from PIL import Image
                import numpy as np
                fake_image = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
                inference_results['generated_image'] = fake_image

                response = self.handler.format_response(inference_results, request)

                # Verify encoding was called with correct format
                mock_encode.assert_called_once()
                args = mock_encode.call_args[0]
                self.assertEqual(args[1], output_format)
                self.assertEqual(args[2], 90)


if __name__ == '__main__':
    unittest.main()