"""
Integration tests for complete text-to-image workflow.

Tests end-to-end text-to-image pipeline from request validation
through FLUX.1 inference to response formatting.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import time
from PIL import Image
import base64
import io

# Import the components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.handlers.multi_modal_handler import MultiModalHandler
    from src.handlers.flux_handler import FluxHandler
    from src.models.flux_model import FluxModel
    from src.models.model_manager import ModelManager
    from src.utils.exceptions import ValidationError, InferenceError
except ImportError:
    # Standalone implementation for testing
    pass


class TestTextToImageWorkflow(unittest.TestCase):
    """Test complete text-to-image workflow integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock model manager
        self.mock_model_manager = Mock(spec=ModelManager)

        # Initialize multi-modal handler
        with patch.object(MultiModalHandler, '_initialize_handlers'):
            self.multi_handler = MultiModalHandler(self.mock_model_manager)

        # Manually add FLUX handler for testing
        self.flux_handler = FluxHandler()
        self.multi_handler.register_handler('text-to-image', self.flux_handler)

        # Create test request data
        self.valid_request = {
            'prompt': 'A beautiful mountain landscape at sunset with vibrant colors',
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 4,
            'guidance_scale': 0.0,
            'seed': 42,
            'output_format': 'png',
            'quality': 95
        }

        # Create sample output image
        self.sample_image = Image.new('RGB', (1024, 1024), color='red')

        # Mock model loading result
        self.mock_flux_model = Mock(spec=FluxModel)
        self.mock_flux_model.model_name = "flux-1-schnell-fp8"
        self.mock_flux_model.is_loaded = True

    def create_base64_image(self, image: Image.Image) -> str:
        """Helper to create base64 encoded image."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_complete_text_to_image_workflow_success(self, mock_encode_image):
        """Test successful end-to-end text-to-image generation."""
        # Mock image encoding
        base64_data = self.create_base64_image(self.sample_image)
        mock_encode_image.return_value = (base64_data, 1024000)

        # Mock model manager to return loaded FLUX model
        self.mock_model_manager.get_model.return_value = self.mock_flux_model

        # Mock FLUX model inference
        mock_inference_result = {
            'image': self.sample_image,
            'inference_time': 12.5,
            'parameters': self.valid_request.copy(),
            'memory_usage_mb': 14000,
            'model_info': {
                'name': 'FLUX.1-schnell-fp8',
                'variant': 'fp8',
                'inference_count': 1,
                'average_time': 12.5
            }
        }
        self.mock_flux_model.infer.return_value = mock_inference_result

        # Process complete request
        response = self.multi_handler.process_request(self.valid_request)

        # Verify successful response structure
        self.assertIsInstance(response, dict)
        self.assertEqual(response['status'], 'success')
        self.assertIn('output', response)

        # Verify output contains image data
        output = response['output']
        self.assertIn('images', output)
        self.assertEqual(len(output['images']), 1)

        # Verify image metadata
        image_output = output['images'][0]
        self.assertIn('image_data', image_output)
        self.assertEqual(image_output['format'], 'png')
        self.assertEqual(image_output['width'], 1024)
        self.assertEqual(image_output['height'], 1024)
        self.assertEqual(image_output['seed_used'], 42)

        # Verify request parameters were preserved
        self.assertEqual(output['prompt_used'], self.valid_request['prompt'])
        self.assertIn('inference_time', output)
        self.assertIn('model_info', output)

        # Verify model manager was called correctly
        self.mock_model_manager.get_model.assert_called_with('flux-1-schnell-fp8')

        # Verify FLUX model was called with correct parameters
        self.mock_flux_model.infer.assert_called_once()
        inference_args = self.mock_flux_model.infer.call_args[0][0]
        self.assertEqual(inference_args['prompt'], self.valid_request['prompt'])
        self.assertEqual(inference_args['width'], 1024)
        self.assertEqual(inference_args['height'], 1024)

    def test_text_to_image_workflow_validation_error(self):
        """Test workflow handling of validation errors."""
        # Invalid request (missing prompt)
        invalid_request = self.valid_request.copy()
        del invalid_request['prompt']

        # Process request
        response = self.multi_handler.process_request(invalid_request)

        # Verify error response
        self.assertEqual(response['status'], 'error')
        self.assertIn('error_type', response)
        self.assertIn('error_message', response)

        # Verify model manager was not called
        self.mock_model_manager.get_model.assert_not_called()

    def test_text_to_image_workflow_model_loading_error(self):
        """Test workflow handling of model loading errors."""
        # Mock model loading failure
        self.mock_model_manager.get_model.side_effect = Exception("Model loading failed")

        # Process request
        response = self.multi_handler.process_request(self.valid_request)

        # Verify error response
        self.assertEqual(response['status'], 'error')
        self.assertIn('error_type', response)

        # Verify model loading was attempted
        self.mock_model_manager.get_model.assert_called()

    def test_text_to_image_workflow_inference_error(self):
        """Test workflow handling of inference errors."""
        # Mock model manager to return model
        self.mock_model_manager.load_models.return_value = {
            'flux-1-schnell-fp8': self.mock_flux_model
        }

        # Mock inference failure
        self.mock_flux_model.infer.side_effect = InferenceError("Inference failed")

        # Process request
        response = self.multi_handler.process_request(self.valid_request)

        # Verify error response
        self.assertEqual(response['status'], 'error')
        self.assertIn('error_type', response)

        # Verify inference was attempted
        self.mock_flux_model.infer.assert_called_once()

    def test_text_to_image_different_parameters(self):
        """Test workflow with different parameter combinations."""
        # Test different resolutions and settings
        test_cases = [
            {'width': 512, 'height': 512, 'num_inference_steps': 8},
            {'width': 768, 'height': 1024, 'guidance_scale': 1.0},
            {'width': 1024, 'height': 768, 'output_format': 'jpeg', 'quality': 85},
        ]

        for test_params in test_cases:
            with self.subTest(params=test_params):
                # Create request with test parameters
                request = self.valid_request.copy()
                request.update(test_params)

                # Mock successful inference
                self.mock_model_manager.reset_mock()
                self.mock_flux_model.reset_mock()

                self.mock_model_manager.load_models.return_value = {
                    'flux-1-schnell-fp8': self.mock_flux_model
                }

                mock_inference_result = {
                    'image': Image.new('RGB', (request['width'], request['height']), color='blue'),
                    'inference_time': 10.0,
                    'parameters': request.copy(),
                    'memory_usage_mb': 12000,
                    'model_info': {}
                }
                self.mock_flux_model.infer.return_value = mock_inference_result

                with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
                    mock_encode.return_value = ("encoded_data", 500000)

                    # Process request
                    response = self.multi_handler.process_request(request)

                    # Verify success
                    self.assertEqual(response['status'], 'success')

                    # Verify parameters were passed correctly
                    inference_args = self.mock_flux_model.infer.call_args[0][0]
                    for key, value in test_params.items():
                        if key in ['output_format', 'quality']:
                            continue  # These are handled by response formatter
                        self.assertEqual(inference_args.get(key, self.flux_handler.DEFAULT_PARAMS.get(key)), value)

    def test_text_to_image_workflow_performance(self):
        """Test workflow performance tracking."""
        # Mock successful workflow
        self.mock_model_manager.load_models.return_value = {
            'flux-1-schnell-fp8': self.mock_flux_model
        }

        mock_inference_result = {
            'image': self.sample_image,
            'inference_time': 8.5,
            'parameters': self.valid_request.copy(),
            'memory_usage_mb': 13500,
            'model_info': {}
        }
        self.mock_flux_model.infer.return_value = mock_inference_result

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("encoded_data", 800000)

            # Record initial stats
            initial_request_count = self.multi_handler.request_count
            initial_successful = self.flux_handler.successful_inferences

            # Process request
            start_time = time.time()
            response = self.multi_handler.process_request(self.valid_request)
            end_time = time.time()

            # Verify performance tracking
            self.assertEqual(self.multi_handler.request_count, initial_request_count + 1)
            self.assertEqual(self.flux_handler.successful_inferences, initial_successful + 1)

            # Verify timing information
            self.assertIn('inference_time', response['output'])
            self.assertEqual(response['output']['inference_time'], 8.5)

            # Verify total processing time is reasonable
            total_time = end_time - start_time
            self.assertLess(total_time, 5.0)  # Should be much faster in mock

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_text_to_image_output_formats(self, mock_encode):
        """Test different output format handling."""
        formats_to_test = ['png', 'jpeg', 'webp']

        for output_format in formats_to_test:
            with self.subTest(format=output_format):
                # Setup request with specific format
                request = self.valid_request.copy()
                request['output_format'] = output_format
                request['quality'] = 90

                # Mock encoding with different file sizes for different formats
                format_sizes = {'png': 1200000, 'jpeg': 600000, 'webp': 800000}
                mock_encode.return_value = (f"encoded_{output_format}_data", format_sizes[output_format])

                # Mock successful workflow
                self.mock_model_manager.load_models.return_value = {
                    'flux-1-schnell-fp8': self.mock_flux_model
                }

                mock_inference_result = {
                    'image': self.sample_image,
                    'inference_time': 11.0,
                    'parameters': request.copy(),
                    'memory_usage_mb': 14000,
                    'model_info': {}
                }
                self.mock_flux_model.infer.return_value = mock_inference_result

                # Process request
                response = self.multi_handler.process_request(request)

                # Verify format handling
                self.assertEqual(response['status'], 'success')

                image_output = response['output']['images'][0]
                self.assertEqual(image_output['format'], output_format)
                self.assertEqual(image_output['file_size'], format_sizes[output_format])

                # Verify encoding was called with correct parameters
                mock_encode.assert_called_with(self.sample_image, output_format, 90)

                # Reset mock for next iteration
                mock_encode.reset_mock()

    def test_text_to_image_seed_reproducibility(self):
        """Test that using the same seed produces consistent parameters."""
        # Mock model manager
        self.mock_model_manager.load_models.return_value = {
            'flux-1-schnell-fp8': self.mock_flux_model
        }

        # Mock inference with seed tracking
        def mock_inference_with_seed(inputs):
            return {
                'image': self.sample_image,
                'inference_time': 10.0,
                'parameters': inputs.copy(),
                'memory_usage_mb': 13000,
                'model_info': {}
            }

        self.mock_flux_model.infer.side_effect = mock_inference_with_seed

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("encoded_data", 900000)

            # Process same request twice with same seed
            seed = 12345
            request1 = self.valid_request.copy()
            request1['seed'] = seed

            request2 = self.valid_request.copy()
            request2['seed'] = seed

            response1 = self.multi_handler.process_request(request1)
            response2 = self.multi_handler.process_request(request2)

            # Both should succeed
            self.assertEqual(response1['status'], 'success')
            self.assertEqual(response2['status'], 'success')

            # Both should have same seed in output
            self.assertEqual(response1['output']['images'][0]['seed_used'], seed)
            self.assertEqual(response2['output']['images'][0]['seed_used'], seed)

            # Verify both calls used the same seed
            call_args = self.mock_flux_model.infer.call_args_list
            self.assertEqual(len(call_args), 2)
            self.assertEqual(call_args[0][0][0]['seed'], seed)
            self.assertEqual(call_args[1][0][0]['seed'], seed)


class TestTextToImageEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in text-to-image workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model_manager = Mock(spec=ModelManager)

        with patch.object(MultiModalHandler, '_initialize_handlers'):
            self.multi_handler = MultiModalHandler(self.mock_model_manager)

        self.flux_handler = FluxHandler()
        self.multi_handler.register_handler('text-to-image', self.flux_handler)

    def test_extremely_long_prompt_handling(self):
        """Test handling of very long prompts."""
        # Create request with maximum allowed prompt length
        max_prompt = "A" * 2000  # Maximum allowed length
        request = {
            'prompt': max_prompt,
            'width': 512,
            'height': 512
        }

        # Should validate successfully
        try:
            validated = self.flux_handler.validate_request(request)
            self.assertEqual(validated['prompt'], max_prompt)
        except ValidationError:
            self.fail("Valid maximum length prompt should not raise ValidationError")

    def test_unusual_aspect_ratios(self):
        """Test handling of unusual but valid aspect ratios."""
        unusual_dimensions = [
            (256, 2048),   # Very tall
            (2048, 256),   # Very wide
            (512, 1536),   # 1:3 ratio
            (1536, 512),   # 3:1 ratio
        ]

        for width, height in unusual_dimensions:
            with self.subTest(dimensions=(width, height)):
                request = {
                    'prompt': 'Test image',
                    'width': width,
                    'height': height
                }

                # Should validate successfully
                try:
                    validated = self.flux_handler.validate_request(request)
                    self.assertEqual(validated['width'], width)
                    self.assertEqual(validated['height'], height)
                except ValidationError:
                    self.fail(f"Valid dimensions {width}x{height} should not raise ValidationError")

    def test_minimal_valid_request(self):
        """Test handling of minimal valid request with only required fields."""
        minimal_request = {'prompt': 'A'}  # Only required field

        # Mock successful workflow
        mock_flux_model = Mock(spec=FluxModel)
        mock_flux_model.model_name = "flux-1-schnell-fp8"
        mock_flux_model.is_loaded = True

        self.mock_model_manager.load_models.return_value = {
            'flux-1-schnell-fp8': mock_flux_model
        }

        mock_inference_result = {
            'image': Image.new('RGB', (1024, 1024), color='green'),
            'inference_time': 15.0,
            'parameters': minimal_request.copy(),
            'memory_usage_mb': 12000,
            'model_info': {}
        }
        mock_flux_model.infer.return_value = mock_inference_result

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("minimal_data", 500000)

            # Process minimal request
            response = self.multi_handler.process_request(minimal_request)

            # Should succeed with defaults applied
            self.assertEqual(response['status'], 'success')

            # Verify defaults were used
            inference_args = mock_flux_model.infer.call_args[0][0]
            self.assertEqual(inference_args['prompt'], 'A')
            self.assertEqual(inference_args['width'], 1024)  # Default
            self.assertEqual(inference_args['height'], 1024)  # Default
            self.assertEqual(inference_args['num_inference_steps'], 4)  # Default


if __name__ == '__main__':
    unittest.main()