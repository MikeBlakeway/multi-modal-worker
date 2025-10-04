"""
Integration tests for ControlNet workflow.

Tests end-to-end ControlNet guided image generation workflows
including control image processing and model integration.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import base64
from PIL import Image
import io
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from handlers.controlnet_handler import ControlNetHandler
from models.controlnet_model import ControlNetModel
from utils.control_processors import CannyProcessor, DepthProcessor
from schemas.controlnet_schema import ControlNetRequest
from models.model_manager import ModelManager
from utils.exceptions import ValidationError, InferenceError


class TestControlNetWorkflow(unittest.TestCase):
    """Test cases for complete ControlNet workflow integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.test_control_image = self._create_test_image(256, 256, 'red')
        self.test_control_image_b64 = self._image_to_base64(self.test_control_image)

        # Valid test requests
        self.canny_request = {
            'prompt': 'A beautiful landscape with mountains',
            'control_image': self.test_control_image_b64,
            'control_type': 'canny',
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'control_strength': 1.0,
            'canny_low_threshold': 100,
            'canny_high_threshold': 200
        }

        self.depth_request = self.canny_request.copy()
        self.depth_request['control_type'] = 'depth'

    def _create_test_image(self, width, height, color):
        """Create a test image."""
        return Image.new('RGB', (width, height), color=color)

    def _image_to_base64(self, image):
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @patch('models.controlnet_model.StableDiffusionControlNetPipeline')
    @patch('models.controlnet_model.ControlNetModel')
    @patch('cv2.Canny')
    @patch('cv2.cvtColor')
    def test_canny_controlnet_full_workflow(self, mock_cvtColor, mock_Canny,
                                           mock_controlnet_class, mock_pipeline_class):
        """Test complete Canny ControlNet workflow."""
        # Mock Canny edge detection
        gray_image = np.zeros((256, 256), dtype=np.uint8)
        edges = np.zeros((256, 256), dtype=np.uint8)
        edges[50:150, 50:52] = 255  # Vertical edge

        mock_cvtColor.side_effect = [gray_image, np.stack([edges, edges, edges], axis=-1)]
        mock_Canny.return_value = edges

        # Mock ControlNet model and pipeline
        mock_controlnet = Mock()
        mock_controlnet_class.from_pretrained.return_value = mock_controlnet

        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [self._create_test_image(512, 512, 'blue')]
        mock_pipeline.return_value = mock_result
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Create ControlNet model
        controlnet_model = ControlNetModel(['canny'])
        controlnet_model._shared_vae = Mock()
        controlnet_model._shared_text_encoder = Mock()
        controlnet_model._shared_tokenizer = Mock()
        controlnet_model._shared_scheduler = Mock()

        # Mock model loading
        with patch.object(controlnet_model, '_load_shared_components'):
            controlnet_model.controlnets = {'canny': mock_controlnet}
            controlnet_model.pipelines = {'canny': mock_pipeline}

        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.get_model.return_value = controlnet_model

        # Initialize handler and process request
        handler = ControlNetHandler(['canny'])
        result = handler.process_request(self.canny_request, mock_model_manager)

        # Verify successful result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['images']), 1)
        self.assertIn('inference_time_ms', result)
        self.assertIn('preprocessing_time_ms', result)

        # Verify image output structure
        image_output = result['images'][0]
        self.assertIn('image', image_output)
        self.assertIn('control_info', image_output)
        self.assertEqual(image_output['control_info']['control_type'], 'canny')

        # Verify Canny processing was called
        mock_Canny.assert_called_once_with(gray_image, 100, 200)

    @patch('torch.hub.load')
    @patch('torch.no_grad')
    @patch('torch.nn.functional.interpolate')
    @patch('models.controlnet_model.StableDiffusionControlNetPipeline')
    @patch('models.controlnet_model.ControlNetModel')
    def test_depth_controlnet_full_workflow(self, mock_controlnet_class, mock_pipeline_class,
                                          mock_interpolate, mock_no_grad, mock_hub_load):
        """Test complete depth ControlNet workflow."""
        # Mock MiDaS depth estimation
        mock_midas_model = Mock()
        mock_transform = Mock()
        mock_hub_load.side_effect = [mock_midas_model, {'small_transform': mock_transform}]

        # Mock depth prediction
        depth_map = np.random.rand(256, 256).astype(np.float32)
        mock_prediction = Mock()
        mock_prediction.unsqueeze.return_value = mock_prediction
        mock_prediction.squeeze.return_value = mock_prediction
        mock_prediction.cpu.return_value.numpy.return_value = depth_map

        mock_midas_model.return_value = mock_prediction
        mock_transform.return_value.to.return_value = Mock()
        mock_interpolate.return_value = mock_prediction

        # Mock ControlNet model and pipeline
        mock_controlnet = Mock()
        mock_controlnet_class.from_pretrained.return_value = mock_controlnet

        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [self._create_test_image(512, 512, 'green')]
        mock_pipeline.return_value = mock_result
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Create ControlNet model
        controlnet_model = ControlNetModel(['depth'])
        controlnet_model._shared_vae = Mock()
        controlnet_model._shared_text_encoder = Mock()
        controlnet_model._shared_tokenizer = Mock()
        controlnet_model._shared_scheduler = Mock()

        # Mock model loading
        with patch.object(controlnet_model, '_load_shared_components'):
            controlnet_model.controlnets = {'depth': mock_controlnet}
            controlnet_model.pipelines = {'depth': mock_pipeline}

        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.get_model.return_value = controlnet_model

        # Initialize handler and process request
        handler = ControlNetHandler(['depth'])
        result = handler.process_request(self.depth_request, mock_model_manager)

        # Verify successful result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['images']), 1)
        self.assertIn('inference_time_ms', result)
        self.assertIn('preprocessing_time_ms', result)

        # Verify image output structure
        image_output = result['images'][0]
        self.assertEqual(image_output['control_info']['control_type'], 'depth')

        # Verify depth processing was performed
        self.assertGreater(image_output['control_info']['preprocessing_time_ms'], 0)

    def test_invalid_control_image_workflow(self):
        """Test workflow with invalid control image."""
        invalid_request = self.canny_request.copy()
        invalid_request['control_image'] = 'invalid_base64'

        handler = ControlNetHandler(['canny'])
        mock_model_manager = Mock()

        result = handler.process_request(invalid_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(result['error_type'], 'validationerror')

    def test_unsupported_control_type_workflow(self):
        """Test workflow with unsupported control type."""
        invalid_request = self.canny_request.copy()
        invalid_request['control_type'] = 'invalid'

        handler = ControlNetHandler(['canny'])
        mock_model_manager = Mock()

        result = handler.process_request(invalid_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    @patch('models.controlnet_model.StableDiffusionControlNetPipeline')
    @patch('models.controlnet_model.ControlNetModel')
    def test_model_loading_failure_workflow(self, mock_controlnet_class, mock_pipeline_class):
        """Test workflow with model loading failure."""
        # Mock model manager that fails to load
        mock_model_manager = Mock()
        mock_model_manager.get_model.side_effect = Exception("Model load failed")

        handler = ControlNetHandler(['canny'])
        result = handler.process_request(self.canny_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    @patch('cv2.Canny')
    @patch('cv2.cvtColor')
    def test_control_processing_failure_workflow(self, mock_cvtColor, mock_Canny):
        """Test workflow with control processing failure."""
        # Mock Canny to raise exception
        mock_Canny.side_effect = Exception("OpenCV error")

        # Mock model manager with valid model
        mock_model_manager = Mock()
        mock_controlnet_model = Mock()
        mock_controlnet_model.generate_image.side_effect = Exception("Processing failed")
        mock_model_manager.get_model.return_value = mock_controlnet_model

        handler = ControlNetHandler(['canny'])
        result = handler.process_request(self.canny_request, mock_model_manager)

        # Verify error response
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    @patch('models.controlnet_model.StableDiffusionControlNetPipeline')
    @patch('models.controlnet_model.ControlNetModel')
    @patch('cv2.Canny')
    @patch('cv2.cvtColor')
    def test_performance_tracking_workflow(self, mock_cvtColor, mock_Canny,
                                         mock_controlnet_class, mock_pipeline_class):
        """Test performance tracking during workflow."""
        # Mock successful processing
        gray_image = np.zeros((256, 256), dtype=np.uint8)
        edges = np.zeros((256, 256), dtype=np.uint8)

        mock_cvtColor.side_effect = [gray_image, np.stack([edges, edges, edges], axis=-1)]
        mock_Canny.return_value = edges

        mock_controlnet = Mock()
        mock_controlnet_class.from_pretrained.return_value = mock_controlnet

        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [self._create_test_image(512, 512, 'yellow')]
        mock_pipeline.return_value = mock_result
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Create mock model
        controlnet_model = ControlNetModel(['canny'])
        controlnet_model._shared_vae = Mock()
        controlnet_model._shared_text_encoder = Mock()
        controlnet_model._shared_tokenizer = Mock()
        controlnet_model._shared_scheduler = Mock()

        with patch.object(controlnet_model, '_load_shared_components'):
            controlnet_model.controlnets = {'canny': mock_controlnet}
            controlnet_model.pipelines = {'canny': mock_pipeline}

        mock_model_manager = Mock()
        mock_model_manager.get_model.return_value = controlnet_model

        # Initialize handler
        handler = ControlNetHandler(['canny'])
        initial_stats = handler.get_performance_stats()

        # Process request
        result = handler.process_request(self.canny_request, mock_model_manager)

        # Verify performance tracking
        self.assertTrue(result['success'])

        final_stats = handler.get_performance_stats()
        self.assertEqual(final_stats['successful_inferences'],
                        initial_stats['successful_inferences'] + 1)
        self.assertGreater(final_stats['total_processing_time_s'],
                          initial_stats['total_processing_time_s'])

    def test_multiple_control_types_handler_workflow(self):
        """Test handler supporting multiple control types."""
        # Test that handler can be initialized with multiple types
        handler = ControlNetHandler(['canny', 'depth'])

        self.assertEqual(handler.control_types, ['canny', 'depth'])

        # Test that handler info includes all types
        info = handler.get_handler_info()
        self.assertEqual(info['control_types'], ['canny', 'depth'])

        # Test required models includes all types
        models = handler.get_required_models()
        self.assertEqual(models, ['controlnet-canny-depth'])

    @patch('cv2.Canny')
    @patch('cv2.cvtColor')
    def test_parameter_validation_integration(self, mock_cvtColor, mock_Canny):
        """Test parameter validation integration in workflow."""
        # Test various parameter combinations
        test_cases = [
            # Valid parameters
            {
                'control_strength': 0.8,
                'control_guidance_start': 0.2,
                'control_guidance_end': 0.9,
                'expected_valid': True
            },
            # Invalid guidance range
            {
                'control_guidance_start': 0.8,
                'control_guidance_end': 0.5,
                'expected_valid': False
            },
            # Invalid Canny thresholds
            {
                'canny_low_threshold': 200,
                'canny_high_threshold': 150,
                'expected_valid': False
            }
        ]

        handler = ControlNetHandler(['canny'])

        for case in test_cases:
            test_request = self.canny_request.copy()
            test_request.update({k: v for k, v in case.items() if k != 'expected_valid'})

            try:
                validated = handler.validate_request(test_request)
                is_valid = True
            except ValidationError:
                is_valid = False

            self.assertEqual(is_valid, case['expected_valid'],
                           f"Parameter validation failed for case: {case}")


class TestControlNetPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for ControlNet workflows."""

    @unittest.skip("Performance test - run manually")
    def test_inference_time_benchmark(self):
        """Benchmark ControlNet inference times."""
        # This would be a real performance test
        # Skip by default but available for manual execution
        pass

    def test_processing_time_estimation_accuracy(self):
        """Test accuracy of processing time estimation."""
        handler = ControlNetHandler(['canny', 'depth'])

        # Test estimation for different configurations
        base_request = {
            'prompt': 'test',
            'control_image': 'dummy',
            'control_type': 'canny',
            'width': 512,
            'height': 512,
            'num_inference_steps': 20
        }

        # Test different image sizes
        for width, height in [(512, 512), (768, 768), (1024, 1024)]:
            test_request = base_request.copy()
            test_request.update({'width': width, 'height': height})

            estimated_time = handler.estimate_processing_time(test_request)

            # Larger images should take proportionally longer
            # (This is a basic sanity check)
            self.assertGreater(estimated_time, 10.0)  # At least 10 seconds
            self.assertLess(estimated_time, 60.0)     # Less than 1 minute

        # Test different control types
        for control_type in ['canny', 'depth']:
            test_request = base_request.copy()
            test_request['control_type'] = control_type

            estimated_time = handler.estimate_processing_time(test_request)
            self.assertGreater(estimated_time, 15.0)  # Should be within reasonable range
            self.assertLess(estimated_time, 25.0)


if __name__ == '__main__':
    # Run all tests except performance benchmarks by default
    unittest.main(verbosity=2)