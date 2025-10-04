"""
Unit tests for ControlNet control processors.

Tests the Canny edge detection and depth estimation preprocessing
utilities for ControlNet guidance.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import base64
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.control_processors import (
    ControlProcessor, CannyProcessor, DepthProcessor,
    ControlProcessorFactory, process_control_image
)
from src.utils.exceptions import ValidationError, ProcessingError


class TestControlProcessor(unittest.TestCase):
    """Test cases for base ControlProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = ControlProcessor()

        # Create test images
        self.test_image_pil = Image.new('RGB', (256, 256), color='red')

        # Create numpy array version
        self.test_image_np = np.array(self.test_image_pil)

        # Create base64 version
        buffer = io.BytesIO()
        self.test_image_pil.save(buffer, format='PNG')
        self.test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    def test_prepare_image_pil(self):
        """Test image preparation from PIL Image."""
        result = self.processor._prepare_image(self.test_image_pil)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, (256, 256))

    def test_prepare_image_numpy(self):
        """Test image preparation from numpy array."""
        result = self.processor._prepare_image(self.test_image_np)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, (256, 256))

    def test_prepare_image_base64(self):
        """Test image preparation from base64 string."""
        result = self.processor._prepare_image(self.test_image_b64)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, (256, 256))

    def test_prepare_image_invalid_base64(self):
        """Test image preparation with invalid base64."""
        with self.assertRaises(ValidationError):
            self.processor._prepare_image('invalid_base64')

    def test_prepare_image_unsupported_type(self):
        """Test image preparation with unsupported type."""
        with self.assertRaises(ValidationError):
            self.processor._prepare_image(123)

    def test_prepare_image_rgba_conversion(self):
        """Test RGBA to RGB conversion."""
        rgba_image = Image.new('RGBA', (256, 256), color=(255, 0, 0, 128))
        result = self.processor._prepare_image(rgba_image)

        self.assertEqual(result.mode, 'RGB')

    def test_process_not_implemented(self):
        """Test that base process method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.processor.process(self.test_image_pil)


class TestCannyProcessor(unittest.TestCase):
    """Test cases for CannyProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CannyProcessor()

        # Create test image with some structure for edge detection
        self.test_image = self._create_structured_image()

    def _create_structured_image(self):
        """Create a test image with edges for Canny detection."""
        # Create image with a white square on black background
        image_array = np.zeros((256, 256, 3), dtype=np.uint8)
        image_array[64:192, 64:192] = [255, 255, 255]  # White square
        return Image.fromarray(image_array)

    @patch('cv2.Canny')
    @patch('cv2.cvtColor')
    def test_process_canny_success(self, mock_cvtColor, mock_Canny):
        """Test successful Canny edge detection."""
        # Mock OpenCV functions
        gray_image = np.zeros((256, 256), dtype=np.uint8)
        edges = np.zeros((256, 256), dtype=np.uint8)
        edges[64:192, 64:66] = 255  # Left edge
        edges[64:192, 190:192] = 255  # Right edge

        mock_cvtColor.side_effect = [gray_image, np.stack([edges, edges, edges], axis=-1)]
        mock_Canny.return_value = edges

        # Process image
        result_image, info = self.processor.process(
            self.test_image,
            low_threshold=100,
            high_threshold=200
        )

        # Verify result
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (256, 256))
        self.assertEqual(info['control_type'], 'canny')
        self.assertEqual(info['low_threshold'], 100)
        self.assertEqual(info['high_threshold'], 200)
        self.assertIn('processing_time_ms', info)
        self.assertIn('edge_pixels', info)
        self.assertIn('edge_density', info)

        # Verify OpenCV was called correctly
        mock_Canny.assert_called_once_with(gray_image, 100, 200)

    def test_process_canny_default_params(self):
        """Test Canny processing with default parameters."""
        with patch('cv2.Canny') as mock_Canny, \
             patch('cv2.cvtColor') as mock_cvtColor:

            # Mock OpenCV functions
            gray_image = np.zeros((256, 256), dtype=np.uint8)
            edges = np.zeros((256, 256), dtype=np.uint8)

            mock_cvtColor.side_effect = [gray_image, np.stack([edges, edges, edges], axis=-1)]
            mock_Canny.return_value = edges

            # Process with defaults
            result_image, info = self.processor.process(self.test_image)

            # Verify default thresholds were used
            self.assertEqual(info['low_threshold'], self.processor.DEFAULT_LOW_THRESHOLD)
            self.assertEqual(info['high_threshold'], self.processor.DEFAULT_HIGH_THRESHOLD)
            mock_Canny.assert_called_once_with(gray_image, 100, 200)

    def test_process_canny_invalid_low_threshold(self):
        """Test Canny processing with invalid low threshold."""
        with self.assertRaises(ValidationError) as context:
            self.processor.process(self.test_image, low_threshold=0)

        self.assertIn('must be between 1-255', str(context.exception))

    def test_process_canny_invalid_high_threshold(self):
        """Test Canny processing with invalid high threshold."""
        with self.assertRaises(ValidationError) as context:
            self.processor.process(self.test_image, high_threshold=300)

        self.assertIn('must be between 1-255', str(context.exception))

    def test_process_canny_threshold_order_error(self):
        """Test Canny processing with high threshold <= low threshold."""
        with self.assertRaises(ValidationError) as context:
            self.processor.process(self.test_image, low_threshold=150, high_threshold=100)

        self.assertIn('high_threshold must be > low_threshold', str(context.exception))

    @patch('cv2.Canny', side_effect=Exception("OpenCV error"))
    def test_process_canny_opencv_error(self, mock_Canny):
        """Test Canny processing with OpenCV error."""
        with self.assertRaises(ProcessingError):
            self.processor.process(self.test_image)


class TestDepthProcessor(unittest.TestCase):
    """Test cases for DepthProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DepthProcessor()
        self.test_image = Image.new('RGB', (256, 256), color='red')

    @patch('torch.hub.load')
    @patch('torch.device')
    def test_load_midas_model_success(self, mock_device, mock_hub_load):
        """Test successful MiDaS model loading."""
        # Mock torch components
        mock_device.return_value = 'cpu'
        mock_model = Mock()
        mock_transform = Mock()

        mock_hub_load.side_effect = [mock_model, {'small_transform': mock_transform}]

        # Load model
        self.processor._load_midas_model()

        # Verify model was loaded
        self.assertIsNotNone(self.processor._midas_model)
        self.assertIsNotNone(self.processor._midas_transform)
        self.assertEqual(self.processor._device, 'cpu')

    @patch('torch.hub.load', side_effect=Exception("Model load error"))
    def test_load_midas_model_failure(self, mock_hub_load):
        """Test MiDaS model loading failure."""
        with self.assertRaises(ProcessingError):
            self.processor._load_midas_model()

    @patch('torch.no_grad')
    @patch('torch.nn.functional.interpolate')
    def test_process_depth_success(self, mock_interpolate, mock_no_grad):
        """Test successful depth processing."""
        # Mock MiDaS model and components
        self.processor._midas_model = Mock()
        self.processor._midas_transform = Mock()
        self.processor._device = 'cpu'

        # Mock depth prediction
        mock_prediction = Mock()
        mock_prediction.unsqueeze.return_value = mock_prediction
        mock_prediction.squeeze.return_value = mock_prediction
        mock_prediction.cpu.return_value.numpy.return_value = np.ones((256, 256)) * 0.5

        self.processor._midas_model.return_value = mock_prediction
        self.processor._midas_transform.return_value.to.return_value = Mock()
        mock_interpolate.return_value = mock_prediction

        # Process image
        result_image, info = self.processor.process(
            self.test_image,
            normalize_depth=True,
            invert_depth=True
        )

        # Verify result
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_image.size, (256, 256))
        self.assertEqual(info['control_type'], 'depth')
        self.assertTrue(info['normalize_depth'])
        self.assertTrue(info['invert_depth'])
        self.assertIn('processing_time_ms', info)
        self.assertIn('depth_range', info)

    @patch('torch.no_grad')
    def test_process_depth_model_not_loaded(self, mock_no_grad):
        """Test depth processing when model not loaded (should trigger loading)."""
        with patch.object(self.processor, '_load_midas_model') as mock_load:
            mock_load.side_effect = ProcessingError("load_operation", "Load failed")

            with self.assertRaises(ProcessingError):
                self.processor.process(self.test_image)

    def test_process_depth_default_params(self):
        """Test depth processing with default parameters."""
        with patch.object(self.processor, '_load_midas_model'), \
             patch('torch.no_grad'), \
             patch('torch.nn.functional.interpolate'):

            # Mock components
            self.processor._midas_model = Mock()
            self.processor._midas_transform = Mock()
            self.processor._device = 'cpu'

            mock_prediction = Mock()
            mock_prediction.unsqueeze.return_value = mock_prediction
            mock_prediction.squeeze.return_value = mock_prediction
            mock_prediction.cpu.return_value.numpy.return_value = np.ones((256, 256)) * 0.5

            self.processor._midas_model.return_value = mock_prediction
            self.processor._midas_transform.return_value.to.return_value = Mock()

            # Process with defaults
            result_image, info = self.processor.process(self.test_image)

            # Verify defaults
            self.assertTrue(info['normalize_depth'])
            self.assertTrue(info['invert_depth'])


class TestControlProcessorFactory(unittest.TestCase):
    """Test cases for ControlProcessorFactory class."""

    def test_create_processor_canny(self):
        """Test creating Canny processor."""
        processor = ControlProcessorFactory.create_processor('canny')
        self.assertIsInstance(processor, CannyProcessor)

    def test_create_processor_depth(self):
        """Test creating depth processor."""
        processor = ControlProcessorFactory.create_processor('depth')
        self.assertIsInstance(processor, DepthProcessor)

    def test_create_processor_invalid_type(self):
        """Test creating processor with invalid type."""
        with self.assertRaises(Exception) as context:
            ControlProcessorFactory.create_processor('invalid')

        self.assertIn("Unsupported control type 'invalid'", str(context.exception))

    def test_get_supported_types(self):
        """Test getting supported processor types."""
        types = ControlProcessorFactory.get_supported_types()
        self.assertIn('canny', types)
        self.assertIn('depth', types)

    def test_register_processor(self):
        """Test registering new processor type."""
        class CustomProcessor(ControlProcessor):
            def process(self, image, **kwargs):
                pass

        # Register processor
        ControlProcessorFactory.register_processor('custom', CustomProcessor)

        # Verify it can be created
        processor = ControlProcessorFactory.create_processor('custom')
        self.assertIsInstance(processor, CustomProcessor)

        # Verify it appears in supported types
        types = ControlProcessorFactory.get_supported_types()
        self.assertIn('custom', types)

    def test_register_processor_invalid_class(self):
        """Test registering processor with invalid class."""
        class InvalidProcessor:
            pass

        with self.assertRaises(ValueError):
            ControlProcessorFactory.register_processor('invalid', InvalidProcessor)


class TestProcessControlImageFunction(unittest.TestCase):
    """Test cases for process_control_image convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.new('RGB', (256, 256), color='red')

    @patch('utils.control_processors.ControlProcessorFactory.create_processor')
    def test_process_control_image_canny(self, mock_create_processor):
        """Test process_control_image function with Canny."""
        mock_processor = Mock()
        mock_result = (self.test_image, {'control_type': 'canny'})
        mock_processor.process.return_value = mock_result
        mock_create_processor.return_value = mock_processor

        result = process_control_image(
            self.test_image,
            'canny',
            low_threshold=100,
            high_threshold=200
        )

        self.assertEqual(result, mock_result)
        mock_processor.process.assert_called_once_with(
            self.test_image,
            low_threshold=100,
            high_threshold=200
        )

    def test_process_control_image_invalid_type(self):
        """Test process_control_image function with invalid type."""
        with self.assertRaises(ValidationError):
            process_control_image(self.test_image, 'invalid')


if __name__ == '__main__':
    unittest.main()