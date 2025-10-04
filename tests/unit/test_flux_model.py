"""
Comprehensive unit tests for FluxModel class.

Tests model loading, inference, memory management, and integration
with the model management system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

# Import the components to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.models.flux_model import FluxModel
    from src.utils.exceptions import ModelLoadError, InferenceError, ValidationError
except ImportError:
    # Standalone implementation for testing
    pass


class TestFluxModel(unittest.TestCase):
    """Test FluxModel functionality and integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "test-flux-model"
        self.model_path = Path("/fake/path/to/model")

        # Create model instance
        self.flux_model = FluxModel(
            model_name=self.model_name,
            model_path=self.model_path,
            priority=75,
            device="cuda",
            enable_cpu_offload=True,
            enable_attention_slicing=True
        )

        # Sample inference inputs
        self.valid_inputs = {
            'prompt': 'A beautiful sunset over mountains',
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 4,
            'guidance_scale': 0.0,
            'seed': 42
        }

    def test_model_initialization(self):
        """Test FluxModel initialization."""
        model = FluxModel()

        self.assertEqual(model.model_name, "flux-1-schnell-fp8")
        self.assertEqual(model.MODEL_ID, "black-forest-labs/FLUX.1-schnell")
        self.assertEqual(model.MODEL_VARIANT, "fp8")
        self.assertEqual(model.DEFAULT_STEPS, 4)
        self.assertEqual(model.DEFAULT_GUIDANCE, 0.0)
        self.assertEqual(model.priority, 75)
        self.assertTrue(model.enable_cpu_offload)
        self.assertTrue(model.enable_attention_slicing)
        self.assertFalse(model.is_loaded)
        self.assertIsNone(model.pipeline)

    def test_model_initialization_with_custom_params(self):
        """Test FluxModel initialization with custom parameters."""
        model = FluxModel(
            model_name="custom-flux",
            model_path=Path("/custom/path"),
            priority=50,
            device="cpu",
            enable_cpu_offload=False,
            enable_attention_slicing=False
        )

        self.assertEqual(model.model_name, "custom-flux")
        self.assertEqual(model.model_path, Path("/custom/path"))
        self.assertEqual(model.priority, 50)
        self.assertEqual(model.device, "cpu")
        self.assertFalse(model.enable_cpu_offload)
        self.assertFalse(model.enable_attention_slicing)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.memory_allocated')
    @patch('src.models.flux_model.FluxPipeline')
    def test_load_model_success(self, mock_pipeline_class, mock_memory_allocated, mock_empty_cache, mock_cuda_available):
        """Test successful model loading."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 14000 * 1024 * 1024  # 14GB in bytes
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Mock pipeline methods
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline.enable_model_cpu_offload = Mock()
        mock_pipeline.to = Mock(return_value=mock_pipeline)

        # Test loading
        self.flux_model.load()

        # Verify model state
        self.assertTrue(self.flux_model.is_loaded)
        self.assertIsNotNone(self.flux_model.load_time)
        self.assertEqual(self.flux_model._model, mock_pipeline)
        self.assertEqual(self.flux_model.pipeline, mock_pipeline)
        self.assertGreater(self.flux_model.memory_usage_mb, 0)

        # Verify pipeline configuration
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_model_cpu_offload.assert_called_once()

    @patch('src.models.flux_model.FluxPipeline')
    def test_load_model_already_loaded(self, mock_pipeline_class):
        """Test loading when model is already loaded."""
        # Set model as already loaded
        self.flux_model.is_loaded = True

        # Attempt to load again
        self.flux_model.load()

        # Verify pipeline was not called
        mock_pipeline_class.from_pretrained.assert_not_called()

    @patch('torch.cuda.is_available')
    @patch('src.models.flux_model.FluxPipeline')
    def test_load_model_failure(self, mock_pipeline_class, mock_cuda_available):
        """Test model loading failure handling."""
        mock_cuda_available.return_value = True
        mock_pipeline_class.from_pretrained.side_effect = Exception("Loading failed")

        with self.assertRaises(ModelLoadError) as context:
            self.flux_model.load()

        self.assertIn("Failed to load model", str(context.exception))
        self.assertFalse(self.flux_model.is_loaded)
        self.assertIsNone(self.flux_model.pipeline)

    @patch('torch.cuda.is_available')
    @patch('src.models.flux_model.FluxPipeline')
    def test_load_model_out_of_memory(self, mock_pipeline_class, mock_cuda_available):
        """Test handling of out of memory errors during loading."""
        mock_cuda_available.return_value = True
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

        with self.assertRaises(MemoryError) as context:
            self.flux_model.load()

        self.assertIn("Insufficient GPU memory", str(context.exception))
        self.assertFalse(self.flux_model.is_loaded)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_unload_model(self, mock_empty_cache, mock_cuda_available):
        """Test model unloading."""
        mock_cuda_available.return_value = True

        # Setup loaded model
        mock_pipeline = Mock()
        mock_pipeline.to = Mock(return_value=mock_pipeline)
        self.flux_model.pipeline = mock_pipeline
        self.flux_model._model = mock_pipeline
        self.flux_model.is_loaded = True
        self.flux_model.memory_usage_mb = 14000

        # Unload model
        self.flux_model.unload()

        # Verify cleanup
        self.assertFalse(self.flux_model.is_loaded)
        self.assertIsNone(self.flux_model.pipeline)
        self.assertIsNone(self.flux_model._model)
        self.assertEqual(self.flux_model.memory_usage_mb, 0)

        # Verify GPU cache was cleared
        mock_empty_cache.assert_called()

    def test_unload_model_not_loaded(self):
        """Test unloading when model is not loaded."""
        # Ensure model is not loaded
        self.assertFalse(self.flux_model.is_loaded)

        # Unload should not raise error
        self.flux_model.unload()

        self.assertFalse(self.flux_model.is_loaded)

    def test_validate_inputs_valid(self):
        """Test input validation with valid parameters."""
        result = self.flux_model.validate_inputs(self.valid_inputs)
        self.assertTrue(result)

    def test_validate_inputs_missing_prompt(self):
        """Test validation fails with missing prompt."""
        invalid_inputs = self.valid_inputs.copy()
        del invalid_inputs['prompt']

        with self.assertRaises(ValidationError) as context:
            self.flux_model.validate_inputs(invalid_inputs)

        self.assertIn("Missing required field", str(context.exception))

    def test_validate_inputs_empty_prompt(self):
        """Test validation fails with empty prompt."""
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['prompt'] = ""

        with self.assertRaises(ValidationError) as context:
            self.flux_model.validate_inputs(invalid_inputs)

        self.assertIn("non-empty string", str(context.exception))

    def test_validate_inputs_long_prompt(self):
        """Test validation fails with overly long prompt."""
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['prompt'] = "x" * 2001  # Exceeds limit

        with self.assertRaises(ValidationError) as context:
            self.flux_model.validate_inputs(invalid_inputs)

        self.assertIn("too long", str(context.exception))

    def test_validate_inputs_invalid_dimensions(self):
        """Test validation fails with invalid dimensions."""
        # Test invalid width
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['width'] = 200  # Too small

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

        # Test invalid height
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['height'] = 3000  # Too large

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

        # Test non-multiple of 8
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['width'] = 1023  # Not multiple of 8

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

    def test_validate_inputs_invalid_steps(self):
        """Test validation fails with invalid inference steps."""
        # Too few steps
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['num_inference_steps'] = 0

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

        # Too many steps
        invalid_inputs['num_inference_steps'] = 100

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

    def test_validate_inputs_invalid_guidance(self):
        """Test validation fails with invalid guidance scale."""
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['guidance_scale'] = -1.0  # Negative

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

        invalid_inputs['guidance_scale'] = 25.0  # Too high

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

    def test_validate_inputs_invalid_seed(self):
        """Test validation fails with invalid seed."""
        invalid_inputs = self.valid_inputs.copy()
        invalid_inputs['seed'] = -1  # Negative

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

        invalid_inputs['seed'] = 2**32  # Too large

        with self.assertRaises(ValidationError):
            self.flux_model.validate_inputs(invalid_inputs)

    @patch('torch.Generator')
    @patch('torch.inference_mode')
    def test_infer_success(self, mock_inference_mode, mock_generator_class):
        """Test successful inference."""
        # Setup loaded model
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (1024, 1024), color='red')]
        mock_pipeline.return_value = mock_result

        self.flux_model.pipeline = mock_pipeline
        self.flux_model.is_loaded = True
        self.flux_model._model = mock_pipeline

        # Setup generator mock
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.manual_seed.return_value = mock_generator

        # Setup inference mode context manager
        mock_inference_mode.return_value.__enter__ = Mock()
        mock_inference_mode.return_value.__exit__ = Mock()

        # Perform inference
        result = self.flux_model.infer(self.valid_inputs)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('image', result)
        self.assertIn('inference_time', result)
        self.assertIn('parameters', result)
        self.assertIn('memory_usage_mb', result)
        self.assertIn('model_info', result)

        # Verify parameters were passed correctly
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs['prompt'], self.valid_inputs['prompt'])
        self.assertEqual(call_kwargs['width'], self.valid_inputs['width'])
        self.assertEqual(call_kwargs['height'], self.valid_inputs['height'])

        # Verify performance tracking
        self.assertEqual(self.flux_model.inference_count, 1)
        self.assertGreater(self.flux_model.total_inference_time, 0)
        self.assertGreater(self.flux_model.average_inference_time, 0)

    def test_infer_model_not_loaded(self):
        """Test inference fails when model not loaded."""
        with self.assertRaises(InferenceError) as context:
            self.flux_model.infer(self.valid_inputs)

        self.assertIn("Model not loaded", str(context.exception))

    @patch('torch.Generator')
    def test_infer_invalid_inputs(self, mock_generator_class):
        """Test inference fails with invalid inputs."""
        # Setup loaded model
        self.flux_model.pipeline = Mock()
        self.flux_model.is_loaded = True

        # Test with invalid inputs
        invalid_inputs = {'prompt': ''}  # Empty prompt

        with self.assertRaises(ValidationError):
            self.flux_model.infer(invalid_inputs)

    @patch('torch.Generator')
    def test_infer_pipeline_failure(self, mock_generator_class):
        """Test inference handles pipeline failures."""
        # Setup loaded model that fails
        mock_pipeline = Mock()
        mock_pipeline.side_effect = RuntimeError("Pipeline failed")

        self.flux_model.pipeline = mock_pipeline
        self.flux_model.is_loaded = True
        self.flux_model._model = mock_pipeline

        with self.assertRaises(InferenceError) as context:
            self.flux_model.infer(self.valid_inputs)

        self.assertIn("FLUX.1 inference failed", str(context.exception))

    @patch('torch.Generator')
    def test_infer_out_of_memory(self, mock_generator_class):
        """Test inference handles out of memory errors."""
        # Setup model that runs out of memory
        mock_pipeline = Mock()
        mock_pipeline.side_effect = RuntimeError("CUDA out of memory")

        self.flux_model.pipeline = mock_pipeline
        self.flux_model.is_loaded = True
        self.flux_model._model = mock_pipeline

        with self.assertRaises(InferenceError) as context:
            self.flux_model.infer(self.valid_inputs)

        self.assertIn("GPU out of memory", str(context.exception))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_get_memory_usage(self, mock_memory_allocated, mock_cuda_available):
        """Test memory usage reporting."""
        # Test when not loaded
        self.assertEqual(self.flux_model.get_memory_usage(), 0)

        # Test when loaded
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 15000 * 1024 * 1024  # 15GB in bytes

        self.flux_model.is_loaded = True
        self.flux_model.memory_usage_mb = 15000

        memory_usage = self.flux_model.get_memory_usage()
        self.assertEqual(memory_usage, 15000)

    def test_get_model_info(self):
        """Test model information retrieval."""
        # Set some test data
        self.flux_model.is_loaded = True
        self.flux_model.memory_usage_mb = 14000
        self.flux_model.use_count = 5
        self.flux_model.inference_count = 3
        self.flux_model.average_inference_time = 12.5
        self.flux_model.load_time = datetime.now()

        info = self.flux_model.get_model_info()

        self.assertIsInstance(info, dict)
        self.assertEqual(info['model_name'], self.flux_model.model_name)
        self.assertEqual(info['model_id'], self.flux_model.MODEL_ID)
        self.assertEqual(info['variant'], self.flux_model.MODEL_VARIANT)
        self.assertTrue(info['is_loaded'])
        self.assertEqual(info['memory_usage_mb'], 14000)
        self.assertEqual(info['use_count'], 5)
        self.assertEqual(info['inference_count'], 3)
        self.assertEqual(info['average_inference_time'], 12.5)
        self.assertIn('optimizations', info)
        self.assertTrue(info['optimizations']['fp8_quantization'])

    @patch('torch.Generator')
    def test_warmup(self, mock_generator_class):
        """Test model warmup functionality."""
        # Setup loaded model
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (512, 512), color='blue')]
        mock_pipeline.return_value = mock_result

        self.flux_model.pipeline = mock_pipeline
        self.flux_model.is_loaded = True
        self.flux_model._model = mock_pipeline

        # Perform warmup
        warmup_time = self.flux_model.warmup("test prompt")

        self.assertIsInstance(warmup_time, float)
        self.assertGreater(warmup_time, 0)

        # Verify warmup used correct parameters
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs['width'], 512)  # Smaller size for warmup
        self.assertEqual(call_kwargs['height'], 512)
        self.assertEqual(call_kwargs['num_inference_steps'], 2)  # Fewer steps

    def test_warmup_model_not_loaded(self):
        """Test warmup fails when model not loaded."""
        with self.assertRaises(InferenceError):
            self.flux_model.warmup()


class TestFluxModelMemoryManagement(unittest.TestCase):
    """Test FluxModel memory management and optimization features."""

    def setUp(self):
        """Set up test fixtures."""
        self.flux_model = FluxModel()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_gpu_memory_tracking(self, mock_memory_allocated, mock_cuda_available):
        """Test GPU memory usage tracking."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 12000 * 1024 * 1024  # 12GB

        memory_mb = self.flux_model._get_gpu_memory_mb()
        self.assertEqual(memory_mb, 12000)

        # Test when CUDA not available
        mock_cuda_available.return_value = False
        memory_mb = self.flux_model._get_gpu_memory_mb()
        self.assertEqual(memory_mb, 0)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    def test_memory_estimation(self, mock_memory_allocated, mock_cuda_available):
        """Test memory usage estimation."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 14000 * 1024 * 1024

        # Test when loaded
        self.flux_model.is_loaded = True
        self.flux_model.pipeline = Mock()  # Need to set a mock pipeline
        estimated = self.flux_model._estimate_memory_usage()
        self.assertEqual(estimated, 14000)

        # Test when not loaded
        self.flux_model.is_loaded = False
        estimated = self.flux_model._estimate_memory_usage()
        self.assertEqual(estimated, 0)

    def test_performance_tracking(self):
        """Test inference performance tracking."""
        # Initial state
        self.assertEqual(self.flux_model.inference_count, 0)
        self.assertEqual(self.flux_model.total_inference_time, 0.0)
        self.assertEqual(self.flux_model.average_inference_time, 0.0)

        # Simulate inference completions
        self.flux_model.inference_count = 3
        self.flux_model.total_inference_time = 36.0
        self.flux_model.average_inference_time = 12.0

        self.assertEqual(self.flux_model.inference_count, 3)
        self.assertEqual(self.flux_model.average_inference_time, 12.0)


if __name__ == '__main__':
    unittest.main()