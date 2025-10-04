"""
Tests for LTX-Video text-to-video model wrapper.

This test module follows Test-Driven Development (TDD) principles, defining the complete
interface and behavior requirements for the LTXVideoModel before implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import torch
from PIL import Image
import numpy as np
from datetime import datetime

from src.models.base_model import BaseModel
from src.utils.exceptions import ModelLoadError, InferenceError, ValidationError


class TestLTXVideoModel:
    """Test suite for LTXVideoModel implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_device = "cuda"
        self.mock_torch_dtype = torch.float16

        # Mock model configuration
        self.default_params = {
            'width': 704,
            'height': 704,
            'num_frames': 25,  # (8*3)+1 for LTX-Video
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'fps': 8
        }

    @pytest.fixture
    def ltx_video_model(self):
        """Create LTXVideoModel instance for testing."""
        from src.models.ltx_video_model import LTXVideoModel
        return LTXVideoModel(
            device=self.mock_device,
            torch_dtype=self.mock_torch_dtype
        )

    def test_ltx_video_model_initialization(self, ltx_video_model):
        """Test LTXVideoModel proper initialization."""
        assert ltx_video_model.device == self.mock_device
        assert ltx_video_model.torch_dtype == self.mock_torch_dtype
        assert ltx_video_model.model_id == "Lightricks/LTX-Video"
        assert ltx_video_model.is_loaded is False
        assert ltx_video_model.pipeline is None
        assert ltx_video_model.inference_count == 0
        assert ltx_video_model.total_inference_time == 0.0

    def test_ltx_video_model_inherits_from_base_model(self, ltx_video_model):
        """Test that LTXVideoModel properly inherits from BaseModel."""
        assert isinstance(ltx_video_model, BaseModel)
        assert hasattr(ltx_video_model, 'load_model')
        assert hasattr(ltx_video_model, 'unload_model')
        assert hasattr(ltx_video_model, 'perform_inference')
        assert hasattr(ltx_video_model, 'get_memory_usage')

    def test_ltx_video_model_default_configuration(self, ltx_video_model):
        """Test default configuration matches LTX-Video requirements."""
        default_params = ltx_video_model.get_default_params()

        # Test resolution requirements (divisible by 32)
        assert default_params['width'] % 32 == 0
        assert default_params['height'] % 32 == 0

        # Test frame requirements ((8*n)+1 pattern)
        frames = default_params['num_frames']
        assert (frames - 1) % 8 == 0  # Validates (8*n)+1 pattern

        # Test performance optimized defaults
        assert default_params['num_inference_steps'] == 20
        assert default_params['guidance_scale'] == 7.5
        assert default_params['fps'] == 8

    @patch('src.models.ltx_video_model.LTXPipeline.from_pretrained')
    def test_model_loading_success(self, mock_from_pretrained, ltx_video_model):
        """Test successful model loading with LTXPipeline."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline

        # Mock memory efficient methods
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline.enable_vae_slicing = Mock()
        mock_pipeline.to = Mock(return_value=mock_pipeline)

        # Load model
        ltx_video_model.load_model()

        # Verify pipeline loading
        mock_from_pretrained.assert_called_once_with(
            "Lightricks/LTX-Video",
            torch_dtype=self.mock_torch_dtype
        )

        # Verify device placement
        mock_pipeline.to.assert_called_once_with(self.mock_device)

        # Verify memory optimizations
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_vae_slicing.assert_called_once()

        # Verify state
        assert ltx_video_model.is_loaded is True
        assert ltx_video_model.pipeline == mock_pipeline
        assert ltx_video_model.load_time is not None

    @patch('src.models.ltx_video_model.LTXPipeline.from_pretrained')
    def test_model_loading_failure(self, mock_from_pretrained, ltx_video_model):
        """Test model loading failure handling."""
        mock_from_pretrained.side_effect = Exception("Model loading failed")

        with pytest.raises(ModelLoadError) as exc_info:
            ltx_video_model.load_model()

        assert "Failed to load LTX-Video model" in str(exc_info.value)
        assert ltx_video_model.is_loaded is False
        assert ltx_video_model.pipeline is None

    def test_model_unloading(self, ltx_video_model):
        """Test model unloading releases resources."""
        # Mock loaded state
        ltx_video_model.pipeline = Mock()
        ltx_video_model.is_loaded = True

        ltx_video_model.unload_model()

        assert ltx_video_model.pipeline is None
        assert ltx_video_model.is_loaded is False

    @patch('src.models.ltx_video_model.torch.manual_seed')
    @patch('src.models.ltx_video_model.torch.cuda.manual_seed')
    def test_text_to_video_generation_success(self, mock_cuda_seed, mock_seed, ltx_video_model):
        """Test successful text-to-video generation."""
        # Setup loaded model
        mock_pipeline = Mock()
        ltx_video_model.pipeline = mock_pipeline
        ltx_video_model.is_loaded = True

        # Mock pipeline result
        mock_frames = [Image.new('RGB', (704, 704), color='red') for _ in range(25)]
        mock_result = Mock()
        mock_result.frames = [mock_frames]  # LTX returns list of sequences
        mock_pipeline.return_value = mock_result

        # Mock video encoding
        with patch('src.models.ltx_video_model.encode_video_to_base64') as mock_encode:
            mock_encode.return_value = ("base64_video_data", {"format": "mp4"})

            # Generate video
            result = ltx_video_model.generate_video(
                prompt="A cat walking in the garden",
                width=704,
                height=704,
                num_frames=25,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=42
            )

        # Verify seed setting
        mock_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_called_once_with(42)

        # Verify pipeline call
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['prompt'] == "A cat walking in the garden"
        assert call_kwargs['width'] == 704
        assert call_kwargs['height'] == 704
        assert call_kwargs['num_frames'] == 25
        assert call_kwargs['num_inference_steps'] == 20
        assert call_kwargs['guidance_scale'] == 7.5

        # Verify result structure
        assert isinstance(result, tuple)
        video_data, generation_info = result
        assert video_data == "base64_video_data"
        assert isinstance(generation_info, dict)
        assert 'inference_time' in generation_info
        assert 'frames_generated' in generation_info
        assert generation_info['frames_generated'] == 25

    def test_text_to_video_generation_model_not_loaded(self, ltx_video_model):
        """Test text-to-video generation fails when model not loaded."""
        assert ltx_video_model.is_loaded is False

        with pytest.raises(InferenceError) as exc_info:
            ltx_video_model.generate_video("A cat walking")

        assert "Model not loaded" in str(exc_info.value)

    def test_text_to_video_generation_pipeline_failure(self, ltx_video_model):
        """Test text-to-video generation handles pipeline failures."""
        # Setup loaded model with failing pipeline
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline inference failed")
        ltx_video_model.pipeline = mock_pipeline
        ltx_video_model.is_loaded = True

        with pytest.raises(InferenceError) as exc_info:
            ltx_video_model.generate_video("A cat walking")

        assert "Video generation failed" in str(exc_info.value)

    def test_parameter_preprocessing(self, ltx_video_model):
        """Test parameter preprocessing for LTX-Video compatibility."""
        # Test resolution adjustment to nearest divisible by 32
        params = ltx_video_model._preprocess_params(
            width=720, height=480, num_frames=24
        )

        assert params['width'] % 32 == 0
        assert params['height'] % 32 == 0
        assert params['width'] in [704, 736]  # Closest to 720
        assert params['height'] in [480, 512]  # Closest to 480

        # Test frame adjustment to (8*n)+1 pattern
        assert (params['num_frames'] - 1) % 8 == 0
        assert params['num_frames'] in [17, 25]  # Closest to 24

    def test_prompt_validation(self, ltx_video_model):
        """Test prompt validation for LTX-Video requirements."""
        # Test valid prompts
        assert ltx_video_model._validate_prompt("A cat walking") is True
        assert ltx_video_model._validate_prompt("Short prompt") is True

        # Test empty prompt handling
        assert ltx_video_model._validate_prompt("") is False
        assert ltx_video_model._validate_prompt("   ") is False

        # Test prompt length limits
        long_prompt = "A" * 1000
        assert ltx_video_model._validate_prompt(long_prompt) is False

    def test_perform_inference_integration(self, ltx_video_model):
        """Test perform_inference method for BaseModel compatibility."""
        # Mock loaded state
        ltx_video_model.is_loaded = True

        # Mock generate_video method
        with patch.object(ltx_video_model, 'generate_video') as mock_generate:
            mock_generate.return_value = ("video_data", {"frames": 25})

            inputs = {
                'prompt': 'A cat walking',
                'width': 704,
                'height': 704,
                'num_frames': 25
            }

            result = ltx_video_model.perform_inference(inputs)

            mock_generate.assert_called_once_with(
                prompt='A cat walking',
                width=704,
                height=704,
                num_frames=25
            )

            assert 'video_data' in result
            assert 'frames' in result

    def test_validate_inputs(self, ltx_video_model):
        """Test input validation for text-to-video parameters."""
        # Test valid inputs
        valid_inputs = {
            'prompt': 'A cat walking',
            'width': 704,
            'height': 704,
            'num_frames': 25,
            'fps': 8
        }
        assert ltx_video_model.validate_inputs(valid_inputs) is True

        # Test missing prompt
        invalid_inputs = {'width': 704, 'height': 704}
        assert ltx_video_model.validate_inputs(invalid_inputs) is False

        # Test invalid dimensions
        invalid_inputs = {'prompt': 'test', 'width': 100, 'height': 100}
        assert ltx_video_model.validate_inputs(invalid_inputs) is False

        # Test invalid frame count
        invalid_inputs = {'prompt': 'test', 'num_frames': 30}  # Not (8*n)+1
        assert ltx_video_model.validate_inputs(invalid_inputs) is False

    def test_memory_usage_tracking(self, ltx_video_model):
        """Test memory usage tracking and statistics."""
        # Test unloaded model memory
        assert ltx_video_model.get_memory_usage() == 0

        # Test loaded model memory estimation
        ltx_video_model.is_loaded = True
        ltx_video_model._current_memory_mb = 8000.0

        memory_usage = ltx_video_model.get_memory_usage()
        assert isinstance(memory_usage, int)
        assert memory_usage > 0
        assert memory_usage <= 12000  # LTX-Video should be < 12GB

    def test_model_info_reporting(self, ltx_video_model):
        """Test comprehensive model information reporting."""
        # Setup some statistics
        ltx_video_model.inference_count = 5
        ltx_video_model.total_inference_time = 100.0
        ltx_video_model._peak_memory_mb = 8500.0

        info = ltx_video_model.get_model_info()

        assert info['model_id'] == "Lightricks/LTX-Video"
        assert info['model_type'] == "text-to-video"
        assert info['device'] == self.mock_device
        assert info['torch_dtype'] == str(self.mock_torch_dtype)
        assert info['is_loaded'] is False  # Not loaded in this test
        assert info['inference_count'] == 5
        assert info['avg_inference_time'] == 20.0  # 100/5
        assert info['peak_memory_mb'] == 8500.0

    @patch('src.models.ltx_video_model.validate_video_frames')
    def test_frame_validation_integration(self, mock_validate, ltx_video_model):
        """Test integration with video frame validation utilities."""
        # Setup loaded model
        mock_pipeline = Mock()
        ltx_video_model.pipeline = mock_pipeline
        ltx_video_model.is_loaded = True

        # Mock successful generation
        mock_frames = [np.random.randint(0, 255, (704, 704, 3), dtype=np.uint8) for _ in range(25)]
        mock_result = Mock()
        mock_result.frames = [mock_frames]
        mock_pipeline.return_value = mock_result

        with patch('src.models.ltx_video_model.encode_video_to_base64') as mock_encode:
            mock_encode.return_value = ("video_data", {"format": "mp4"})
            ltx_video_model.generate_video("test prompt")

        # Verify frame validation was called
        mock_validate.assert_called_once()
        validated_frames = mock_validate.call_args[0][0]
        assert len(validated_frames) == 25

    def test_performance_metrics_tracking(self, ltx_video_model):
        """Test inference performance metrics tracking."""
        # Setup loaded model
        ltx_video_model.pipeline = Mock()
        ltx_video_model.is_loaded = True

        # Mock successful inference
        with patch.object(ltx_video_model, 'generate_video') as mock_generate:
            mock_generate.return_value = ("video_data", {"inference_time": 15.5})

            # Perform multiple inferences
            for i in range(3):
                ltx_video_model.perform_inference({'prompt': f'test {i}'})

        # Verify performance tracking
        assert ltx_video_model.inference_count >= 3

    def test_ltx_video_specific_optimizations(self, ltx_video_model):
        """Test LTX-Video specific optimizations and configurations."""
        default_params = ltx_video_model.get_default_params()

        # Test optimized default parameters for performance
        assert default_params['num_inference_steps'] == 20  # Balanced quality/speed
        assert default_params['guidance_scale'] == 7.5      # Optimal for LTX-Video

        # Test resolution optimization
        width, height = ltx_video_model._optimize_resolution(720, 480)
        assert width % 32 == 0
        assert height % 32 == 0
        assert abs(width - 720) <= 32  # Close to original
        assert abs(height - 480) <= 32

    def test_error_handling_comprehensive(self, ltx_video_model):
        """Test comprehensive error handling across all methods."""
        # Test validation errors
        with pytest.raises(ValidationError):
            ltx_video_model._validate_prompt("")

        # Test inference errors when not loaded
        with pytest.raises(InferenceError):
            ltx_video_model.perform_inference({'prompt': 'test'})

        # Test model loading errors
        with patch('src.models.ltx_video_model.LTXPipeline.from_pretrained') as mock_load:
            mock_load.side_effect = RuntimeError("CUDA out of memory")

            with pytest.raises(ModelLoadError):
                ltx_video_model.load_model()

    def test_string_representation(self, ltx_video_model):
        """Test string representation of LTXVideoModel."""
        repr_str = str(ltx_video_model)
        assert "LTXVideoModel" in repr_str
        assert "unloaded" in repr_str

        # Test loaded state
        ltx_video_model.is_loaded = True
        repr_str = str(ltx_video_model)
        assert "loaded" in repr_str


class TestLTXVideoModelIntegration:
    """Integration tests for LTXVideoModel with external dependencies."""

    @pytest.fixture
    def mock_ltx_pipeline(self):
        """Mock LTX Pipeline for integration tests."""
        with patch('src.models.ltx_video_model.LTXPipeline') as mock:
            yield mock

    def test_full_text_to_video_workflow(self, mock_ltx_pipeline):
        """Test complete text-to-video generation workflow."""
        from src.models.ltx_video_model import LTXVideoModel

        # Setup pipeline mock
        mock_pipeline_instance = Mock()
        mock_ltx_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Mock successful generation
        mock_frames = [Image.new('RGB', (704, 704), color='blue') for _ in range(25)]
        mock_result = Mock()
        mock_result.frames = [mock_frames]
        mock_pipeline_instance.return_value = mock_result

        # Mock video encoding
        with patch('src.models.ltx_video_model.encode_video_to_base64') as mock_encode:
            mock_encode.return_value = ("encoded_video_data", {"format": "mp4"})

            # Initialize and load model
            model = LTXVideoModel()
            model.load_model()

            # Generate video
            video_data, info = model.generate_video(
                prompt="A beautiful sunset over mountains",
                width=704,
                height=704,
                num_frames=25,
                seed=123
            )

            # Verify results
            assert video_data == "encoded_video_data"
            assert info['frames_generated'] == 25
            assert 'inference_time' in info

            # Verify model statistics updated
            assert model.inference_count == 1
            assert model.total_inference_time > 0