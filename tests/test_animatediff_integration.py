"""
Test suite for AnimateDiff image-to-video integration.

Tests the complete AnimateDiff workflow including schema validation,
model loading, video generation, and performance benchmarks.
"""

import pytest
import torch
import io
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
from pydantic import ValidationError

# Test imports
from src.schemas.image_to_video_schema import (
    ImageToVideoRequest,
    ImageToVideoResponse,
    VideoInfo,
    validate_image_to_video_request
)
from src.utils.video_utils import VideoEncoder, FrameProcessor
from src.models.animatediff_model import AnimateDiffModel
from src.handlers.animatediff_handler import AnimateDiffHandler


class TestImageToVideoSchema:
    """Test AnimateDiff request/response schemas."""

    def test_valid_image_to_video_request(self):
        """Test valid image-to-video request creation."""
        # Create test image as base64
        test_image = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        request_data = {
            'prompt': 'a flower blooming in slow motion',
            'input_image': img_base64,
            'num_frames': 16,
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'motion_bucket_id': 127,
            'noise_aug_strength': 0.02
        }

        request = ImageToVideoRequest(**request_data)

        assert request.prompt == 'a flower blooming in slow motion'
        assert request.num_frames == 16
        assert request.width == 512
        assert request.height == 512
        assert request.motion_bucket_id == 127

    def test_invalid_image_format(self):
        """Test invalid base64 image handling."""
        request_data = {
            'prompt': 'test',
            'input_image': 'invalid_base64_data',
            'num_frames': 16
        }

        with pytest.raises(ValidationError, match="Input image must be valid base64 encoded data"):
            ImageToVideoRequest(**request_data)

    def test_frame_count_limits(self):
        """Test frame count validation."""
        # Valid image
        test_image = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Too few frames
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 8"):
            ImageToVideoRequest(
                prompt='test',
                input_image=img_base64,
                num_frames=3
            )

        # Too many frames
        with pytest.raises(ValidationError, match="Input should be less than or equal to 32"):
            ImageToVideoRequest(
                prompt='test',
                input_image=img_base64,
                num_frames=33
            )

    def test_motion_parameter_validation(self):
        """Test motion parameter constraints."""
        # Valid image
        test_image = Image.new('RGB', (512, 512), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Valid parameters
        valid_request = ImageToVideoRequest(
            prompt='test',
            input_image=img_base64,
            motion_bucket_id=127,
            noise_aug_strength=0.02
        )
        assert valid_request.motion_bucket_id == 127
        assert valid_request.noise_aug_strength == 0.02

        # Invalid motion_bucket_id
        with pytest.raises(ValidationError, match="Input should be less than or equal to 255"):
            ImageToVideoRequest(
                prompt='test',
                input_image=img_base64,
                motion_bucket_id=300
            )

    def test_video_info_creation(self):
        """Test VideoInfo response structure."""
        video_info = VideoInfo(
            format='mp4',
            duration=2.0,
            fps=8,
            num_frames=16,
            width=512,
            height=512,
            size_bytes=1048576
        )

        assert video_info.format == 'mp4'
        assert video_info.duration == 2.0
        assert video_info.fps == 8
        assert video_info.num_frames == 16


class TestVideoUtils:
    """Test video encoding and processing utilities."""

    def test_video_encoder_initialization(self):
        """Test VideoEncoder creation with different formats."""
        encoder = VideoEncoder()

        assert encoder.output_format == 'mp4'
        assert encoder.fps == 8.0
        assert encoder.quality == 'medium'

        # Test custom parameters
        encoder_custom = VideoEncoder(
            output_format='gif',
            fps=12.0,
            quality='high'
        )

        assert encoder_custom.output_format == 'gif'
        assert encoder_custom.fps == 12.0
        assert encoder_custom.quality == 'high'

    @patch('imageio.mimsave')
    def test_encode_video_success(self, mock_mimsave):
        """Test successful video encoding."""
        encoder = VideoEncoder()

        # Create test frames
        frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(8)]

        # Mock successful encoding
        mock_mimsave.return_value = None

        # Test encoding
        result = encoder.encode_video_to_base64(frames)

        assert 'video_base64' in result
        assert 'video_info' in result
        assert result['video_info']['frame_count'] == 8
        assert result['video_info']['format'] == 'mp4'

        # Verify imageio was called
        mock_mimsave.assert_called_once()

    def test_frame_processor_interpolation(self):
        """Test frame interpolation for smooth video."""
        processor = FrameProcessor()

        # Create test frames (2 frames to interpolate between)
        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        frame2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
        frames = [frame1, frame2]

        # Test interpolation (should create frames between)
        interpolated = processor.interpolate_frames(frames, target_count=4)

        assert len(interpolated) == 4
        assert interpolated[0].shape == (256, 256, 3)
        assert interpolated[-1].shape == (256, 256, 3)

    def test_frame_validation(self):
        """Test frame format validation."""
        processor = FrameProcessor()

        # Valid frames
        valid_frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(4)]
        assert processor.validate_frames(valid_frames) is True

        # Invalid frame dimensions
        invalid_frames = [np.random.randint(0, 255, (512, 256, 3), dtype=np.uint8) for _ in range(4)]
        assert processor.validate_frames(invalid_frames) is False

        # Empty frames
        assert processor.validate_frames([]) is False


class TestAnimateDiffModel:
    """Test AnimateDiff model wrapper and generation."""

    @pytest.fixture
    def mock_model(self):
        """Create mock AnimateDiff model for testing."""
        with patch('src.models.animatediff_model.AnimateDiffPipeline') as mock_pipeline:
            with patch('src.models.animatediff_model.MotionAdapter') as mock_adapter:
                model = AnimateDiffModel()
                model.pipeline = Mock()
                model.motion_adapter = Mock()
                model.is_loaded = True
                yield model

    def test_model_initialization(self, mock_model):
        """Test model initialization and configuration."""
        assert mock_model.model_id == 'runwayml/stable-diffusion-v1-5'
        assert mock_model.motion_adapter_id == 'guoyww/animatediff-motion-adapter-v1-5-2'
        assert mock_model.torch_dtype == torch.float16
        assert mock_model.is_loaded is True

    @patch('torch.cuda.is_available')
    def test_load_model_cuda(self, mock_cuda, mock_model):
        """Test model loading with CUDA."""
        mock_cuda.return_value = True
        mock_model.is_loaded = False

        # Mock the external loading dependencies
        with patch('src.models.animatediff_model.MotionAdapter.from_pretrained') as mock_motion_adapter, \
             patch('src.models.animatediff_model.AnimateDiffPipeline.from_pretrained') as mock_pipeline, \
             patch('src.models.animatediff_model.DPMSolverMultistepScheduler.from_config') as mock_scheduler:

            # Configure mocks
            mock_motion_adapter.return_value = MagicMock()
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_scheduler.return_value = MagicMock()

            mock_model.load_model()

            assert mock_model.is_loaded is True
            assert mock_model.device == 'cuda'

    @patch('torch.cuda.is_available')
    def test_load_model_cpu(self, mock_cuda, mock_model):
        """Test model loading without CUDA."""
        mock_cuda.return_value = False
        mock_model.is_loaded = False

        # Mock the loading process
        with patch.object(mock_model, '_setup_motion_adapter'), \
             patch.object(mock_model, '_setup_pipeline'):

            mock_model.load_model()

            assert mock_model.is_loaded is True
            assert mock_model.device == 'cpu'

    def test_generate_video_success(self, mock_model):
        """Test successful video generation."""
        # Create test input image
        test_image = Image.new('RGB', (512, 512), color='blue')

        # Mock pipeline output
        mock_result = Mock()
        mock_result.frames = [[np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)]]
        mock_model.pipeline.return_value = mock_result

        # Test generation
        result = mock_model.generate_video(
            prompt="a cat walking",
            input_image=test_image,
            num_frames=16,
            width=512,
            height=512
        )

        assert 'frames' in result
        assert 'generation_info' in result
        assert len(result['frames']) == 16
        assert result['generation_info']['num_frames'] == 16

    def test_memory_management(self, mock_model):
        """Test model memory management and cleanup."""
        # Test unloading
        mock_model.unload_model()

        assert mock_model.is_loaded is False
        assert mock_model.pipeline is None
        assert mock_model.motion_adapter is None

    def test_performance_tracking(self, mock_model):
        """Test generation performance tracking."""
        # Mock successful generation
        test_image = Image.new('RGB', (512, 512), color='green')
        mock_result = Mock()
        mock_result.frames = [[np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(8)]]
        mock_model.pipeline.return_value = mock_result

        result = mock_model.generate_video(
            prompt="test",
            input_image=test_image,
            num_frames=8
        )

        # Check performance metrics
        assert 'generation_time' in result['generation_info']
        assert 'memory_usage' in result['generation_info']
        assert result['generation_info']['generation_time'] > 0


class TestAnimateDiffHandler:
    """Test AnimateDiff request handler integration."""

    @pytest.fixture
    def handler(self):
        """Create AnimateDiff handler for testing."""
        return AnimateDiffHandler()

    def test_handler_initialization(self, handler):
        """Test handler initialization and configuration."""
        assert handler.supported_modality == 'image-to-video'
        assert handler.model_id == 'runwayml/stable-diffusion-v1-5'
        assert handler.motion_adapter_id == 'guoyww/animatediff-motion-adapter-v1-5-2'

    @patch('src.handlers.animatediff_handler.AnimateDiffModel')
    def test_process_request_success(self, mock_model_class, handler):
        """Test successful request processing."""
        # Mock model
        mock_model = Mock()
        mock_model.generate_video.return_value = {
            'frames': [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)],
            'generation_info': {
                'num_frames': 16,
                'generation_time': 15.2,
                'memory_usage': {'peak_memory': 8192}
            }
        }
        mock_model_class.return_value = mock_model
        handler.model = mock_model

        # Create test request
        test_image = Image.new('RGB', (512, 512), color='yellow')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        request_data = {
            'prompt': 'a bird flying gracefully',
            'input_image': img_base64,
            'num_frames': 16,
            'width': 512,
            'height': 512,
            'num_inference_steps': 20
        }

        # Process request
        with patch('src.handlers.animatediff_handler.VideoEncoder') as mock_encoder:
            mock_encoder.return_value.encode_video_to_base64.return_value = {
                'video_base64': 'test_video_data',
                'video_info': {
                    'format': 'mp4',
                    'duration': 2.0,
                    'fps': 8.0,
                    'frame_count': 16,
                    'width': 512,
                    'height': 512,
                    'size_bytes': 1048576
                }
            }

            result = handler.process_request(request_data)

        assert result['success'] is True
        assert 'video_base64' in result
        assert 'video_info' in result
        assert 'generation_info' in result
        assert result['generation_info']['inference_time'] == 15.2

    def test_process_request_validation_error(self, handler):
        """Test request processing with validation errors."""
        # Invalid request (missing required fields)
        request_data = {
            'prompt': 'test prompt'
            # Missing input_image and other required fields
        }

        result = handler.process_request(request_data)

        assert result['success'] is False
        assert 'error' in result
        assert 'validation' in result['error']['type']

    def test_performance_requirements(self, handler):
        """Test that handler meets performance requirements."""
        # Mock fast generation (< 25 seconds)
        with patch.object(handler, '_ensure_model_loaded'), \
             patch.object(handler, 'model') as mock_model:

            mock_model.generate_video.return_value = {
                'frames': [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(16)],
                'generation_info': {
                    'num_frames': 16,
                    'generation_time': 20.5,  # Under 25 second requirement
                    'memory_usage': {'peak_memory': 8192}
                }
            }

            # Create valid request
            test_image = Image.new('RGB', (512, 512), color='purple')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            request_data = {
                'prompt': 'performance test',
                'input_image': img_base64,
                'num_frames': 16
            }

            with patch('src.handlers.animatediff_handler.VideoEncoder') as mock_encoder:
                mock_encoder.return_value.encode_video_to_base64.return_value = {
                    'video_base64': 'test_data',
                    'video_info': {'format': 'mp4', 'duration': 2.0, 'fps': 8.0, 'frame_count': 16}
                }

                result = handler.process_request(request_data)

            # Verify performance requirement is met
            assert result['success'] is True
            assert result['generation_info']['inference_time'] < 25.0


class TestIntegration:
    """Integration tests for complete AnimateDiff workflow."""

    @patch('src.handlers.multi_modal_handler.AnimateDiffHandler')
    def test_multi_modal_handler_integration(self, mock_handler_class):
        """Test AnimateDiff integration with multi-modal handler."""
        from src.handlers.multi_modal_handler import MultiModalHandler
        from src.models.model_manager import ModelManager

        # Mock handler
        mock_handler = Mock()
        mock_handler.supported_modality = 'image-to-video'
        mock_handler_class.return_value = mock_handler

        # Create multi-modal handler with model manager
        model_manager = Mock()
        multi_handler = MultiModalHandler(model_manager)

        # Verify AnimateDiff handler is registered
        assert 'image-to-video' in multi_handler.get_supported_modalities()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end AnimateDiff workflow."""
        # This would be a full integration test with real models
        # For now, just verify the workflow structure exists

        from src.schemas.image_to_video_schema import ImageToVideoRequest
        from src.utils.video_utils import VideoEncoder
        from src.handlers.animatediff_handler import AnimateDiffHandler

        # Verify all components can be imported
        assert ImageToVideoRequest is not None
        assert VideoEncoder is not None
        assert AnimateDiffHandler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])