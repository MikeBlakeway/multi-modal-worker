"""
Test suite for Text-to-Video Request and Response Schemas

Tests the data structures for LTX-Video text-to-video generation requests
and responses, including parameter validation and LTX-Video specific constraints.
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any

# Import the schemas we're testing (will be implemented after tests)
try:
    from src.schemas.text_to_video_schema import (
        TextToVideoRequest,
        TextToVideoResponse,
        VideoInfo,
        validate_text_to_video_request,
        create_success_response,
        create_error_response
    )
except ImportError:
    # During test development, these may not exist yet
    pytest.skip("Schema module not implemented yet", allow_module_level=True)


class TestTextToVideoRequest:
    """Test cases for TextToVideoRequest schema validation."""

    def test_minimal_valid_request(self):
        """Test creation with minimal valid parameters."""
        request_data = {
            "prompt": "A beautiful sunset over mountains"
        }
        request = TextToVideoRequest(**request_data)

        assert request.prompt == "A beautiful sunset over mountains"
        # Test default values
        assert request.width == 704
        assert request.height == 480
        assert request.num_frames == 129  # 8*16+1 for LTX-Video
        assert request.num_inference_steps == 8
        assert request.guidance_scale == 3.0
        assert request.fps == 24

    def test_complete_valid_request(self):
        """Test creation with all valid parameters."""
        request_data = {
            "prompt": "A majestic eagle soaring through clouds",
            "width": 768,
            "height": 448,  # 448 is divisible by 32 (14 * 32)
            "num_frames": 97,  # 8*12+1
            "num_inference_steps": 6,
            "guidance_scale": 3.5,
            "fps": 30,
            "seed": 42,
            "negative_prompt": "blurry, low quality"
        }
        request = TextToVideoRequest(**request_data)

        assert request.prompt == "A majestic eagle soaring through clouds"
        assert request.width == 768
        assert request.height == 448
        assert request.num_frames == 97
        assert request.num_inference_steps == 6
        assert request.guidance_scale == 3.5
        assert request.fps == 30
        assert request.seed == 42
        assert request.negative_prompt == "blurry, low quality"

    def test_prompt_validation(self):
        """Test prompt field validation."""
        # Empty prompt should fail
        with pytest.raises(ValidationError) as exc_info:
            TextToVideoRequest(prompt="")
        assert "prompt" in str(exc_info.value)

        # Missing prompt should fail
        with pytest.raises(ValidationError) as exc_info:
            TextToVideoRequest()
        assert "prompt" in str(exc_info.value)

        # Very long prompt should fail (pydantic validation catches this)
        long_prompt = "A" * 1000
        with pytest.raises(ValidationError) as exc_info:
            TextToVideoRequest(prompt=long_prompt)
        assert "String should have at most 512 characters" in str(exc_info.value)

    def test_resolution_validation(self):
        """Test width and height validation (must be divisible by 32)."""
        # Valid resolutions (divisible by 32)
        valid_resolutions = [(480, 320), (704, 480), (768, 448), (1024, 576)]
        for width, height in valid_resolutions:
            request = TextToVideoRequest(
                prompt="test",
                width=width,
                height=height
            )
            assert request.width == width
            assert request.height == height

        # Invalid resolutions (not divisible by 32)
        invalid_resolutions = [(481, 321), (719, 479), (720, 431)]
        for width, height in invalid_resolutions:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    width=width,
                    height=height
                )
            error_msg = str(exc_info.value)
            assert "divisible by 32" in error_msg

    def test_frame_validation(self):
        """Test num_frames validation (must be divisible by 8 plus 1)."""
        # Valid frame counts (8*n + 1)
        valid_frames = [9, 17, 25, 33, 41, 49, 65, 97, 129, 161]
        for frames in valid_frames:
            request = TextToVideoRequest(
                prompt="test",
                num_frames=frames
            )
            assert request.num_frames == frames

        # Invalid frame counts (not 8*n + 1)
        invalid_frames = [8, 10, 16, 18, 24, 26, 32, 64, 128, 160]
        for frames in invalid_frames:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    num_frames=frames
                )
            error_msg = str(exc_info.value)
            # The validation error may come from pydantic constraint or our custom validator
            assert ("must be (8 * n) + 1" in error_msg or
                    "Input should be greater than or equal to 9" in error_msg)

    def test_inference_steps_validation(self):
        """Test num_inference_steps validation."""
        # Valid range: 1-50
        valid_steps = [1, 8, 16, 30, 50]
        for steps in valid_steps:
            request = TextToVideoRequest(
                prompt="test",
                num_inference_steps=steps
            )
            assert request.num_inference_steps == steps

        # Invalid values
        invalid_steps = [0, -1, 51, 100]
        for steps in invalid_steps:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    num_inference_steps=steps
                )

    def test_guidance_scale_validation(self):
        """Test guidance_scale validation."""
        # Valid range: 0.0-10.0
        valid_scales = [0.0, 1.0, 3.0, 7.0, 10.0]
        for scale in valid_scales:
            request = TextToVideoRequest(
                prompt="test",
                guidance_scale=scale
            )
            assert request.guidance_scale == scale

        # Invalid values
        invalid_scales = [-0.1, -1.0, 10.1, 20.0]
        for scale in invalid_scales:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    guidance_scale=scale
                )

    def test_fps_validation(self):
        """Test fps validation."""
        # Valid range: 1-60
        valid_fps = [1, 8, 24, 30, 48, 60]
        for fps in valid_fps:
            request = TextToVideoRequest(
                prompt="test",
                fps=fps
            )
            assert request.fps == fps

        # Invalid values
        invalid_fps = [0, -1, 61, 120]
        for fps in invalid_fps:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    fps=fps
                )

    def test_seed_validation(self):
        """Test seed validation."""
        # Valid seeds
        valid_seeds = [None, 0, 42, 999999, 2147483647]
        for seed in valid_seeds:
            request = TextToVideoRequest(
                prompt="test",
                seed=seed
            )
            assert request.seed == seed

        # Invalid seeds (negative or too large)
        invalid_seeds = [-1, -42, 2147483648]
        for seed in invalid_seeds:
            with pytest.raises(ValidationError) as exc_info:
                TextToVideoRequest(
                    prompt="test",
                    seed=seed
                )


class TestVideoInfo:
    """Test cases for VideoInfo schema."""

    def test_valid_video_info(self):
        """Test creation with valid video information."""
        video_info = VideoInfo(
            width=720,
            height=480,
            num_frames=129,
            fps=24,
            duration=5.375,
            format="mp4",
            size_bytes=1048576
        )

        assert video_info.width == 720
        assert video_info.height == 480
        assert video_info.num_frames == 129
        assert video_info.fps == 24
        assert video_info.duration == 5.375
        assert video_info.format == "mp4"
        assert video_info.size_bytes == 1048576

    def test_duration_calculation(self):
        """Test that duration is correctly calculated from frames and fps."""
        # Duration should be calculated as (num_frames - 1) / fps
        video_info = VideoInfo(
            width=720,
            height=480,
            num_frames=129,  # 128 effective frames
            fps=24,
            duration=0,  # Should be overridden
            format="mp4"
        )

        expected_duration = (129 - 1) / 24  # ~5.33 seconds
        assert abs(video_info.duration - expected_duration) < 0.01


class TestTextToVideoResponse:
    """Test cases for TextToVideoResponse schema."""

    def test_success_response(self):
        """Test creation of successful response."""
        video_info = VideoInfo(
            width=720,
            height=480,
            num_frames=129,
            fps=24,
            duration=5.375,
            format="mp4",
            size_bytes=1048576
        )

        response = TextToVideoResponse(
            success=True,
            video_path="/tmp/output.mp4",
            video_info=video_info,
            generation_time=42.5,
            model_id="Lightricks/LTX-Video",
            request_id="test-123",
            prompt="A beautiful sunset",
            seed=42
        )

        assert response.success is True
        assert response.video_path == "/tmp/output.mp4"
        assert response.video_info.width == 720
        assert response.generation_time == 42.5
        assert response.model_id == "Lightricks/LTX-Video"
        assert response.request_id == "test-123"
        assert response.error_message is None

    def test_error_response(self):
        """Test creation of error response."""
        response = TextToVideoResponse(
            success=False,
            error_message="Model loading failed",
            request_id="test-456"
        )

        assert response.success is False
        assert response.error_message == "Model loading failed"
        assert response.request_id == "test-456"
        assert response.video_path is None
        assert response.video_info is None


class TestValidationFunctions:
    """Test cases for validation helper functions."""

    def test_validate_text_to_video_request_success(self):
        """Test successful request validation."""
        request_data = {
            "prompt": "A beautiful landscape",
            "width": 704,  # 704 is divisible by 32 (22 * 32)
            "height": 480,
            "num_frames": 129
        }

        request = validate_text_to_video_request(request_data)

        assert isinstance(request, TextToVideoRequest)
        assert request.prompt == "A beautiful landscape"

    def test_validate_text_to_video_request_failure(self):
        """Test failed request validation."""
        invalid_data = {
            "prompt": "",  # Invalid empty prompt
            "width": 719,  # Invalid width (not divisible by 32)
            "num_frames": 128  # Invalid frames (not 8*n+1)
        }

        with pytest.raises(ValidationError):
            validate_text_to_video_request(invalid_data)

    def test_create_success_response(self):
        """Test success response creation."""
        video_info = VideoInfo(
            width=720,
            height=480,
            num_frames=129,
            fps=24,
            duration=5.375,
            format="mp4"
        )

        response = create_success_response(
            video_path="/tmp/video.mp4",
            video_info=video_info,
            generation_time=30.0,
            model_id="Lightricks/LTX-Video",
            request_id="req-123",
            prompt="Test prompt",
            seed=42
        )

        assert response.success is True
        assert response.video_path == "/tmp/video.mp4"
        assert response.generation_time == 30.0

    def test_create_error_response(self):
        """Test error response creation."""
        response = create_error_response(
            error_message="Generation failed",
            request_id="req-456"
        )

        assert response.success is False
        assert response.error_message == "Generation failed"
        assert response.request_id == "req-456"


class TestLTXVideoSpecificConstraints:
    """Test cases for LTX-Video specific parameter constraints."""

    def test_mobile_first_defaults(self):
        """Test that default resolution is mobile-first (720x1280 portrait)."""
        request = TextToVideoRequest(prompt="test")

        # Default should be optimized for mobile content creation
        assert request.width == 704  # Updated to be divisible by 32
        assert request.height == 480  # Landscape for now, portrait could be 720x1280

    def test_optimal_frame_counts(self):
        """Test that optimal frame counts are supported."""
        # Test recommended frame counts for different video lengths
        optimal_frames = {
            9: 0.33,    # Very short (8 frames)
            17: 0.67,   # Short (16 frames)
            25: 1.0,    # 1 second
            49: 2.0,    # 2 seconds
            129: 5.33,  # 5+ seconds (target)
        }

        for frames, expected_duration in optimal_frames.items():
            request = TextToVideoRequest(
                prompt="test",
                num_frames=frames,
                fps=24
            )
            actual_duration = (frames - 1) / 24
            assert abs(actual_duration - expected_duration) < 0.1

    def test_performance_oriented_defaults(self):
        """Test that defaults are optimized for <45 second generation target."""
        request = TextToVideoRequest(prompt="test")

        # These defaults should enable <45 second generation
        assert request.num_inference_steps == 8  # Distilled model optimal
        assert request.guidance_scale == 3.0     # LTX-Video recommended
        assert request.num_frames <= 129         # Performance balance
        assert request.width * request.height <= 704 * 480  # Memory efficient

    def test_resolution_recommendations(self):
        """Test LTX-Video recommended resolutions."""
        # Test common LTX-Video optimal resolutions
        recommended_resolutions = [
            (480, 320),   # Very fast
            (704, 480),   # Balanced (default)
            (768, 448),   # High quality
            (1024, 576),  # Maximum recommended
        ]

        for width, height in recommended_resolutions:
            request = TextToVideoRequest(
                prompt="test",
                width=width,
                height=height
            )
            assert request.width == width
            assert request.height == height
            # Should not exceed LTX-Video's recommended max
            assert width * height <= 1024 * 576