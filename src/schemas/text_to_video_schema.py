"""
Text-to-Video Request and Response Schemas

Defines the data structures for LTX-Video text-to-video generation requests
and responses, including video parameters and LTX-Video specific constraints.
"""

from typing import Optional, Dict, Any, List, Literal, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field
import uuid
from datetime import datetime


class TextToVideoRequest(BaseModel):
    """Request schema for LTX-Video text-to-video generation."""

    model_config = ConfigDict(protected_namespaces=())

    # Required fields
    prompt: str = Field(
        ...,
        description="Text prompt to guide video generation",
        min_length=1,
        max_length=512
    )

    # Optional fields with LTX-Video optimized defaults
    width: int = Field(
        default=704,  # 22*32, divisible by 32 and close to 720
        ge=256,
        le=1280,
        description="Output video width in pixels (must be divisible by 32)"
    )

    height: int = Field(
        default=480,
        ge=256,
        le=1280,
        description="Output video height in pixels (must be divisible by 32)"
    )

    num_frames: int = Field(
        default=129,  # 8*16+1, ~5.3 seconds at 24fps
        ge=9,         # 8*1+1, minimum for LTX-Video
        le=257,       # 8*32+1, maximum recommended
        description="Number of frames to generate (must be 8*n+1)"
    )

    num_inference_steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of denoising steps (8 recommended for distilled model)"
    )

    guidance_scale: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Guidance scale for prompt adherence (3.0-3.5 recommended)"
    )

    fps: int = Field(
        default=24,
        ge=1,
        le=60,
        description="Frames per second for output video"
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducible generation"
    )

    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Negative prompt to avoid unwanted elements"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and process prompt."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")

        return v.strip()

    @field_validator('width', 'height')
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        """Validate resolution is divisible by 32 (LTX-Video requirement)."""
        if v % 32 != 0:
            raise ValueError(f"Resolution must be divisible by 32, got {v}")
        return v

    @field_validator('num_frames')
    @classmethod
    def validate_frames(cls, v: int) -> int:
        """Validate frames follow LTX-Video requirement (8*n + 1)."""
        if (v - 1) % 8 != 0:
            raise ValueError(f"Number of frames must be (8 * n) + 1, got {v}")
        return v

    @computed_field
    @property
    def estimated_duration(self) -> float:
        """Calculate estimated video duration in seconds."""
        return (self.num_frames - 1) / self.fps


class VideoInfo(BaseModel):
    """Information about generated video output."""

    model_config = ConfigDict(protected_namespaces=())

    width: int = Field(description="Video width in pixels")
    height: int = Field(description="Video height in pixels")
    num_frames: int = Field(description="Total number of frames")
    fps: int = Field(description="Frames per second")
    duration: float = Field(description="Video duration in seconds")
    format: str = Field(default="mp4", description="Video file format")
    size_bytes: Optional[int] = Field(default=None, description="File size in bytes")

    def __init__(self, **data):
        """Initialize VideoInfo with duration calculation."""
        # Calculate duration if not provided
        if 'duration' not in data or data['duration'] == 0:
            if 'num_frames' in data and 'fps' in data:
                data['duration'] = (data['num_frames'] - 1) / data['fps']
        super().__init__(**data)


class TextToVideoResponse(BaseModel):
    """Response schema for LTX-Video text-to-video generation."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(description="Whether generation was successful")

    # Success fields
    video_path: Optional[str] = Field(
        default=None,
        description="Path to generated video file"
    )

    video_info: Optional[VideoInfo] = Field(
        default=None,
        description="Information about generated video"
    )

    generation_time: Optional[float] = Field(
        default=None,
        description="Time taken for generation in seconds"
    )

    model_id: Optional[str] = Field(
        default=None,
        description="Model identifier used for generation"
    )

    # Request tracking
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )

    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )

    # Request echo for tracking
    prompt: Optional[str] = Field(
        default=None,
        description="Original prompt used"
    )

    seed: Optional[int] = Field(
        default=None,
        description="Seed used for generation"
    )

    # Error fields
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )

    error_type: Optional[str] = Field(
        default=None,
        description="Type of error that occurred"
    )


# Validation helper functions

def validate_text_to_video_request(request_data: Dict[str, Any]) -> TextToVideoRequest:
    """
    Validate text-to-video request data.

    Args:
        request_data: Raw request data dictionary

    Returns:
        Validated TextToVideoRequest instance

    Raises:
        ValidationError: If validation fails
    """
    return TextToVideoRequest(**request_data)


def create_success_response(
    video_path: str,
    video_info: VideoInfo,
    generation_time: float,
    model_id: str,
    request_id: Optional[str] = None,
    prompt: Optional[str] = None,
    seed: Optional[int] = None
) -> TextToVideoResponse:
    """
    Create a successful text-to-video response.

    Args:
        video_path: Path to generated video file
        video_info: Video metadata
        generation_time: Time taken for generation
        model_id: Model identifier used
        request_id: Optional request ID
        prompt: Original prompt
        seed: Seed used for generation

    Returns:
        TextToVideoResponse indicating success
    """
    return TextToVideoResponse(
        success=True,
        video_path=video_path,
        video_info=video_info,
        generation_time=generation_time,
        model_id=model_id,
        request_id=request_id,
        prompt=prompt,
        seed=seed
    )


def create_error_response(
    error_message: str,
    error_type: str = "GenerationError",
    request_id: Optional[str] = None
) -> TextToVideoResponse:
    """
    Create an error text-to-video response.

    Args:
        error_message: Description of the error
        error_type: Type of error
        request_id: Optional request ID

    Returns:
        TextToVideoResponse indicating failure
    """
    return TextToVideoResponse(
        success=False,
        error_message=error_message,
        error_type=error_type,
        request_id=request_id
    )


# Additional utility functions for LTX-Video specific validation

def is_valid_ltx_resolution(width: int, height: int) -> bool:
    """
    Check if resolution is valid for LTX-Video.

    Args:
        width: Video width
        height: Video height

    Returns:
        True if resolution is valid
    """
    return (width % 32 == 0 and
            height % 32 == 0 and
            256 <= width <= 1280 and
            256 <= height <= 1280)


def is_valid_ltx_frames(num_frames: int) -> bool:
    """
    Check if frame count is valid for LTX-Video.

    Args:
        num_frames: Number of frames

    Returns:
        True if frame count is valid
    """
    return (num_frames >= 9 and
            num_frames <= 257 and
            (num_frames - 1) % 8 == 0)


def get_optimal_ltx_settings(target_duration: float,
                            target_quality: Literal["fast", "balanced", "quality"] = "balanced"
                           ) -> Dict[str, Any]:
    """
    Get optimal LTX-Video settings for given requirements.

    Args:
        target_duration: Desired video duration in seconds
        target_quality: Quality vs speed preference

    Returns:
        Dictionary of recommended settings
    """
    # Calculate optimal frame count for target duration
    fps = 24
    target_frames = int(target_duration * fps) + 1

    # Round to nearest valid frame count (8*n + 1)
    remainder = (target_frames - 1) % 8
    if remainder != 0:
        target_frames = target_frames - remainder + 8 + 1

    # Clamp to valid range
    target_frames = max(9, min(257, target_frames))

    # Quality-based settings
    quality_settings = {
        "fast": {
            "width": 480,
            "height": 320,
            "num_inference_steps": 6,
            "guidance_scale": 2.5
        },
        "balanced": {
            "width": 704,  # Divisible by 32
            "height": 480,
            "num_inference_steps": 8,
            "guidance_scale": 3.0
        },
        "quality": {
            "width": 768,
            "height": 448,  # Divisible by 32
            "num_inference_steps": 12,
            "guidance_scale": 3.5
        }
    }

    settings = quality_settings[target_quality].copy()
    settings.update({
        "num_frames": target_frames,
        "fps": fps
    })

    return settings