"""
ControlNet Request and Response Schemas

Defines the data structures for ControlNet guided image generation requests
and responses, including control image processing and parameter validation.
"""

from typing import Optional, Dict, Any, List, Literal, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import base64


class ControlNetRequest(BaseModel):
    """Request schema for ControlNet guided image generation."""

    # Required parameters
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text description of the image to generate"
    )

    control_image: str = Field(
        ...,
        description="Base64-encoded control image for structural guidance"
    )

    control_type: Literal["canny", "depth"] = Field(
        ...,
        description="Type of control processing to apply to the control image"
    )

    # Image dimensions
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Width of generated image in pixels"
    )

    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Height of generated image in pixels"
    )

    # ControlNet-specific parameters
    control_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Strength of control guidance (0.0 = no control, 1.0 = full control)"
    )

    control_guidance_start: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of steps to start applying control guidance"
    )

    control_guidance_end: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of steps to stop applying control guidance"
    )

    # Inference parameters
    num_inference_steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of denoising steps for ControlNet inference"
    )

    guidance_scale: float = Field(
        default=7.5,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale"
    )

    # Control preprocessing parameters
    canny_low_threshold: Optional[int] = Field(
        default=100,
        ge=1,
        le=255,
        description="Low threshold for Canny edge detection (only for canny control_type)"
    )

    canny_high_threshold: Optional[int] = Field(
        default=200,
        ge=1,
        le=255,
        description="High threshold for Canny edge detection (only for canny control_type)"
    )

    # Optional parameters
    negative_prompt: Optional[str] = Field(
        default="",
        max_length=2000,
        description="Negative prompt to guide what not to generate"
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducible generation"
    )

    # Output parameters
    output_format: Literal["png", "jpg", "webp"] = Field(
        default="png",
        description="Output image format"
    )

    quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="Output image quality (for lossy formats)"
    )

    @field_validator('control_image')
    @classmethod
    def validate_control_image(cls, v: str) -> str:
        """Validate control image is proper base64."""
        try:
            import base64
            # Try to decode to verify it's valid base64
            decoded = base64.b64decode(v)
            if len(decoded) == 0:
                raise ValueError("Control image cannot be empty")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 control image: {str(e)}")

    @field_validator('control_guidance_end')
    @classmethod
    def validate_guidance_range(cls, v: float, info) -> float:
        """Validate guidance end is after guidance start."""
        if hasattr(info.data, 'control_guidance_start') and info.data.get('control_guidance_start') is not None:
            if v <= info.data['control_guidance_start']:
                raise ValueError("control_guidance_end must be greater than control_guidance_start")
        return v

    @field_validator('canny_high_threshold')
    @classmethod
    def validate_canny_thresholds(cls, v: Optional[int], info) -> Optional[int]:
        """Validate Canny high threshold is greater than low threshold."""
        if v is not None and hasattr(info.data, 'canny_low_threshold'):
            low_threshold = info.data.get('canny_low_threshold')
            if low_threshold is not None and v <= low_threshold:
                raise ValueError("canny_high_threshold must be greater than canny_low_threshold")
        return v


class ControlImageInfo(BaseModel):
    """Information about processed control image."""

    original_width: int = Field(description="Original control image width")
    original_height: int = Field(description="Original control image height")
    processed_width: int = Field(description="Processed control image width")
    processed_height: int = Field(description="Processed control image height")
    control_type: str = Field(description="Type of control processing applied")
    preprocessing_time_ms: float = Field(description="Time spent preprocessing control image")


class ControlNetOutput(BaseModel):
    """Generated image output from ControlNet."""

    image: str = Field(description="Base64-encoded generated image")
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    format: str = Field(description="Image format (png, jpg, webp)")
    control_info: ControlImageInfo = Field(description="Control image processing information")


class ControlNetResponse(BaseModel):
    """Response schema for ControlNet guided image generation."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(description="Whether the generation was successful")
    images: List[ControlNetOutput] = Field(
        default_factory=list,
        description="List of generated images"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for generation"
    )
    inference_time_ms: float = Field(description="Total inference time in milliseconds")
    model_load_time_ms: Optional[float] = Field(
        default=None,
        description="Model loading time if model was loaded"
    )
    preprocessing_time_ms: float = Field(description="Control image preprocessing time")
    memory_used_mb: Optional[float] = Field(
        default=None,
        description="Peak memory usage during inference"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracking"
    )


class ControlNetError(BaseModel):
    """Error response for ControlNet generation."""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_type: str = Field(description="Type of error (validation, inference, etc.)")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracking"
    )


# Utility functions for validation and response creation
def validate_controlnet_request(request_data: Dict[str, Any]) -> ControlNetRequest:
    """
    Validate and parse ControlNet request data.

    Args:
        request_data: Raw request data dictionary

    Returns:
        Validated ControlNetRequest instance

    Raises:
        ValidationError: If request data is invalid
    """
    try:
        return ControlNetRequest(**request_data)
    except Exception as e:
        from ..utils.exceptions import ValidationError
        raise ValidationError("controlnet_request", str(request_data), f"Invalid ControlNet request: {str(e)}")


def create_success_response(
    images: List[ControlNetOutput],
    inference_time_ms: float,
    parameters: Dict[str, Any],
    preprocessing_time_ms: float = 0.0,
    model_load_time_ms: Optional[float] = None,
    memory_used_mb: Optional[float] = None,
    request_id: Optional[str] = None
) -> ControlNetResponse:
    """
    Create a successful ControlNet response.

    Args:
        images: List of generated image outputs
        inference_time_ms: Total inference time
        parameters: Generation parameters used
        preprocessing_time_ms: Control preprocessing time
        model_load_time_ms: Model loading time if applicable
        memory_used_mb: Peak memory usage
        request_id: Request identifier

    Returns:
        ControlNetResponse instance
    """
    return ControlNetResponse(
        success=True,
        images=images,
        parameters=parameters,
        inference_time_ms=inference_time_ms,
        preprocessing_time_ms=preprocessing_time_ms,
        model_load_time_ms=model_load_time_ms,
        memory_used_mb=memory_used_mb,
        request_id=request_id
    )


def create_error_response(
    error_message: str,
    error_type: str = "unknown",
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ControlNetError:
    """
    Create a ControlNet error response.

    Args:
        error_message: Human-readable error message
        error_type: Type of error for categorization
        details: Additional error details
        request_id: Request identifier

    Returns:
        ControlNetError instance
    """
    return ControlNetError(
        error=error_message,
        error_type=error_type,
        details=details,
        request_id=request_id
    )