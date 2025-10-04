"""
Text-to-Image Request and Response Schemas

Defines the data structures for FLUX.1 text-to-image generation requests
and responses, including parameter validation and default values.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import base64


class TextToImageRequest(BaseModel):
    """Request schema for text-to-image generation using FLUX.1."""

    # Required parameters
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text description of the image to generate"
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

    # Inference parameters
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of denoising steps for FLUX.1 Schnell"
    )

    guidance_scale: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Guidance scale (FLUX.1 Schnell typically uses 0.0)"
    )

    # Seed for reproducibility
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducible generation"
    )

    # Output format
    output_format: str = Field(
        default="png",
        description="Output image format (png, jpeg, webp)"
    )

    # Quality settings
    quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="Output quality for JPEG format (1-100)"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        """Validate and clean prompt text."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")

        # Remove excessive whitespace
        cleaned = ' '.join(v.strip().split())

        if len(cleaned) < 1:
            raise ValueError("Prompt must contain meaningful content")

        return cleaned

    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 8 for optimal FLUX.1 performance."""
        if v % 8 != 0:
            # Round to nearest multiple of 8
            rounded = round(v / 8) * 8
            return max(256, min(2048, rounded))
        return v

    @field_validator('output_format')
    @classmethod
    def validate_format(cls, v):
        """Validate output format."""
        allowed_formats = {'png', 'jpeg', 'jpg', 'webp'}
        if v.lower() not in allowed_formats:
            raise ValueError(f"Output format must be one of: {', '.join(allowed_formats)}")
        return v.lower()

    model_config = {
        "extra": "forbid",  # Reject unknown fields
        "json_schema_extra": {
            "example": {
                "prompt": "A beautiful sunset over mountains with realistic lighting",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "seed": 42,
                "output_format": "png",
                "quality": 95
            }
        }
    }


class ImageOutput(BaseModel):
    """Single image output with metadata."""

    image_data: str = Field(
        ...,
        description="Base64-encoded image data"
    )

    format: str = Field(
        ...,
        description="Image format (png, jpeg, webp)"
    )

    width: int = Field(
        ...,
        description="Actual image width in pixels"
    )

    height: int = Field(
        ...,
        description="Actual image height in pixels"
    )

    file_size: int = Field(
        ...,
        description="Image file size in bytes"
    )

    seed_used: Optional[int] = Field(
        default=None,
        description="Actual seed used for generation"
    )


class TextToImageResponse(BaseModel):
    """Response schema for text-to-image generation."""

    model_config = ConfigDict(protected_namespaces=())

    # Success/status information
    status: str = Field(
        default="success",
        description="Response status (success, error)"
    )

    # Generated images
    images: List[ImageOutput] = Field(
        default_factory=list,
        description="List of generated images with metadata"
    )

    # Request metadata
    prompt_used: str = Field(
        ...,
        description="Actual prompt used for generation"
    )

    # Generation parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for generation"
    )

    # Timing information
    inference_time: float = Field(
        ...,
        description="Total inference time in seconds"
    )

    # Model information
    model_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the model used"
    )

    # Memory usage
    peak_memory_mb: Optional[float] = Field(
        default=None,
        description="Peak memory usage during inference (MB)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "images": [
                    {
                        "image_data": "iVBORw0KGgoAAAANSUhEUgAAA...",
                        "format": "png",
                        "width": 1024,
                        "height": 1024,
                        "file_size": 2457600,
                        "seed_used": 42
                    }
                ],
                "prompt_used": "A beautiful sunset over mountains with realistic lighting",
                "parameters": {
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0
                },
                "inference_time": 12.34,
                "model_info": {
                    "name": "FLUX.1-schnell-fp8",
                    "version": "1.0",
                    "precision": "fp8"
                },
                "peak_memory_mb": 14580.5
            }
        }
    }


class TextToImageError(BaseModel):
    """Error response for text-to-image requests."""

    status: str = Field(
        default="error",
        description="Response status"
    )

    error_type: str = Field(
        ...,
        description="Type of error (validation, model, inference, resource)"
    )

    error_message: str = Field(
        ...,
        description="Human-readable error description"
    )

    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context and debugging information"
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier for debugging"
    )

    timestamp: str = Field(
        ...,
        description="ISO timestamp when error occurred"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "error",
                "error_type": "validation",
                "error_message": "Invalid image dimensions: width must be between 256 and 2048 pixels",
                "error_details": {
                    "field": "width",
                    "value": 4096,
                    "constraint": "le=2048"
                },
                "request_id": "flux_123",
                "timestamp": "2025-09-29T10:30:00Z"
            }
        }
    }


# Utility functions for schema validation
def validate_text_to_image_request(data: Dict[str, Any]) -> TextToImageRequest:
    """
    Validate and parse text-to-image request data.

    Args:
        data: Raw request data dictionary

    Returns:
        Validated TextToImageRequest object

    Raises:
        ValidationError: If data doesn't match schema
    """
    try:
        return TextToImageRequest(**data)
    except Exception as e:
        from ..utils.exceptions import ValidationError
        raise ValidationError(f"Invalid text-to-image request: {str(e)}")


def create_success_response(
    images: List[ImageOutput],
    prompt: str,
    parameters: Dict[str, Any],
    inference_time: float,
    model_info: Dict[str, Any],
    peak_memory_mb: Optional[float] = None
) -> TextToImageResponse:
    """
    Create a successful text-to-image response.

    Args:
        images: List of generated images
        prompt: Prompt used for generation
        parameters: Generation parameters
        inference_time: Time taken for inference
        model_info: Model information
        peak_memory_mb: Peak memory usage

    Returns:
        TextToImageResponse object
    """
    return TextToImageResponse(
        status="success",
        images=images,
        prompt_used=prompt,
        parameters=parameters,
        inference_time=inference_time,
        model_info=model_info,
        peak_memory_mb=peak_memory_mb
    )


def create_error_response(
    error_type: str,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> TextToImageError:
    """
    Create an error response for text-to-image requests.

    Args:
        error_type: Type of error
        error_message: Human-readable error message
        error_details: Additional error context
        request_id: Request identifier

    Returns:
        TextToImageError object
    """
    from datetime import datetime

    return TextToImageError(
        error_type=error_type,
        error_message=error_message,
        error_details=error_details,
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )