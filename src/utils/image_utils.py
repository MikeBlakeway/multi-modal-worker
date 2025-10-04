"""
Image Utilities for Multi-Modal Worker

Provides image processing, encoding, validation, and format conversion
utilities for AI-generated images across different modalities.
"""

import io
import base64
import logging
from typing import Optional, Tuple, Union, Dict, Any
from PIL import Image, ImageOps
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for AI-generated content."""

    # Supported image formats
    SUPPORTED_FORMATS = {'PNG', 'JPEG', 'WEBP'}

    # Default quality settings
    DEFAULT_QUALITY = {
        'PNG': None,  # PNG is lossless
        'JPEG': 95,
        'WEBP': 95
    }

    # Format-specific options
    FORMAT_OPTIONS = {
        'PNG': {'optimize': True},
        'JPEG': {'optimize': True, 'progressive': True},
        'WEBP': {'optimize': True, 'method': 6}  # Best compression
    }

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert PyTorch tensor to PIL Image.

        Args:
            tensor: PyTorch tensor with shape (C, H, W) or (H, W, C)

        Returns:
            PIL Image object

        Raises:
            ValueError: If tensor format is not supported
        """
        try:
            # Handle different tensor shapes
            if tensor.dim() == 4:
                # Batch dimension, take first image
                tensor = tensor[0]

            if tensor.dim() == 3:
                # Check if channels first or last
                if tensor.shape[0] in [1, 3, 4]:  # Likely channels first
                    tensor = tensor.permute(1, 2, 0)

            # Ensure tensor is on CPU
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()

            # Convert to numpy
            if tensor.dtype != torch.uint8:
                # Assume tensor is in range [0, 1] or [-1, 1]
                if tensor.min() >= -1.0 and tensor.max() <= 1.0:
                    # Normalize from [-1, 1] or [0, 1] to [0, 255]
                    tensor = (tensor + 1.0) / 2.0 if tensor.min() < 0 else tensor
                    tensor = (tensor * 255).clamp(0, 255)

                tensor = tensor.to(torch.uint8)

            array = tensor.numpy()

            # Handle different channel counts
            if array.shape[-1] == 1:
                # Grayscale
                array = np.squeeze(array, axis=-1)
                return Image.fromarray(array, mode='L')
            elif array.shape[-1] == 3:
                # RGB
                return Image.fromarray(array, mode='RGB')
            elif array.shape[-1] == 4:
                # RGBA
                return Image.fromarray(array, mode='RGBA')
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        except Exception as e:
            logger.error(f"Failed to convert tensor to PIL: {e}")
            raise ValueError(f"Failed to convert tensor to PIL Image: {e}")

    @staticmethod
    def numpy_to_pil(array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.

        Args:
            array: Numpy array with shape (H, W) or (H, W, C)

        Returns:
            PIL Image object
        """
        try:
            # Ensure array is in correct format
            if array.dtype != np.uint8:
                # Normalize to [0, 255]
                if array.max() <= 1.0:
                    array = (array * 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)

            # Handle different shapes
            if array.ndim == 2:
                # Grayscale
                return Image.fromarray(array, mode='L')
            elif array.ndim == 3:
                if array.shape[2] == 3:
                    # RGB
                    return Image.fromarray(array, mode='RGB')
                elif array.shape[2] == 4:
                    # RGBA
                    return Image.fromarray(array, mode='RGBA')
                else:
                    raise ValueError(f"Unsupported channel count: {array.shape[2]}")
            else:
                raise ValueError(f"Unsupported array shape: {array.shape}")

        except Exception as e:
            logger.error(f"Failed to convert numpy array to PIL: {e}")
            raise ValueError(f"Failed to convert numpy array to PIL Image: {e}")

    @classmethod
    def encode_image(
        cls,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        format: str = 'PNG',
        quality: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Encode image to base64 string with specified format and quality.

        Args:
            image: PIL Image, PyTorch tensor, or numpy array
            format: Output format ('PNG', 'JPEG', 'WEBP')
            quality: Quality setting (1-100, ignored for PNG)

        Returns:
            Tuple of (base64_string, file_size_bytes)

        Raises:
            ValueError: If format is not supported or encoding fails
        """
        try:
            # Normalize format
            format = format.upper()
            if format == 'JPG':
                format = 'JPEG'

            if format not in cls.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {format}. Supported: {cls.SUPPORTED_FORMATS}")

            # Convert to PIL Image if needed
            if isinstance(image, torch.Tensor):
                pil_image = cls.tensor_to_pil(image)
            elif isinstance(image, np.ndarray):
                pil_image = cls.numpy_to_pil(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Convert RGBA to RGB for JPEG
            if format == 'JPEG' and pil_image.mode in ['RGBA', 'LA']:
                # Create white background
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'RGBA':
                    background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(pil_image)
                pil_image = background

            # Prepare save options
            save_kwargs = cls.FORMAT_OPTIONS.get(format, {}).copy()

            # Add quality if specified and applicable
            if quality is not None and format in ['JPEG', 'WEBP']:
                save_kwargs['quality'] = max(1, min(100, quality))
            elif quality is None and format in cls.DEFAULT_QUALITY:
                default_quality = cls.DEFAULT_QUALITY[format]
                if default_quality is not None:
                    save_kwargs['quality'] = default_quality

            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, **save_kwargs)
            buffer.seek(0)

            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')

            logger.debug(f"Encoded image: {format}, {len(image_bytes)} bytes, quality={quality}")

            return base64_string, len(image_bytes)

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise ValueError(f"Failed to encode image as {format}: {e}")

    @staticmethod
    def decode_image(base64_string: str) -> Image.Image:
        """
        Decode base64 string to PIL Image.

        Args:
            base64_string: Base64-encoded image data

        Returns:
            PIL Image object

        Raises:
            ValueError: If decoding fails
        """
        try:
            image_bytes = base64.b64decode(base64_string)
            buffer = io.BytesIO(image_bytes)
            image = Image.open(buffer)
            return image

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError(f"Failed to decode base64 image: {e}")

    @staticmethod
    def validate_dimensions(width: int, height: int, max_pixels: int = 4194304) -> Tuple[int, int]:
        """
        Validate and potentially adjust image dimensions.

        Args:
            width: Desired width
            height: Desired height
            max_pixels: Maximum total pixels allowed (default: 2048x2048)

        Returns:
            Tuple of validated (width, height)

        Raises:
            ValueError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

        if width > 2048 or height > 2048:
            raise ValueError("Width and height must not exceed 2048 pixels")

        if width * height > max_pixels:
            raise ValueError(f"Total pixels ({width * height}) exceeds maximum ({max_pixels})")

        # Ensure dimensions are multiples of 8 for optimal performance
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        return width, height

    @staticmethod
    def get_image_info(image: Image.Image) -> Dict[str, Any]:
        """
        Get comprehensive information about an image.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with image information
        """
        return {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'has_transparency': image.mode in ['RGBA', 'LA'] or 'transparency' in image.info,
            'total_pixels': image.width * image.height,
            'aspect_ratio': round(image.width / image.height, 3) if image.height > 0 else 0
        }

    @classmethod
    def resize_if_needed(
        cls,
        image: Image.Image,
        max_width: int = 2048,
        max_height: int = 2048,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image if it exceeds maximum dimensions.

        Args:
            image: PIL Image to potentially resize
            max_width: Maximum allowed width
            max_height: Maximum allowed height
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized image (or original if no resize needed)
        """
        if image.width <= max_width and image.height <= max_height:
            return image

        if maintain_aspect:
            # Calculate scale factor to fit within bounds
            scale_w = max_width / image.width
            scale_h = max_height / image.height
            scale = min(scale_w, scale_h)

            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
        else:
            new_width = min(image.width, max_width)
            new_height = min(image.height, max_height)

        # Ensure dimensions are multiples of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        logger.info(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class ImageValidator:
    """Validates image inputs and parameters."""

    ALLOWED_FORMATS = {'PNG', 'JPEG', 'JPG', 'WEBP'}
    MIN_DIMENSION = 256
    MAX_DIMENSION = 2048
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    @classmethod
    def validate_format(cls, format_str: str) -> str:
        """
        Validate and normalize image format.

        Args:
            format_str: Format string to validate

        Returns:
            Normalized format string

        Raises:
            ValueError: If format is not supported
        """
        format_str = format_str.upper().strip()
        if format_str == 'JPG':
            format_str = 'JPEG'

        if format_str not in cls.ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format: {format_str}. Allowed: {cls.ALLOWED_FORMATS}")

        return format_str

    @classmethod
    def validate_quality(cls, quality: int, format_str: str) -> int:
        """
        Validate image quality setting.

        Args:
            quality: Quality value to validate
            format_str: Image format

        Returns:
            Validated quality value

        Raises:
            ValueError: If quality is invalid for format
        """
        format_str = format_str.upper()

        if format_str == 'PNG':
            # PNG is lossless, quality is ignored
            return 100

        if format_str in ['JPEG', 'WEBP']:
            if not isinstance(quality, int) or quality < 1 or quality > 100:
                raise ValueError("Quality must be an integer between 1 and 100")
            return quality

        raise ValueError(f"Quality setting not applicable for format: {format_str}")

    @classmethod
    def validate_base64_image(cls, base64_string: str) -> Dict[str, Any]:
        """
        Validate base64-encoded image data.

        Args:
            base64_string: Base64 string to validate

        Returns:
            Dictionary with validation results and image info

        Raises:
            ValueError: If image data is invalid
        """
        try:
            # Decode and validate
            image_bytes = base64.b64decode(base64_string)

            if len(image_bytes) > cls.MAX_FILE_SIZE:
                raise ValueError(f"Image file too large: {len(image_bytes)} bytes (max: {cls.MAX_FILE_SIZE})")

            # Try to open as image
            buffer = io.BytesIO(image_bytes)
            image = Image.open(buffer)

            # Validate dimensions
            if image.width < cls.MIN_DIMENSION or image.height < cls.MIN_DIMENSION:
                raise ValueError(f"Image too small: {image.width}x{image.height} (min: {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})")

            if image.width > cls.MAX_DIMENSION or image.height > cls.MAX_DIMENSION:
                raise ValueError(f"Image too large: {image.width}x{image.height} (max: {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})")

            return {
                'valid': True,
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'format': image.format,
                'file_size': len(image_bytes)
            }

        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")


# Convenience functions
def encode_pil_image(image: Image.Image, format: str = 'PNG', quality: Optional[int] = None) -> Tuple[str, int]:
    """Convenience function to encode PIL image."""
    return ImageProcessor.encode_image(image, format, quality)


def encode_tensor_image(tensor: torch.Tensor, format: str = 'PNG', quality: Optional[int] = None) -> Tuple[str, int]:
    """Convenience function to encode tensor image."""
    return ImageProcessor.encode_image(tensor, format, quality)


def decode_base64_image(base64_string: str) -> Image.Image:
    """Convenience function to decode base64 image."""
    return ImageProcessor.decode_image(base64_string)


def validate_image_format(format_str: str) -> str:
    """Convenience function to validate image format."""
    return ImageValidator.validate_format(format_str)