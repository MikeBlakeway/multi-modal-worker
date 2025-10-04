"""
ControlNet Control Processors

Implements image preprocessing for ControlNet guidance including Canny edge
detection and depth estimation. Provides utilities for converting input images
into control maps suitable for guided image generation.
"""

import logging
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
import time
from datetime import datetime
import base64
from io import BytesIO

try:
    from ..utils.exceptions import ValidationError, ProcessingError
    from ..utils.image_utils import ImageProcessor, ImageValidator
except ImportError:
    from src.utils.exceptions import ValidationError, ProcessingError
    from src.utils.image_utils import ImageProcessor, ImageValidator

logger = logging.getLogger(__name__)


class ControlProcessor:
    """
    Base class for control image processors.

    Provides common functionality for preprocessing images into control maps
    for ControlNet guidance.
    """

    def __init__(self):
        """Initialize control processor."""
        self.image_validator = ImageValidator()
        self.image_processor = ImageProcessor()

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str],
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Process an input image into a control map.

        Args:
            image: Input image (PIL Image, numpy array, or base64 string)
            **kwargs: Processor-specific parameters

        Returns:
            Tuple of (processed_control_image, processing_info)

        Raises:
            ValidationError: If input image is invalid
            ProcessingError: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process method")

    def _prepare_image(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """
        Convert input to PIL Image and validate.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image instance

        Raises:
            ValidationError: If image is invalid or cannot be processed
        """
        if isinstance(image, str):
            # Base64 encoded image
            try:
                image_data = base64.b64decode(image)
                pil_image = Image.open(BytesIO(image_data))
            except Exception as e:
                raise ValidationError("base64_image", str(image)[:50], f"Invalid base64 image data: {str(e)}")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image.copy()
        else:
            raise ValidationError("image_type", str(type(image)), f"Unsupported image type: {type(image)}")

        # Validate image
        try:
            if hasattr(self.image_validator, 'validate_image'):
                if not self.image_validator.validate_image(pil_image):
                    raise ValidationError("image_content", "PIL Image", "Image validation failed")

            # Convert to RGB if necessary
            if pil_image.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            return pil_image
        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError("image_preparation", f"Failed to prepare image: {str(e)}")


class CannyProcessor(ControlProcessor):
    """
    Canny edge detection processor for ControlNet.

    Converts input images to edge maps using Canny edge detection algorithm
    for structural guidance in image generation.
    """

    DEFAULT_LOW_THRESHOLD = 100
    DEFAULT_HIGH_THRESHOLD = 200

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str],
        low_threshold: int = DEFAULT_LOW_THRESHOLD,
        high_threshold: int = DEFAULT_HIGH_THRESHOLD,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Process image using Canny edge detection.

        Args:
            image: Input image for edge detection
            low_threshold: Lower threshold for Canny algorithm
            high_threshold: Upper threshold for Canny algorithm
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (edge_image, processing_info)

        Raises:
            ValidationError: If parameters are invalid
            ProcessingError: If edge detection fails
        """
        start_time = time.time()

        # Validate parameters
        if low_threshold < 1 or low_threshold > 255:
            raise ValidationError("low_threshold", str(low_threshold), "must be between 1-255")
        if high_threshold < 1 or high_threshold > 255:
            raise ValidationError("high_threshold", str(high_threshold), "must be between 1-255")
        if high_threshold <= low_threshold:
            raise ValidationError("threshold_order", f"{high_threshold}<={low_threshold}", "high_threshold must be > low_threshold")

        try:
            # Prepare input image
            pil_image = self._prepare_image(image)
            original_size = pil_image.size

            # Convert to numpy array for OpenCV processing
            image_array = np.array(pil_image)

            # Convert to grayscale for Canny detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, low_threshold, high_threshold)

            # Convert back to RGB (edges are white on black background)
            edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Convert back to PIL Image
            edge_image = Image.fromarray(edge_rgb)

            processing_time = (time.time() - start_time) * 1000

            processing_info = {
                'control_type': 'canny',
                'original_width': original_size[0],
                'original_height': original_size[1],
                'processed_width': edge_image.width,
                'processed_height': edge_image.height,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'processing_time_ms': processing_time,
                'edge_pixels': int(np.sum(edges > 0)),  # Count of edge pixels
                'edge_density': float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
            }

            logger.debug(f"Canny processing completed in {processing_time:.2f}ms, "
                        f"edge density: {processing_info['edge_density']:.3f}")

            return edge_image, processing_info

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError("canny_edge_detection", f"Canny edge detection failed: {str(e)}")


class DepthProcessor(ControlProcessor):
    """
    Depth estimation processor for ControlNet.

    Converts input images to depth maps for depth-guided image generation.
    Currently implements MiDaS-based depth estimation.
    """

    def __init__(self):
        """Initialize depth processor."""
        super().__init__()
        self._midas_model = None
        self._midas_transform = None
        self._device = None

    def _load_midas_model(self):
        """Load MiDaS model for depth estimation."""
        if self._midas_model is not None:
            return

        try:
            import torch

            # Set device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load MiDaS model (using small model for speed)
            self._midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self._midas_model.to(self._device)
            self._midas_model.eval()

            # Load transform
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self._midas_transform = midas_transforms.small_transform

            logger.info(f"MiDaS depth model loaded on {self._device}")

        except Exception as e:
            raise ProcessingError("midas_model_loading", f"Failed to load MiDaS depth model: {str(e)}")

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str],
        normalize_depth: bool = True,
        invert_depth: bool = True,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Process image using depth estimation.

        Args:
            image: Input image for depth estimation
            normalize_depth: Whether to normalize depth values to 0-255 range
            invert_depth: Whether to invert depth (far=dark, near=bright)
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (depth_image, processing_info)

        Raises:
            ProcessingError: If depth estimation fails
        """
        start_time = time.time()

        try:
            # Prepare input image
            pil_image = self._prepare_image(image)
            original_size = pil_image.size

            # Load MiDaS model if needed
            self._load_midas_model()

            # Convert to numpy array
            image_array = np.array(pil_image)

            # Prepare input for MiDaS
            input_batch = self._midas_transform(image_array).to(self._device)

            # Run depth estimation
            import torch
            with torch.no_grad():
                prediction = self._midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_array.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_map = prediction.cpu().numpy()

            # Process depth map
            if normalize_depth:
                # Normalize to 0-255 range
                depth_min, depth_max = depth_map.min(), depth_map.max()
                if depth_max > depth_min:
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                else:
                    depth_map = np.zeros_like(depth_map)
                depth_map = (depth_map * 255).astype(np.uint8)
            else:
                depth_map = depth_map.astype(np.uint8)

            if invert_depth:
                depth_map = 255 - depth_map

            # Convert to RGB image
            depth_rgb = np.stack([depth_map, depth_map, depth_map], axis=-1)
            depth_image = Image.fromarray(depth_rgb)

            processing_time = (time.time() - start_time) * 1000

            processing_info = {
                'control_type': 'depth',
                'original_width': original_size[0],
                'original_height': original_size[1],
                'processed_width': depth_image.width,
                'processed_height': depth_image.height,
                'normalize_depth': normalize_depth,
                'invert_depth': invert_depth,
                'processing_time_ms': processing_time,
                'depth_range': {
                    'min': float(depth_map.min()),
                    'max': float(depth_map.max()),
                    'mean': float(depth_map.mean())
                }
            }

            logger.debug(f"Depth processing completed in {processing_time:.2f}ms")

            return depth_image, processing_info

        except Exception as e:
            if isinstance(e, (ValidationError, ProcessingError)):
                raise
            raise ProcessingError("depth_estimation", f"Depth estimation failed: {str(e)}")


class ControlProcessorFactory:
    """
    Factory class for creating control processors.

    Provides a unified interface for creating different types of control
    processors based on control type specifications.
    """

    _processors = {
        'canny': CannyProcessor,
        'depth': DepthProcessor,
    }

    @classmethod
    def create_processor(cls, control_type: str) -> ControlProcessor:
        """
        Create a control processor for the specified type.

        Args:
            control_type: Type of control processor ('canny', 'depth')

        Returns:
            ControlProcessor instance

        Raises:
            ValidationError: If control type is not supported
        """
        if control_type not in cls._processors:
            supported = ', '.join(cls._processors.keys())
            raise ValidationError("control_type", control_type, f"Unsupported control type '{control_type}'. Supported types: {supported}")

        processor_class = cls._processors[control_type]
        return processor_class()

    @classmethod
    def get_supported_types(cls) -> list:
        """
        Get list of supported control types.

        Returns:
            List of supported control type strings
        """
        return list(cls._processors.keys())

    @classmethod
    def register_processor(cls, control_type: str, processor_class: type):
        """
        Register a new control processor type.

        Args:
            control_type: Name of the control type
            processor_class: ControlProcessor subclass
        """
        if not issubclass(processor_class, ControlProcessor):
            raise ValueError("Processor class must inherit from ControlProcessor")

        cls._processors[control_type] = processor_class
        logger.info(f"Registered control processor: {control_type}")


def process_control_image(
    image: Union[Image.Image, np.ndarray, str],
    control_type: str,
    **kwargs
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Convenience function for processing control images.

    Args:
        image: Input image for control processing
        control_type: Type of control processing ('canny', 'depth')
        **kwargs: Parameters specific to the processor

    Returns:
        Tuple of (processed_control_image, processing_info)

    Raises:
        ValidationError: If control type is unsupported
        ProcessingError: If processing fails
    """
    processor = ControlProcessorFactory.create_processor(control_type)
    return processor.process(image, **kwargs)