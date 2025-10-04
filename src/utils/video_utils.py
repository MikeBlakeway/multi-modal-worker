"""
Video Utilities

Provides video encoding, frame processing, and format conversion utilities
for AnimateDiff image-to-video generation and other video processing tasks.
"""

import logging
import base64
import tempfile
import os
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import imageio

logger = logging.getLogger(__name__)


class VideoEncoder:
    """
    Handles video encoding and format conversion for generated frame sequences.

    Supports various output formats compatible with web display components
    and provides optimized encoding parameters for different use cases.
    """

    # Supported video formats and their configurations
    SUPPORTED_FORMATS = {
        'mp4': {
            'extension': '.mp4',
            'codec': 'libx264',
            'mime_type': 'video/mp4',
            'web_compatible': True
        },
        'gif': {
            'extension': '.gif',
            'codec': 'gif',
            'mime_type': 'image/gif',
            'web_compatible': True
        },
        'webm': {
            'extension': '.webm',
            'codec': 'libvpx-vp9',
            'mime_type': 'video/webm',
            'web_compatible': True
        }
    }

    def __init__(self, format: str = 'mp4', quality: str = 'medium', fps: float = None, output_format: str = None):
        """
        Initialize video encoder.

        Args:
            format: Output video format ('mp4', 'gif', 'webm')
            quality: Encoding quality ('low', 'medium', 'high')
            fps: Frame rate override (optional)
            output_format: Alias for format parameter (for test compatibility)
        """
        # Handle output_format parameter as alias for format
        if output_format is not None:
            format = output_format

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.SUPPORTED_FORMATS.keys())}")

        self.format = format
        self.output_format = format  # Alias for compatibility with tests
        self.quality = quality
        self.format_config = self.SUPPORTED_FORMATS[format]

        # Quality settings
        self.quality_params = self._get_quality_params(quality)

        # Set fps attribute - use override if provided, otherwise default to 8.0
        if fps is not None:
            self.fps = fps
        else:
            self.fps = 8.0  # Default to 8.0 for test compatibility

    def _get_quality_params(self, quality: str) -> Dict[str, Any]:
        """Get encoding parameters for specified quality level."""
        params = {
            'low': {
                'fps': 8,
                'bitrate': '500k',
                'crf': 28,
                'preset': 'fast'
            },
            'medium': {
                'fps': 12,
                'bitrate': '1M',
                'crf': 23,
                'preset': 'medium'
            },
            'high': {
                'fps': 16,
                'bitrate': '2M',
                'crf': 18,
                'preset': 'slow'
            }
        }
        return params.get(quality, params['medium'])

    def encode_frames_to_video(
        self,
        frames: List[np.ndarray],
        fps: int = 8,
        loop: bool = False
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encode a sequence of frames to video format.

        Args:
            frames: List of numpy arrays representing video frames
            fps: Frames per second for output video
            loop: Whether to create a seamless loop

        Returns:
            Tuple of (video_bytes, video_info_dict)
        """
        if not frames:
            raise ValueError("No frames provided for encoding")

        # Process frames for looping if requested
        if loop and len(frames) > 1:
            frames = self._create_loop_frames(frames)

        # Create temporary file for video output
        with tempfile.NamedTemporaryFile(
            suffix=self.format_config['extension'],
            delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            # Encode video based on format
            if self.format == 'gif':
                self._encode_gif(frames, temp_path, fps)
            else:
                self._encode_mp4_webm(frames, temp_path, fps)

            # Read encoded video
            with open(temp_path, 'rb') as f:
                video_bytes = f.read()

            # Get video info
            video_info = self._get_video_info(frames, fps, len(video_bytes))

            return video_bytes, video_info

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _create_loop_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Create seamless loop by adding reverse frames (excluding duplicates)."""
        # Add reverse frames (excluding first and last to avoid duplicates)
        if len(frames) > 2:
            reverse_frames = frames[-2:0:-1]  # Reverse excluding first and last
            return frames + reverse_frames
        return frames

    def _encode_gif(self, frames: List[np.ndarray], output_path: str, fps: int):
        """Encode frames as GIF."""
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Convert to PIL Image
            if len(frame.shape) == 3:
                pil_frame = Image.fromarray(frame, 'RGB')
            else:
                pil_frame = Image.fromarray(frame, 'L')

            pil_frames.append(pil_frame)

        # Calculate duration per frame in milliseconds
        duration = int(1000 / fps)

        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )

    def _encode_mp4_webm(self, frames: List[np.ndarray], output_path: str, fps: int):
        """Encode frames as MP4 or WebM using imageio."""
        # Ensure frames are uint8
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            processed_frames.append(frame)

        # Configure codec parameters - use compatible parameters for imageio
        if self.format == 'mp4':
            codec_params = {
                'codec': 'libx264',
                'fps': fps,
                'bitrate': self.quality_params['bitrate']
                # Note: preset and crf not supported by current imageio version
            }
        else:  # webm
            codec_params = {
                'codec': 'libvpx-vp9',
                'fps': fps,
                'bitrate': self.quality_params['bitrate']
            }

        # Write video
        with imageio.get_writer(output_path, **codec_params) as writer:
            for frame in processed_frames:
                writer.append_data(frame)

    def _get_video_info(self, frames: List[np.ndarray], fps: int, size_bytes: int) -> Dict[str, Any]:
        """Generate video information dictionary."""
        if not frames:
            raise ValueError("No frames to analyze")

        height, width = frames[0].shape[:2]
        num_frames = len(frames)
        duration = num_frames / fps

        return {
            'width': int(width),
            'height': int(height),
            'fps': fps,
            'num_frames': num_frames,
            'duration': round(duration, 2),
            'format': self.format,
            'size_bytes': size_bytes
        }


class FrameProcessor:
    """
    Utilities for processing and manipulating video frames.
    """

    @staticmethod
    def resize_frames(
        frames: List[np.ndarray],
        target_size: Tuple[int, int],
        interpolation: str = 'bilinear'
    ) -> List[np.ndarray]:
        """
        Resize all frames to target dimensions.

        Args:
            frames: List of frame arrays
            target_size: (width, height) target dimensions
            interpolation: Interpolation method

        Returns:
            List of resized frames
        """
        resized_frames = []
        width, height = target_size

        for frame in frames:
            # Convert to PIL for resizing
            if frame.dtype != np.uint8:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame

            pil_frame = Image.fromarray(frame_uint8)

            # Resize using PIL
            if interpolation == 'bilinear':
                resample = Image.BILINEAR
            elif interpolation == 'bicubic':
                resample = Image.BICUBIC
            else:
                resample = Image.NEAREST

            resized_pil = pil_frame.resize((width, height), resample)

            # Convert back to numpy
            resized_frame = np.array(resized_pil)
            resized_frames.append(resized_frame)

        return resized_frames

    @staticmethod
    def interpolate_frames(
        frames: List[np.ndarray],
        interpolation_factor: int = 2
    ) -> List[np.ndarray]:
        """
        Interpolate between frames to increase frame rate.

        Args:
            frames: Input frames
            interpolation_factor: Factor to increase frame count by

        Returns:
            Interpolated frame sequence
        """
        if interpolation_factor <= 1 or len(frames) < 2:
            return frames

        interpolated = [frames[0]]

        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            # Generate intermediate frames
            for j in range(1, interpolation_factor):
                alpha = j / interpolation_factor
                interpolated_frame = (
                    (1 - alpha) * current_frame + alpha * next_frame
                ).astype(current_frame.dtype)
                interpolated.append(interpolated_frame)

            interpolated.append(next_frame)

        return interpolated

    @staticmethod
    def normalize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize frame values to [0, 1] range.

        Args:
            frames: Input frames

        Returns:
            Normalized frames
        """
        normalized = []
        for frame in frames:
            if frame.dtype == np.uint8:
                normalized_frame = frame.astype(np.float32) / 255.0
            else:
                # Assume already in [0, 1] range
                normalized_frame = frame.astype(np.float32)

            normalized.append(normalized_frame)

        return normalized


def encode_video_to_base64(
    frames: List[np.ndarray],
    fps: int = 8,
    format: str = 'mp4',
    quality: str = 'high',
    loop: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to encode frames to base64-encoded video.

    Args:
        frames: List of video frames as numpy arrays
        fps: Frames per second
        format: Video format ('mp4', 'gif', 'webm')
        quality: Encoding quality ('low', 'medium', 'high')
        loop: Create seamless loop

    Returns:
        Tuple of (base64_video_string, video_info_dict)
    """
    encoder = VideoEncoder(format=format, quality=quality)
    video_bytes, video_info = encoder.encode_frames_to_video(
        frames=frames,
        fps=fps,
        loop=loop
    )

    # Encode to base64
    base64_video = base64.b64encode(video_bytes).decode('utf-8')

    return base64_video, video_info


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy array image.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        Numpy array representing the image
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        try:
            # Load image using PIL
            pil_image = Image.open(temp_path)

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array
            image_array = np.array(pil_image)

            return image_array

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")


def validate_video_frames(frames: List[np.ndarray]) -> bool:
    """
    Validate that frames form a consistent video sequence.

    Args:
        frames: List of frame arrays

    Returns:
        True if frames are valid, raises ValueError otherwise
    """
    if not frames:
        raise ValueError("No frames provided")

    if len(frames) < 2:
        raise ValueError("At least 2 frames required for video")

    # Check frame consistency
    first_shape = frames[0].shape
    for i, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(
                f"Frame {i} has inconsistent shape: {frame.shape} vs {first_shape}"
            )

        if len(frame.shape) not in [2, 3]:
            raise ValueError(
                f"Frame {i} has invalid dimensions: {frame.shape}"
            )

        if len(frame.shape) == 3 and frame.shape[2] not in [1, 3, 4]:
            raise ValueError(
                f"Frame {i} has invalid channel count: {frame.shape[2]}"
            )

    return True