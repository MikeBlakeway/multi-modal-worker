"""
LTX-Video Specific Utility Functions

Provides specialized video processing utilities for LTX-Video text-to-video generation,
including format optimization, quality enhancement, metadata extraction, and validation.
Extends the base video utilities with LTX-specific features.
"""

import logging
import base64
import tempfile
import os
import json
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import imageio
import cv2
from datetime import datetime, timedelta
import hashlib
import time

# Import base video utilities
from .video_utils import VideoEncoder, FrameProcessor

logger = logging.getLogger(__name__)


class LTXVideoProcessor:
    """
    Specialized video processor for LTX-Video generated content.

    Provides optimization, quality enhancement, and format conversion
    specifically tailored for DiT-based text-to-video generation.
    """

    # LTX-Video optimized encoding presets
    LTX_PRESETS = {
        'mobile': {
            'max_resolution': (480, 854),  # 9:16 mobile
            'max_fps': 15,
            'max_duration': 10.0,
            'quality': 'medium',
            'format': 'mp4'
        },
        'web': {
            'max_resolution': (720, 1280),  # 9:16 web
            'max_fps': 24,
            'max_duration': 15.0,
            'quality': 'high',
            'format': 'mp4'
        },
        'social': {
            'max_resolution': (1080, 1920),  # 9:16 social media
            'max_fps': 30,
            'max_duration': 30.0,
            'quality': 'high',
            'format': 'mp4'
        },
        'preview': {
            'max_resolution': (320, 568),  # Small preview
            'max_fps': 12,
            'max_duration': 5.0,
            'quality': 'low',
            'format': 'gif'
        }
    }

    def __init__(self, preset: str = 'web'):
        """Initialize LTX video processor with preset configuration."""
        if preset not in self.LTX_PRESETS:
            raise ValueError(f"Unsupported preset: {preset}. Available: {list(self.LTX_PRESETS.keys())}")

        self.preset = preset
        self.config = self.LTX_PRESETS[preset].copy()
        self.encoder = VideoEncoder(
            format=self.config['format'],
            quality=self.config['quality']
        )

    def process_ltx_output(
        self,
        video_data: Union[str, bytes, np.ndarray, List[np.ndarray]],
        fps: int = 8,
        enhance_quality: bool = True,
        add_metadata: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process LTX-Video model output into optimized video format.

        Args:
            video_data: Raw video data from LTX model (various formats supported)
            fps: Original frame rate
            enhance_quality: Whether to apply quality enhancement
            add_metadata: Whether to add LTX-specific metadata

        Returns:
            Tuple of (base64_encoded_video, video_metadata)
        """
        start_time = time.time()

        # Convert input to frame sequence
        frames = self._parse_video_data(video_data)

        # Validate frames
        self._validate_frames(frames)

        # Apply quality enhancement if requested
        if enhance_quality:
            frames = self._enhance_quality(frames)

        # Optimize for preset
        frames, target_fps = self._optimize_for_preset(frames, fps)

        # Encode to video
        video_bytes, base_info = self.encoder.encode_frames_to_video(
            frames,
            fps=target_fps,
            loop=False
        )

        # Encode to base64
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        # Generate comprehensive metadata
        metadata = self._generate_metadata(
            base_info,
            frames,
            target_fps,
            add_metadata,
            time.time() - start_time
        )

        return video_base64, metadata

    def _parse_video_data(self, video_data: Union[str, bytes, np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """Parse various video data formats into frame sequence."""
        if isinstance(video_data, str):
            # Assume base64 encoded video or file path
            if os.path.exists(video_data):
                return self._load_frames_from_file(video_data)
            else:
                # Try base64 decode
                try:
                    decoded_bytes = base64.b64decode(video_data)
                    return self._load_frames_from_bytes(decoded_bytes)
                except Exception as e:
                    raise ValueError(f"Invalid video data format: {e}")

        elif isinstance(video_data, bytes):
            return self._load_frames_from_bytes(video_data)

        elif isinstance(video_data, np.ndarray):
            # Single frame or video tensor
            if len(video_data.shape) == 3:
                return [video_data]  # Single frame
            elif len(video_data.shape) == 4:
                return [video_data[i] for i in range(video_data.shape[0])]  # Video tensor
            else:
                raise ValueError(f"Unsupported array shape: {video_data.shape}")

        elif isinstance(video_data, list):
            # Already frame sequence
            if all(isinstance(frame, np.ndarray) for frame in video_data):
                return video_data
            else:
                raise ValueError("List must contain numpy arrays")

        else:
            raise ValueError(f"Unsupported video data type: {type(video_data)}")

    def _load_frames_from_file(self, file_path: str) -> List[np.ndarray]:
        """Load frames from video file."""
        try:
            reader = imageio.get_reader(file_path)
            frames = [frame for frame in reader]
            reader.close()
            return frames
        except Exception as e:
            raise ValueError(f"Failed to load video file {file_path}: {e}")

    def _load_frames_from_bytes(self, video_bytes: bytes) -> List[np.ndarray]:
        """Load frames from video bytes."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        try:
            return self._load_frames_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _validate_frames(self, frames: List[np.ndarray]) -> None:
        """Validate frame sequence for LTX processing."""
        if not frames:
            raise ValueError("No frames to process")

        if len(frames) > 257:  # LTX max frames (8*32)+1
            logger.warning(f"Frame count {len(frames)} exceeds LTX maximum (257), will be truncated")

        # Check frame consistency
        first_shape = frames[0].shape
        for i, frame in enumerate(frames[1:], 1):
            if frame.shape != first_shape:
                raise ValueError(f"Frame {i} shape {frame.shape} differs from frame 0 shape {first_shape}")

        # Check resolution constraints (should be divisible by 32 for LTX)
        height, width = first_shape[:2]
        if height % 32 != 0 or width % 32 != 0:
            logger.warning(f"Frame resolution {width}x{height} not divisible by 32, may affect quality")

    def _enhance_quality(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply quality enhancement to frames."""
        enhanced_frames = []

        for frame in frames:
            # Convert to PIL for enhancement
            if frame.dtype != np.uint8:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame

            pil_frame = Image.fromarray(frame_uint8)

            # Apply subtle enhancements
            enhanced_frame = pil_frame

            # Slight sharpening
            enhanced_frame = enhanced_frame.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))

            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced_frame)
            enhanced_frame = enhancer.enhance(1.1)

            # Slight color enhancement
            enhancer = ImageEnhance.Color(enhanced_frame)
            enhanced_frame = enhancer.enhance(1.05)

            # Convert back to numpy
            enhanced_array = np.array(enhanced_frame)
            enhanced_frames.append(enhanced_array)

        return enhanced_frames

    def _optimize_for_preset(self, frames: List[np.ndarray], original_fps: int) -> Tuple[List[np.ndarray], int]:
        """Optimize frames for the selected preset."""
        # Truncate frames if needed
        max_frames = int(self.config['max_duration'] * self.config['max_fps'])
        if len(frames) > max_frames:
            logger.info(f"Truncating from {len(frames)} to {max_frames} frames for preset {self.preset}")
            frames = frames[:max_frames]

        # Resize if needed
        current_height, current_width = frames[0].shape[:2]
        max_width, max_height = self.config['max_resolution']

        if current_width > max_width or current_height > max_height:
            # Calculate aspect-preserving resize
            width_ratio = max_width / current_width
            height_ratio = max_height / current_height
            scale_ratio = min(width_ratio, height_ratio)

            new_width = int(current_width * scale_ratio)
            new_height = int(current_height * scale_ratio)

            # Ensure divisible by 32 (round down to nearest 32)
            new_width = (new_width // 32) * 32
            new_height = (new_height // 32) * 32

            logger.info(f"Resizing from {current_width}x{current_height} to {new_width}x{new_height}")
            frames = FrameProcessor.resize_frames(frames, (new_width, new_height))

        # Adjust FPS
        target_fps = min(original_fps, self.config['max_fps'])

        return frames, target_fps

    def _generate_metadata(
        self,
        base_info: Dict[str, Any],
        frames: List[np.ndarray],
        fps: int,
        include_ltx_metadata: bool,
        processing_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive video metadata."""
        metadata = base_info.copy()

        # Add LTX-specific metadata
        if include_ltx_metadata:
            metadata.update({
                'ltx_video': {
                    'model_type': 'LTX-Video',
                    'generation_method': 'text-to-video',
                    'preset': self.preset,
                    'processing_time_ms': round(processing_time * 1000, 2),
                    'quality_enhanced': True,
                    'frame_count_original': len(frames),
                    'aspect_ratio': round(metadata['width'] / metadata['height'], 2),
                    'generated_at': datetime.utcnow().isoformat() + 'Z'
                }
            })

        # Add quality metrics
        metadata['quality_metrics'] = self._calculate_quality_metrics(frames)

        # Add compression info
        metadata['compression'] = {
            'preset': self.config['quality'],
            'format': self.config['format'],
            'estimated_bitrate': self._estimate_bitrate(metadata['size_bytes'], metadata['duration'])
        }

        return metadata

    def _calculate_quality_metrics(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate quality metrics for the frame sequence."""
        if not frames:
            return {}

        # Calculate average brightness
        avg_brightness = np.mean([np.mean(frame) for frame in frames])

        # Calculate frame variance (motion indicator)
        if len(frames) > 1:
            frame_diffs = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                frame_diffs.append(diff)
            avg_motion = np.mean(frame_diffs)
        else:
            avg_motion = 0.0

        # Resolution category
        height, width = frames[0].shape[:2]
        if width >= 1920 or height >= 1080:
            resolution_category = 'HD+'
        elif width >= 1280 or height >= 720:
            resolution_category = 'HD'
        elif width >= 640 or height >= 480:
            resolution_category = 'SD'
        else:
            resolution_category = 'Low'

        return {
            'average_brightness': round(avg_brightness, 2),
            'average_motion': round(avg_motion, 2),
            'resolution_category': resolution_category,
            'frame_consistency': self._check_frame_consistency(frames)
        }

    def _check_frame_consistency(self, frames: List[np.ndarray]) -> float:
        """Check consistency between consecutive frames (0.0 to 1.0)."""
        if len(frames) < 2:
            return 1.0

        similarities = []
        for i in range(1, min(len(frames), 11)):  # Check first 10 transitions
            # Calculate structural similarity
            similarity = self._calculate_frame_similarity(frames[i-1], frames[i])
            similarities.append(similarity)

        return round(np.mean(similarities), 3)

    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames."""
        # Simple correlation-based similarity
        f1_flat = frame1.flatten().astype(float)
        f2_flat = frame2.flatten().astype(float)

        # Normalize
        f1_norm = (f1_flat - np.mean(f1_flat)) / (np.std(f1_flat) + 1e-10)
        f2_norm = (f2_flat - np.mean(f2_flat)) / (np.std(f2_flat) + 1e-10)

        # Calculate correlation
        correlation = np.corrcoef(f1_norm, f2_norm)[0, 1]

        # Convert to 0-1 range (handle NaN)
        if np.isnan(correlation):
            return 0.5
        else:
            return (correlation + 1) / 2

    def _estimate_bitrate(self, size_bytes: int, duration: float) -> str:
        """Estimate video bitrate in human-readable format."""
        if duration <= 0:
            return "unknown"

        bits_per_second = (size_bytes * 8) / duration

        if bits_per_second >= 1_000_000:
            return f"{bits_per_second / 1_000_000:.1f}Mbps"
        elif bits_per_second >= 1_000:
            return f"{bits_per_second / 1_000:.0f}kbps"
        else:
            return f"{bits_per_second:.0f}bps"


class LTXVideoValidator:
    """
    Validator for LTX-Video output and processing requirements.
    """

    # LTX-Video constraints
    MAX_FRAMES = 257  # (8*32) + 1
    MAX_DURATION = 32.0  # seconds at 8fps
    MIN_RESOLUTION = (256, 256)
    MAX_RESOLUTION = (1280, 1280)
    SUPPORTED_ASPECT_RATIOS = [
        (16, 9), (9, 16), (1, 1), (4, 3), (3, 4), (21, 9)
    ]

    @classmethod
    def validate_generation_params(
        cls,
        width: int,
        height: int,
        num_frames: int,
        fps: int
    ) -> Dict[str, Any]:
        """
        Validate parameters for LTX-Video generation.

        Returns validation result with warnings and errors.
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        # Check resolution constraints
        if width < cls.MIN_RESOLUTION[0] or height < cls.MIN_RESOLUTION[1]:
            result['errors'].append(f"Resolution {width}x{height} below minimum {cls.MIN_RESOLUTION}")
            result['valid'] = False

        if width > cls.MAX_RESOLUTION[0] or height > cls.MAX_RESOLUTION[1]:
            result['errors'].append(f"Resolution {width}x{height} above maximum {cls.MAX_RESOLUTION}")
            result['valid'] = False

        # Check if dimensions are divisible by 32
        if width % 32 != 0 or height % 32 != 0:
            result['warnings'].append(f"Resolution {width}x{height} not divisible by 32, may affect quality")
            # Suggest nearest valid resolution
            new_width = ((width + 15) // 32) * 32
            new_height = ((height + 15) // 32) * 32
            result['suggestions'].append(f"Consider using {new_width}x{new_height}")

        # Check frame count
        if num_frames > cls.MAX_FRAMES:
            result['errors'].append(f"Frame count {num_frames} exceeds maximum {cls.MAX_FRAMES}")
            result['valid'] = False

        # Check duration
        duration = num_frames / fps
        if duration > cls.MAX_DURATION:
            result['warnings'].append(f"Duration {duration:.1f}s exceeds recommended {cls.MAX_DURATION}s")

        # Check aspect ratio
        aspect_ratio = cls._get_closest_aspect_ratio(width, height)
        if aspect_ratio:
            result['suggestions'].append(f"Closest supported aspect ratio: {aspect_ratio[0]}:{aspect_ratio[1]}")

        return result

    @classmethod
    def _get_closest_aspect_ratio(cls, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Find closest supported aspect ratio."""
        target_ratio = width / height

        closest_ratio = None
        min_diff = float('inf')

        for w, h in cls.SUPPORTED_ASPECT_RATIOS:
            ratio = w / h
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = (w, h)

        return closest_ratio

    @classmethod
    def validate_output_video(cls, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LTX-Video output quality and completeness.
        """
        result = {
            'valid': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }

        score_components = []

        # Check metadata completeness
        required_fields = ['width', 'height', 'fps', 'num_frames', 'duration']
        missing_fields = [field for field in required_fields if field not in video_metadata]

        if missing_fields:
            result['issues'].append(f"Missing metadata fields: {missing_fields}")
            result['valid'] = False
        else:
            score_components.append(20)  # Metadata completeness: 20 points

        # Check quality metrics if available
        if 'quality_metrics' in video_metadata:
            quality_metrics = video_metadata['quality_metrics']

            # Frame consistency
            consistency = quality_metrics.get('frame_consistency', 0)
            if consistency >= 0.8:
                score_components.append(25)  # High consistency: 25 points
            elif consistency >= 0.6:
                score_components.append(15)  # Medium consistency: 15 points
                result['recommendations'].append("Frame consistency could be improved")
            else:
                score_components.append(5)   # Low consistency: 5 points
                result['issues'].append("Low frame consistency detected")

            # Resolution category
            res_category = quality_metrics.get('resolution_category', 'Low')
            if res_category == 'HD+':
                score_components.append(25)
            elif res_category == 'HD':
                score_components.append(20)
            elif res_category == 'SD':
                score_components.append(10)
            else:
                score_components.append(5)
                result['recommendations'].append("Consider higher resolution for better quality")

        # Check duration appropriateness
        duration = video_metadata.get('duration', 0)
        if duration >= 1.0 and duration <= 10.0:
            score_components.append(15)  # Good duration: 15 points
        elif duration > 0:
            score_components.append(10)  # Some duration: 10 points
        else:
            result['issues'].append("Invalid or zero duration")

        # Check file size efficiency
        size_mb = video_metadata.get('size_bytes', 0) / (1024 * 1024)
        if duration > 0:
            mb_per_second = size_mb / duration
            if mb_per_second < 2.0:  # Efficient compression
                score_components.append(15)
            elif mb_per_second < 5.0:  # Reasonable compression
                score_components.append(10)
            else:
                score_components.append(5)
                result['recommendations'].append("Video file size could be optimized")

        # Calculate final quality score
        result['quality_score'] = sum(score_components)

        return result


def create_ltx_video_processor(preset: str = 'web') -> LTXVideoProcessor:
    """Factory function to create LTX video processor with specified preset."""
    return LTXVideoProcessor(preset)


def validate_ltx_generation_request(
    prompt: str,
    width: int = 704,
    height: int = 704,
    num_frames: int = 25,
    fps: int = 8
) -> Dict[str, Any]:
    """
    Convenience function to validate LTX-Video generation parameters.

    Args:
        prompt: Text prompt for generation
        width: Video width in pixels
        height: Video height in pixels
        num_frames: Number of frames to generate
        fps: Frames per second

    Returns:
        Validation result dictionary
    """
    # Validate prompt
    validation_result = LTXVideoValidator.validate_generation_params(width, height, num_frames, fps)

    # Add prompt validation
    if not prompt or len(prompt.strip()) < 3:
        validation_result['errors'].append("Prompt too short (minimum 3 characters)")
        validation_result['valid'] = False
    elif len(prompt) >= 1000:  # Changed from > to >= to match test expectations
        validation_result['warnings'].append("Very long prompt may affect generation quality")

    return validation_result


# Export main classes and functions
__all__ = [
    'LTXVideoProcessor',
    'LTXVideoValidator',
    'create_ltx_video_processor',
    'validate_ltx_generation_request'
]