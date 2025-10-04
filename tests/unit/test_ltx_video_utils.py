"""
Tests for LTX-Video Utility Functions

Comprehensive test suite for LTX-Video specific utilities including
video processing, format conversion, quality enhancement, and validation.
Following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile
import base64
import json
import numpy as np
from PIL import Image
import imageio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.ltx_video_utils import (
    LTXVideoProcessor, LTXVideoValidator,
    create_ltx_video_processor, validate_ltx_generation_request
)
from src.utils.video_utils import VideoEncoder, FrameProcessor


class TestLTXVideoProcessor(unittest.TestCase):
    """Test cases for LTX-Video specialized processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = LTXVideoProcessor(preset='web')

        # Create test frames
        self.test_frames = self._create_test_frames()
        self.single_frame = self.test_frames[0]

        # Test video data in various formats
        self.test_video_base64 = self._create_test_video_base64()

    def _create_test_frames(self, count=25, width=704, height=704):
        """Create test frame sequence."""
        frames = []
        for i in range(count):
            # Create gradient frame with slight variation
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add gradient
            for y in range(height):
                frame[y, :, 0] = min(255, (y * 255) // height + i * 2)  # Red gradient with temporal change
                frame[y, :, 1] = min(255, 128 + (i * 3) % 128)          # Green temporal variation
                frame[y, :, 2] = min(255, 255 - (y * 255) // height)    # Blue inverse gradient

            frames.append(frame)

        return frames

    def _create_test_video_base64(self) -> str:
        """Create test video encoded as base64."""
        encoder = VideoEncoder(format='mp4')
        video_bytes, _ = encoder.encode_frames_to_video(self.test_frames[:5], fps=8)
        return base64.b64encode(video_bytes).decode('utf-8')

    def test_processor_initialization(self):
        """Test LTX video processor initialization."""
        # Test valid presets
        for preset in ['mobile', 'web', 'social', 'preview']:
            processor = LTXVideoProcessor(preset=preset)
            self.assertEqual(processor.preset, preset)
            self.assertIn(preset, processor.LTX_PRESETS)

        # Test invalid preset
        with self.assertRaises(ValueError):
            LTXVideoProcessor(preset='invalid')

    def test_preset_configurations(self):
        """Test preset configuration values."""
        # Test web preset (default)
        web_processor = LTXVideoProcessor(preset='web')
        web_config = web_processor.config

        self.assertEqual(web_config['max_resolution'], (720, 1280))
        self.assertEqual(web_config['max_fps'], 24)
        self.assertEqual(web_config['format'], 'mp4')
        self.assertEqual(web_config['quality'], 'high')

        # Test mobile preset
        mobile_processor = LTXVideoProcessor(preset='mobile')
        mobile_config = mobile_processor.config

        self.assertEqual(mobile_config['max_resolution'], (480, 854))
        self.assertEqual(mobile_config['max_fps'], 15)
        self.assertLessEqual(mobile_config['max_fps'], web_config['max_fps'])

    def test_process_ltx_output_with_frames(self):
        """Test processing LTX output from frame list."""
        video_base64, metadata = self.processor.process_ltx_output(
            self.test_frames,
            fps=8,
            enhance_quality=True,
            add_metadata=True
        )

        # Verify output format
        self.assertIsInstance(video_base64, str)
        self.assertIsInstance(metadata, dict)

        # Verify base64 encoding
        try:
            decoded = base64.b64decode(video_base64)
            self.assertGreater(len(decoded), 0)
        except Exception:
            self.fail("Invalid base64 encoding")

        # Verify metadata structure
        required_fields = ['width', 'height', 'fps', 'num_frames', 'duration']
        for field in required_fields:
            self.assertIn(field, metadata)

        # Verify LTX-specific metadata
        self.assertIn('ltx_video', metadata)
        ltx_meta = metadata['ltx_video']
        self.assertEqual(ltx_meta['model_type'], 'LTX-Video')
        self.assertEqual(ltx_meta['generation_method'], 'text-to-video')
        self.assertEqual(ltx_meta['preset'], 'web')

    def test_process_ltx_output_with_base64(self):
        """Test processing LTX output from base64 string."""
        video_base64, metadata = self.processor.process_ltx_output(
            self.test_video_base64,
            fps=8,
            enhance_quality=False,
            add_metadata=True
        )

        # Verify processing succeeded
        self.assertIsInstance(video_base64, str)
        self.assertIsInstance(metadata, dict)

        # Verify metadata includes processing info
        self.assertIn('ltx_video', metadata)
        self.assertIn('processing_time_ms', metadata['ltx_video'])

    def test_process_ltx_output_with_numpy_tensor(self):
        """Test processing LTX output from numpy tensor."""
        # Create 4D tensor (frames, height, width, channels)
        tensor = np.stack(self.test_frames[:10])

        video_base64, metadata = self.processor.process_ltx_output(
            tensor,
            fps=8,
            enhance_quality=True
        )

        # Verify processing succeeded
        self.assertIsInstance(video_base64, str)
        self.assertEqual(metadata['num_frames'], 10)

    def test_quality_enhancement(self):
        """Test quality enhancement processing."""
        # Process without enhancement
        video_base64_normal, metadata_normal = self.processor.process_ltx_output(
            self.test_frames[:5],
            enhance_quality=False
        )

        # Process with enhancement
        video_base64_enhanced, metadata_enhanced = self.processor.process_ltx_output(
            self.test_frames[:5],
            enhance_quality=True
        )

        # Enhanced version should have quality metadata
        self.assertTrue(metadata_enhanced['ltx_video']['quality_enhanced'])

        # Both should be valid base64
        self.assertIsInstance(video_base64_normal, str)
        self.assertIsInstance(video_base64_enhanced, str)

    def test_preset_optimization(self):
        """Test optimization for different presets."""
        # Create large frame sequence
        large_frames = self._create_test_frames(count=100, width=1920, height=1080)

        # Test mobile preset (should resize and limit frames)
        mobile_processor = LTXVideoProcessor(preset='mobile')
        video_base64, metadata = mobile_processor.process_ltx_output(
            large_frames,
            fps=30
        )

        # Verify mobile constraints applied
        self.assertLessEqual(metadata['width'], 480)
        self.assertLessEqual(metadata['height'], 854)
        self.assertLessEqual(metadata['fps'], 15)

        # Test social preset (higher limits)
        social_processor = LTXVideoProcessor(preset='social')
        video_base64_social, metadata_social = social_processor.process_ltx_output(
            large_frames[:50],
            fps=30
        )

        # Social should allow higher resolution
        self.assertGreaterEqual(metadata_social['width'], metadata['width'])

    def test_metadata_generation(self):
        """Test comprehensive metadata generation."""
        video_base64, metadata = self.processor.process_ltx_output(
            self.test_frames,
            fps=8,
            add_metadata=True
        )

        # Check quality metrics
        self.assertIn('quality_metrics', metadata)
        quality = metadata['quality_metrics']

        self.assertIn('average_brightness', quality)
        self.assertIn('average_motion', quality)
        self.assertIn('resolution_category', quality)
        self.assertIn('frame_consistency', quality)

        # Check compression info
        self.assertIn('compression', metadata)
        compression = metadata['compression']

        self.assertIn('preset', compression)
        self.assertIn('format', compression)
        self.assertIn('estimated_bitrate', compression)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test empty frames
        with self.assertRaises(ValueError):
            self.processor.process_ltx_output([])

        # Test invalid video data type
        with self.assertRaises(ValueError):
            self.processor.process_ltx_output(123)

        # Test invalid base64
        with self.assertRaises(ValueError):
            self.processor.process_ltx_output("invalid_base64_data")

    def test_frame_validation(self):
        """Test frame validation functionality."""
        # Test inconsistent frame shapes
        inconsistent_frames = self.test_frames[:3].copy()
        inconsistent_frames.append(np.zeros((100, 100, 3), dtype=np.uint8))

        with self.assertRaises(ValueError):
            self.processor.process_ltx_output(inconsistent_frames)

    def test_factory_function(self):
        """Test factory function for creating processors."""
        processor = create_ltx_video_processor('mobile')
        self.assertIsInstance(processor, LTXVideoProcessor)
        self.assertEqual(processor.preset, 'mobile')

        # Test default preset
        default_processor = create_ltx_video_processor()
        self.assertEqual(default_processor.preset, 'web')


class TestLTXVideoValidator(unittest.TestCase):
    """Test cases for LTX-Video validation."""

    def test_validate_generation_params_valid(self):
        """Test validation of valid generation parameters."""
        result = LTXVideoValidator.validate_generation_params(
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)

    def test_validate_generation_params_invalid_resolution(self):
        """Test validation of invalid resolution parameters."""
        # Too small
        result = LTXVideoValidator.validate_generation_params(
            width=100,
            height=100,
            num_frames=25,
            fps=8
        )

        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

        # Too large
        result = LTXVideoValidator.validate_generation_params(
            width=2000,
            height=2000,
            num_frames=25,
            fps=8
        )

        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_validate_generation_params_too_many_frames(self):
        """Test validation of excessive frame count."""
        result = LTXVideoValidator.validate_generation_params(
            width=704,
            height=704,
            num_frames=300,  # Exceeds MAX_FRAMES (257)
            fps=8
        )

        self.assertFalse(result['valid'])
        self.assertTrue(any('frame count' in error.lower() for error in result['errors']))

    def test_validate_generation_params_warnings(self):
        """Test generation of warnings for suboptimal parameters."""
        # Non-divisible by 32 resolution
        result = LTXVideoValidator.validate_generation_params(
            width=700,  # Not divisible by 32
            height=700,  # Not divisible by 32
            num_frames=25,
            fps=8
        )

        self.assertTrue(result['valid'])  # Still valid, but with warnings
        self.assertGreater(len(result['warnings']), 0)
        self.assertGreater(len(result['suggestions']), 0)

    def test_validate_output_video_complete(self):
        """Test validation of complete video output."""
        video_metadata = {
            'width': 704,
            'height': 704,
            'fps': 8,
            'num_frames': 25,
            'duration': 3.125,
            'size_bytes': 1024 * 1024,  # 1MB
            'quality_metrics': {
                'average_brightness': 128.0,
                'average_motion': 15.5,
                'resolution_category': 'HD',
                'frame_consistency': 0.85
            }
        }

        result = LTXVideoValidator.validate_output_video(video_metadata)

        self.assertTrue(result['valid'])
        self.assertGreater(result['quality_score'], 50)  # Should have decent score

    def test_validate_output_video_missing_metadata(self):
        """Test validation with missing metadata."""
        incomplete_metadata = {
            'width': 704,
            'height': 704
            # Missing fps, num_frames, duration
        }

        result = LTXVideoValidator.validate_output_video(incomplete_metadata)

        self.assertFalse(result['valid'])
        self.assertGreater(len(result['issues']), 0)

    def test_aspect_ratio_detection(self):
        """Test aspect ratio detection and suggestions."""
        # Test 16:9 aspect ratio
        result = LTXVideoValidator.validate_generation_params(
            width=1280,
            height=720,
            num_frames=25,
            fps=8
        )

        self.assertTrue(any('16:9' in suggestion for suggestion in result['suggestions']))

        # Test square aspect ratio
        result = LTXVideoValidator.validate_generation_params(
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        self.assertTrue(any('1:1' in suggestion for suggestion in result['suggestions']))

    def test_quality_score_calculation(self):
        """Test quality score calculation components."""
        # High quality metadata
        high_quality_metadata = {
            'width': 1280,
            'height': 720,
            'fps': 24,
            'num_frames': 48,
            'duration': 2.0,
            'size_bytes': 2 * 1024 * 1024,  # 2MB
            'quality_metrics': {
                'resolution_category': 'HD+',
                'frame_consistency': 0.9
            }
        }

        result = LTXVideoValidator.validate_output_video(high_quality_metadata)
        high_score = result['quality_score']

        # Low quality metadata
        low_quality_metadata = {
            'width': 320,
            'height': 240,
            'fps': 8,
            'num_frames': 16,
            'duration': 2.0,
            'size_bytes': 10 * 1024 * 1024,  # 10MB (inefficient)
            'quality_metrics': {
                'resolution_category': 'Low',
                'frame_consistency': 0.3
            }
        }

        result_low = LTXVideoValidator.validate_output_video(low_quality_metadata)
        low_score = result_low['quality_score']

        # High quality should score better than low quality
        self.assertGreater(high_score, low_score)


class TestLTXVideoUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions and integration."""

    def test_validate_ltx_generation_request_valid(self):
        """Test validation of valid generation request."""
        result = validate_ltx_generation_request(
            prompt="A majestic eagle soaring through mountain peaks",
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)

    def test_validate_ltx_generation_request_invalid_prompt(self):
        """Test validation with invalid prompts."""
        # Empty prompt
        result = validate_ltx_generation_request(
            prompt="",
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        self.assertFalse(result['valid'])
        self.assertTrue(any('prompt' in error.lower() for error in result['errors']))

        # Very short prompt
        result = validate_ltx_generation_request(
            prompt="hi",
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        self.assertFalse(result['valid'])

    def test_validate_ltx_generation_request_long_prompt(self):
        """Test validation with very long prompt."""
        long_prompt = "A " * 500  # 1000 characters

        result = validate_ltx_generation_request(
            prompt=long_prompt,
            width=704,
            height=704,
            num_frames=25,
            fps=8
        )

        # Should be valid but with warnings
        self.assertTrue(result['valid'])
        self.assertGreater(len(result['warnings']), 0)

    def test_integration_processor_and_validator(self):
        """Test integration between processor and validator."""
        # Create test frames
        frames = []
        for i in range(25):
            frame = np.random.randint(0, 256, (704, 704, 3), dtype=np.uint8)
            frames.append(frame)

        # Process with LTX processor
        processor = LTXVideoProcessor(preset='web')
        video_base64, metadata = processor.process_ltx_output(frames, fps=8)

        # Validate the output
        validation_result = LTXVideoValidator.validate_output_video(metadata)

        # Should produce valid output
        self.assertTrue(validation_result['valid'])
        self.assertGreater(validation_result['quality_score'], 0)

    def test_preset_optimization_integration(self):
        """Test integration of different presets with validation."""
        # Create high-resolution frames
        high_res_frames = []
        for i in range(50):
            frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
            high_res_frames.append(frame)

        # Test with different presets
        presets = ['mobile', 'web', 'social', 'preview']

        for preset in presets:
            processor = LTXVideoProcessor(preset=preset)
            video_base64, metadata = processor.process_ltx_output(
                high_res_frames,
                fps=30
            )

            # Validate each output
            validation_result = LTXVideoValidator.validate_output_video(metadata)

            # All presets should produce valid output
            self.assertTrue(validation_result['valid'],
                           f"Preset {preset} produced invalid output")

            # Check preset-specific constraints
            config = processor.LTX_PRESETS[preset]
            self.assertLessEqual(metadata['width'], config['max_resolution'][0])
            self.assertLessEqual(metadata['height'], config['max_resolution'][1])

    @patch('src.utils.video_utils.imageio')
    def test_error_handling_integration(self, mock_imageio):
        """Test error handling across utility functions."""
        # Mock imageio to raise exception during video encoding
        mock_imageio.get_writer.side_effect = Exception("Encoding failed")

        processor = LTXVideoProcessor(preset='web')
        frames = [np.zeros((704, 704, 3), dtype=np.uint8)]

        # Should propagate encoding errors
        with self.assertRaises(Exception):
            processor.process_ltx_output(frames)


class TestLTXVideoPerformanceIntegration(unittest.TestCase):
    """Test performance characteristics of utility functions."""

    def test_processing_time_measurement(self):
        """Test that processing time is measured and reported."""
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 256, (704, 704, 3), dtype=np.uint8)
            frames.append(frame)

        processor = LTXVideoProcessor(preset='web')
        video_base64, metadata = processor.process_ltx_output(frames, fps=8)

        # Should include processing time
        self.assertIn('ltx_video', metadata)
        self.assertIn('processing_time_ms', metadata['ltx_video'])
        self.assertGreater(metadata['ltx_video']['processing_time_ms'], 0)

    def test_memory_efficiency_large_frames(self):
        """Test memory efficiency with large frame sequences."""
        import psutil
        import gc

        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create large frame sequence
        large_frames = []
        for i in range(50):
            frame = np.random.randint(0, 256, (1280, 720, 3), dtype=np.uint8)
            large_frames.append(frame)

        processor = LTXVideoProcessor(preset='web')
        video_base64, metadata = processor.process_ltx_output(large_frames, fps=8)

        # Clean up
        del large_frames, video_base64, metadata
        gc.collect()

        # Measure final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory

        # Memory growth should be reasonable (less than 100MB)
        self.assertLess(memory_growth, 100, f"Memory growth too high: {memory_growth:.2f}MB")

    def test_validation_performance(self):
        """Test validation performance with various inputs."""
        import time

        # Test parameter validation performance
        start_time = time.time()

        for _ in range(100):
            LTXVideoValidator.validate_generation_params(704, 704, 25, 8)

        param_validation_time = time.time() - start_time

        # Should be very fast (less than 0.1 seconds for 100 validations)
        self.assertLess(param_validation_time, 0.1)

        # Test output validation performance
        test_metadata = {
            'width': 704, 'height': 704, 'fps': 8, 'num_frames': 25,
            'duration': 3.125, 'size_bytes': 1024 * 1024,
            'quality_metrics': {
                'average_brightness': 128.0, 'average_motion': 15.5,
                'resolution_category': 'HD', 'frame_consistency': 0.85
            }
        }

        start_time = time.time()

        for _ in range(100):
            LTXVideoValidator.validate_output_video(test_metadata)

        output_validation_time = time.time() - start_time

        # Should also be very fast
        self.assertLess(output_validation_time, 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)