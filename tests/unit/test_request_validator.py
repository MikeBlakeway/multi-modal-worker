"""
Unit tests for RequestValidator and ModalityDetector classes.

Tests cover parameter validation, modality detection, error handling,
and edge cases for the request routing system.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

# Import the classes to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.request_validator import RequestValidator, ModalityDetector
from utils.exceptions import ValidationError


class TestModalityDetector:
    """Test cases for automatic modality detection based on parameter signatures."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = ModalityDetector()

    def test_detect_text_to_image_modality(self):
        """Test detection of text-to-image requests."""
        # Valid text-to-image parameters
        parameters = {
            'prompt': 'A beautiful sunset',
            'steps': 4,
            'guidance_scale': 1.0
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'text-to-image'

    def test_detect_image_to_video_modality(self):
        """Test detection of image-to-video requests."""
        parameters = {
            'image': 'base64_image_data',
            'motion_bucket_id': 127,
            'fps': 6,
            'num_frames': 25
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'image-to-video'

    def test_detect_text_to_video_modality(self):
        """Test detection of text-to-video requests."""
        parameters = {
            'prompt': 'A cat playing with a ball',
            'num_frames': 49,
            'fps': 8,
            'duration': 6.0
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'text-to-video'

    def test_detect_controlnet_modality(self):
        """Test detection of ControlNet-guided requests."""
        parameters = {
            'prompt': 'A house',
            'control_image': 'base64_control_image',
            'control_type': 'canny',
            'control_strength': 1.0
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'controlnet'

    def test_detect_inpainting_modality(self):
        """Test detection of inpainting requests."""
        parameters = {
            'prompt': 'Fill the masked area',
            'image': 'base64_image_data',
            'mask': 'base64_mask_data',
            'strength': 0.8
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'inpainting'

    def test_detect_camera_control_modality(self):
        """Test detection of camera control requests."""
        parameters = {
            'prompt': 'A rotating view of an object',
            'camera_motion': 'orbit_left',
            'motion_strength': 0.7
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'camera-control'

    def test_detect_no_modality_match(self):
        """Test handling of parameters that don't match any modality."""
        # Parameters that don't match any signature
        parameters = {
            'unknown_param': 'value',
            'another_param': 42
        }

        result = self.detector.detect_modality(parameters)
        assert result is None

    def test_detect_ambiguous_parameters(self):
        """Test handling of parameters that could match multiple modalities."""
        # Parameters that contain prompt (common to many modalities)
        parameters = {
            'prompt': 'Test prompt'
        }

        # Should return the first matching modality (text-to-image in this case)
        result = self.detector.detect_modality(parameters)
        assert result == 'text-to-image'

    def test_detect_empty_parameters(self):
        """Test handling of empty parameter dictionary."""
        parameters = {}

        result = self.detector.detect_modality(parameters)
        assert result is None

    def test_detect_with_extra_parameters(self):
        """Test detection works with extra parameters beyond signature."""
        # Text-to-image with additional parameters
        parameters = {
            'prompt': 'A beautiful sunset',
            'steps': 4,
            'guidance_scale': 1.0,
            'extra_param': 'should_not_interfere',
            'another_extra': 123
        }

        result = self.detector.detect_modality(parameters)
        assert result == 'text-to-image'


class TestRequestValidator:
    """Test cases for comprehensive request parameter validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = RequestValidator()

    def test_validate_text_to_image_success(self):
        """Test successful validation of text-to-image request."""
        request_data = {
            'prompt': 'A beautiful sunset over mountains',
            'steps': 4,
            'guidance_scale': 1.0,
            'width': 1024,
            'height': 1024
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_missing_required_parameter(self):
        """Test validation failure for missing required parameter."""
        request_data = {
            'steps': 4,
            'guidance_scale': 1.0
            # Missing 'prompt'
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'prompt'
        assert 'required' in error['message'].lower()

    def test_validate_invalid_type_with_conversion(self):
        """Test type validation with successful conversion."""
        request_data = {
            'prompt': 'A beautiful sunset',
            'steps': '4',  # String that can be converted to int
            'guidance_scale': '1.0'  # String that can be converted to float
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_invalid_type_no_conversion(self):
        """Test type validation failure when conversion not possible."""
        request_data = {
            'prompt': 'A beautiful sunset',
            'steps': 'not_a_number',  # Cannot convert to int
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'steps'
        assert 'type' in error['message'].lower()

    def test_validate_range_validation_success(self):
        """Test successful range validation."""
        request_data = {
            'prompt': 'A beautiful sunset',
            'steps': 4,  # Within range [1, 50]
            'guidance_scale': 1.0,  # Within range [0.0, 20.0]
            'width': 1024,  # Valid dimension
            'height': 768   # Valid dimension
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_range_validation_failure(self):
        """Test range validation failure."""
        request_data = {
            'prompt': 'A beautiful sunset',
            'steps': 100,  # Outside range [1, 50]
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'steps'
        assert 'range' in error['message'].lower()

    def test_validate_string_length_success(self):
        """Test successful string length validation."""
        request_data = {
            'prompt': 'A valid prompt',  # Within length limits
            'steps': 4,
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_string_length_failure(self):
        """Test string length validation failure."""
        request_data = {
            'prompt': 'x' * 2001,  # Exceeds max length of 2000
            'steps': 4,
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'prompt'
        assert 'length' in error['message'].lower()

    def test_validate_empty_string_failure(self):
        """Test validation failure for empty required string."""
        request_data = {
            'prompt': '',  # Empty string
            'steps': 4,
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'prompt'
        assert 'empty' in error['message'].lower()

    def test_validate_image_to_video_success(self):
        """Test successful validation of image-to-video request."""
        request_data = {
            'image': 'base64_encoded_image_data',
            'motion_bucket_id': 127,
            'fps': 6,
            'num_frames': 25
        }

        error = self.validator.validate_full_request(request_data, 'image-to-video')
        assert error is None

    def test_validate_unsupported_modality(self):
        """Test validation of unsupported modality."""
        request_data = {
            'prompt': 'Test prompt'
        }

        error = self.validator.validate_full_request(request_data, 'unsupported-modality')
        assert error is not None
        assert 'unsupported' in error['message'].lower()

    def test_validate_complex_nested_validation(self):
        """Test validation with multiple parameter types and constraints."""
        request_data = {
            'prompt': 'A complex scene with multiple objects',
            'steps': 20,
            'guidance_scale': 7.5,
            'width': 512,
            'height': 768,
            'seed': 42,
            'batch_size': 1
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_boundary_conditions(self):
        """Test validation at boundary conditions."""
        # Test minimum values
        request_data = {
            'prompt': 'x',  # Minimum length
            'steps': 1,     # Minimum steps
            'guidance_scale': 0.1,  # Near minimum guidance
            'width': 128,   # Minimum width
            'height': 128   # Minimum height
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

        # Test maximum values
        request_data = {
            'prompt': 'x' * 2000,  # Maximum length
            'steps': 50,           # Maximum steps
            'guidance_scale': 20.0, # Maximum guidance
            'width': 2048,         # Maximum width
            'height': 2048         # Maximum height
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_none_values(self):
        """Test handling of None values."""
        request_data = {
            'prompt': 'A beautiful sunset',
            'steps': None,  # None value
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'steps'

    def test_validate_modality_specific_parameters(self):
        """Test validation of modality-specific parameter requirements."""
        # ControlNet requires specific parameters
        request_data = {
            'prompt': 'A house',
            'control_image': 'base64_control_image_data',
            'control_type': 'canny',
            'control_strength': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'controlnet')
        assert error is None

        # Missing controlnet-specific parameter
        request_data = {
            'prompt': 'A house',
            'control_image': 'base64_control_image_data'
            # Missing control_type
        }

        error = self.validator.validate_full_request(request_data, 'controlnet')
        assert error is not None
        assert error['field'] == 'control_type'


class TestRequestValidatorEdgeCases:
    """Test edge cases and error conditions for RequestValidator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = RequestValidator()

    def test_validate_with_exception_handling(self):
        """Test proper exception handling during validation."""
        # Simulate validation that might cause internal error
        with patch.object(RequestValidator, 'validate_parameters') as mock_validate:
            mock_validate.side_effect = Exception("Simulated validation error")

            request_data = {'prompt': 'Test'}
            error = self.validator.validate_full_request(request_data, 'text-to-image')

            # Should handle exception gracefully
            assert error is not None

    def test_validate_unicode_handling(self):
        """Test validation with Unicode characters."""
        request_data = {
            'prompt': '🎨 A beautiful sunset with émojis and àccénts 中文',
            'steps': 4,
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None

    def test_validate_very_large_numbers(self):
        """Test validation with very large numbers."""
        request_data = {
            'prompt': 'Test',
            'steps': 999999999,  # Very large number
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'steps'

    def test_validate_negative_numbers(self):
        """Test validation with negative numbers."""
        request_data = {
            'prompt': 'Test',
            'steps': -5,  # Negative number
            'guidance_scale': 1.0
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is not None
        assert error['field'] == 'steps'

    def test_validate_float_precision(self):
        """Test validation with high-precision float values."""
        request_data = {
            'prompt': 'Test',
            'steps': 4,
            'guidance_scale': 1.23456789123456789  # High precision float
        }

        error = self.validator.validate_full_request(request_data, 'text-to-image')
        assert error is None


if __name__ == '__main__':
    pytest.main([__file__])