"""
Standalone tests for MMI-004 Request Validator functionality.

These tests validate the request validation and modality detection
without requiring the full application stack.
"""

import pytest
from typing import Dict, Any


# Inline RequestValidator implementation for testing
class ModalityDetector:
    """Detects request modality from parameters."""

    @staticmethod
    def detect_modality(request_data: Dict[str, Any]) -> str:
        """Detect modality from request parameters."""
        if not isinstance(request_data, dict):
            return None

        # Explicit modality specification
        if "modality" in request_data:
            return request_data["modality"]

        # Text-to-image detection
        if "prompt" in request_data and isinstance(request_data["prompt"], str):
            if request_data["prompt"].strip():
                return "text-to-image"

        # Image-to-video detection
        if "image_url" in request_data and isinstance(request_data["image_url"], str):
            if request_data["image_url"].strip():
                return "image-to-video"

        return None


class RequestValidator:
    """Validates request parameters for multi-modal inference."""

    def __init__(self):
        self.modality_detector = ModalityDetector()

    def validate_full_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete request and return error details if invalid."""
        if not isinstance(request_data, dict):
            return {
                "field": "request",
                "value": type(request_data).__name__,
                "message": "Request must be a dictionary"
            }

        # Detect modality
        modality = self.modality_detector.detect_modality(request_data)
        if not modality:
            return {
                "field": "modality",
                "value": None,
                "message": "Could not determine request modality"
            }

        # Validate based on modality
        if modality == "text-to-image":
            return self._validate_text_to_image(request_data)
        elif modality == "image-to-video":
            return self._validate_image_to_video(request_data)

        return None  # Valid

    def _validate_text_to_image(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text-to-image parameters."""
        # Required: prompt
        if not request_data.get("prompt"):
            return {
                "field": "prompt",
                "value": request_data.get("prompt"),
                "message": "Prompt is required for text-to-image generation"
            }

        # Optional: steps (1-50)
        steps = request_data.get("steps", 4)
        if not isinstance(steps, int) or steps < 1 or steps > 50:
            return {
                "field": "steps",
                "value": steps,
                "message": "Steps must be an integer between 1 and 50"
            }

        # Optional: guidance_scale (0.1-20.0)
        guidance_scale = request_data.get("guidance_scale", 1.0)
        if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
            return {
                "field": "guidance_scale",
                "value": guidance_scale,
                "message": "Guidance scale must be between 0.1 and 20.0"
            }

        return None  # Valid

    def _validate_image_to_video(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate image-to-video parameters."""
        # Required: image_url
        image_url = request_data.get("image_url")
        if not isinstance(image_url, str) or not image_url.strip():
            return {
                "field": "image_url",
                "value": image_url,
                "message": "Image URL is required for image-to-video generation"
            }

        # Optional: duration (1-30 seconds)
        duration = request_data.get("duration", 4)
        if not isinstance(duration, (int, float)) or duration < 1 or duration > 30:
            return {
                "field": "duration",
                "value": duration,
                "message": "Duration must be between 1 and 30 seconds"
            }

        return None  # Valid


class TestModalityDetector:
    """Test modality detection functionality."""

    def test_explicit_modality_detection(self):
        """Test detection with explicit modality specification."""
        detector = ModalityDetector()

        request = {"modality": "text-to-image", "prompt": "test"}
        assert detector.detect_modality(request) == "text-to-image"

        request = {"modality": "image-to-video", "image_url": "test.jpg"}
        assert detector.detect_modality(request) == "image-to-video"

    def test_text_to_image_detection(self):
        """Test automatic text-to-image detection."""
        detector = ModalityDetector()

        request = {"prompt": "A beautiful sunset", "steps": 4}
        assert detector.detect_modality(request) == "text-to-image"

    def test_image_to_video_detection(self):
        """Test automatic image-to-video detection."""
        detector = ModalityDetector()

        request = {"image_url": "https://example.com/image.jpg", "duration": 4}
        assert detector.detect_modality(request) == "image-to-video"

    def test_empty_prompt_no_detection(self):
        """Test that empty prompt doesn't trigger text-to-image."""
        detector = ModalityDetector()

        request = {"prompt": "", "steps": 4}
        assert detector.detect_modality(request) is None

        request = {"prompt": "   ", "steps": 4}
        assert detector.detect_modality(request) is None

    def test_no_modality_detection(self):
        """Test requests that can't be categorized."""
        detector = ModalityDetector()

        assert detector.detect_modality({}) is None
        assert detector.detect_modality({"unknown": "param"}) is None
        assert detector.detect_modality(None) is None


class TestRequestValidator:
    """Test request validation functionality."""

    def test_valid_text_to_image_request(self):
        """Test validation of valid text-to-image requests."""
        validator = RequestValidator()

        request = {
            "prompt": "A beautiful landscape",
            "steps": 4,
            "guidance_scale": 1.0
        }

        error = validator.validate_full_request(request)
        assert error is None

    def test_valid_image_to_video_request(self):
        """Test validation of valid image-to-video requests."""
        validator = RequestValidator()

        request = {
            "image_url": "https://example.com/image.jpg",
            "duration": 4
        }

        error = validator.validate_full_request(request)
        assert error is None

    def test_missing_prompt_error(self):
        """Test validation error for missing prompt."""
        validator = RequestValidator()

        request = {"steps": 4, "guidance_scale": 1.0}

        error = validator.validate_full_request(request)
        assert error is not None
        # Without a prompt, modality can't be detected
        assert error["field"] == "modality"
        assert "could not determine" in error["message"].lower()

    def test_invalid_steps_error(self):
        """Test validation error for invalid steps."""
        validator = RequestValidator()

        request = {"prompt": "test", "steps": 100}  # Out of range

        error = validator.validate_full_request(request)
        assert error is not None
        assert error["field"] == "steps"
        assert "between 1 and 50" in error["message"]

    def test_invalid_guidance_scale_error(self):
        """Test validation error for invalid guidance scale."""
        validator = RequestValidator()

        request = {"prompt": "test", "guidance_scale": 25.0}  # Out of range

        error = validator.validate_full_request(request)
        assert error is not None
        assert error["field"] == "guidance_scale"
        assert "between 0.1 and 20.0" in error["message"]

    def test_missing_image_url_error(self):
        """Test validation error for missing image URL."""
        validator = RequestValidator()

        request = {"duration": 4}

        error = validator.validate_full_request(request)
        assert error is not None
        # Without an image_url, modality can't be detected
        assert error["field"] == "modality"
        assert "could not determine" in error["message"].lower()

    def test_invalid_duration_error(self):
        """Test validation error for invalid duration."""
        validator = RequestValidator()

        request = {"image_url": "test.jpg", "duration": 50}  # Out of range

        error = validator.validate_full_request(request)
        assert error is not None
        assert error["field"] == "duration"
        assert "between 1 and 30" in error["message"]

    def test_non_dict_request_error(self):
        """Test validation error for non-dictionary request."""
        validator = RequestValidator()

        error = validator.validate_full_request("not a dict")
        assert error is not None
        assert error["field"] == "request"
        assert "dictionary" in error["message"]

    def test_undetectable_modality_error(self):
        """Test validation error when modality can't be determined."""
        validator = RequestValidator()

        request = {"unknown_param": "value"}

        error = validator.validate_full_request(request)
        assert error is not None
        assert error["field"] == "modality"
        assert "could not determine" in error["message"].lower()


class TestRequestValidatorEdgeCases:
    """Test edge cases for request validation."""

    def test_empty_request(self):
        """Test validation of empty request."""
        validator = RequestValidator()

        error = validator.validate_full_request({})
        assert error is not None
        assert "modality" in error["field"]

    def test_prompt_whitespace_handling(self):
        """Test handling of whitespace in prompts."""
        validator = RequestValidator()

        # Whitespace-only prompt should fail
        request = {"prompt": "   ", "steps": 4}
        error = validator.validate_full_request(request)
        assert error is not None

    def test_boundary_values(self):
        """Test boundary values for parameters."""
        validator = RequestValidator()

        # Valid boundary values
        valid_requests = [
            {"prompt": "test", "steps": 1},      # Min steps
            {"prompt": "test", "steps": 50},     # Max steps
            {"prompt": "test", "guidance_scale": 0.1},  # Min guidance
            {"prompt": "test", "guidance_scale": 20.0}, # Max guidance
            {"image_url": "test.jpg", "duration": 1},   # Min duration
            {"image_url": "test.jpg", "duration": 30},  # Max duration
        ]

        for request in valid_requests:
            error = validator.validate_full_request(request)
            assert error is None, f"Request {request} should be valid"

        # Invalid boundary values
        invalid_requests = [
            {"prompt": "test", "steps": 0},       # Below min
            {"prompt": "test", "steps": 51},      # Above max
            {"prompt": "test", "guidance_scale": 0.05},  # Below min
            {"prompt": "test", "guidance_scale": 21.0},  # Above max
            {"image_url": "test.jpg", "duration": 0.5},  # Below min
            {"image_url": "test.jpg", "duration": 31},   # Above max
        ]

        for request in invalid_requests:
            error = validator.validate_full_request(request)
            assert error is not None, f"Request {request} should be invalid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])