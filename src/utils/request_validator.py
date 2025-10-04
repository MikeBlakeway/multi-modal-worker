"""
Request Validation and Modality Detection

Provides utilities for validating incoming requests and determining
the appropriate modality type based on request parameters.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    # Try relative imports first (when running as package)
    from . import ValidationError
except ImportError:
    # Fall back to absolute imports (when running as script or in tests)
    from src.utils import ValidationError

logger = logging.getLogger(__name__)


class ModalityDetector:
    """
    Detects the intended modality based on request parameters.

    Uses a combination of explicit modality specification and
    parameter pattern matching to determine the target modality.
    """

    # Modality signatures - parameters that strongly indicate a specific modality
    MODALITY_SIGNATURES = {
        'text-to-image': {
            'required_any': ['prompt', 'text'],
            'forbidden': ['image', 'video', 'init_image'],
            'indicators': ['steps', 'guidance_scale', 'width', 'height', 'seed']
        },
                'image-to-video': {
            'required_any': ['image', 'init_image'],
            'forbidden': ['mask'],
            'indicators': ['video', 'fps', 'frames', 'motion_strength', 'noise_level', 'init_image']
        },
        'text-to-video': {
            'required_any': ['prompt', 'text'],
            'forbidden': ['image', 'init_image', 'camera_motion', 'camera_pose', 'camera_trajectory'],
            'indicators': ['fps', 'duration', 'frames', 'video']
        },
        'controlnet': {
            'required_any': ['control_image'],
            'required_all': ['control_type'],
            'forbidden': ['video'],
            'indicators': ['control_strength', 'control_guidance_start', 'control_guidance_end', 'canny_low_threshold', 'canny_high_threshold']
        },
        'inpainting': {
            'required_any': ['mask'],
            'forbidden': ['video'],
            'indicators': ['inpaint', 'mask_image', 'init_image', 'image', 'strength']
        },
        'camera-control': {
            'required_any': ['camera_pose', 'camera_trajectory', 'camera_motion'],
            'forbidden': [],
            'indicators': ['pose', 'trajectory', 'camera', 'motion_strength', 'orbit', 'pan', 'tilt', 'zoom']
        }
    }

    @classmethod
    def detect_modality(cls, request_data: Dict[str, Any]) -> str:
        """
        Detect the intended modality from request parameters.

        Args:
            request_data: Raw request data

        Returns:
            Detected modality type

        Raises:
            ValidationError: If no modality can be determined
        """
        # Check for explicit modality specification
        explicit_modality = request_data.get('modality')
        if explicit_modality:
            if explicit_modality in cls.MODALITY_SIGNATURES:
                logger.debug(f"Explicit modality specified: {explicit_modality}")
                return explicit_modality
            else:
                raise ValidationError(
                    'modality',
                    explicit_modality,
                    f"Unsupported modality. Supported: {list(cls.MODALITY_SIGNATURES.keys())}"
                )

        # Attempt automatic detection based on parameter patterns
        scores = {}
        for modality, signature in cls.MODALITY_SIGNATURES.items():
            score = cls._calculate_modality_score(request_data, signature)
            if score > 0:
                scores[modality] = score

        if not scores:
            # Return None when no modality can be detected
            return None

        # Return the modality with highest confidence score
        detected_modality = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Auto-detected modality: {detected_modality} (confidence scores: {scores})")
        return detected_modality

    @classmethod
    def _calculate_modality_score(cls, request_data: Dict[str, Any], signature: Dict[str, List[str]]) -> int:
        """Calculate confidence score for a specific modality signature."""
        score = 0

        # Check if any required parameters are present
        required_any = signature.get('required_any', [])
        if required_any:
            has_required = any(param in request_data for param in required_any)
            if not has_required:
                return 0  # Cannot be this modality
            score += 3  # Strong positive signal

        # Check if all required_all parameters are present
        required_all = signature.get('required_all', [])
        if required_all:
            has_all_required = all(param in request_data for param in required_all)
            if not has_all_required:
                return 0  # Cannot be this modality
            score += 5  # Very strong positive signal

        # Check for forbidden parameters
        forbidden = signature.get('forbidden', [])
        for param in forbidden:
            if param in request_data:
                return 0  # Cannot be this modality

        # Count indicator parameters
        indicators = signature.get('indicators', [])
        for indicator in indicators:
            if indicator in request_data:
                score += 1

        return score


class RequestValidator:
    """
    Validates request parameters for completeness and correctness.

    MMI-004 compliant implementation with instance-based design
    and explicit modality specification.
    """

    # Common validation rules
    COMMON_VALIDATIONS = {
        'prompt': {
            'type': str,
            'min_length': 1,
            'max_length': 2000,
            'required': False
        },
        'seed': {
            'type': int,
            'min_value': 0,
            'max_value': 2**32 - 1,
            'required': False
        },
        'steps': {
            'type': int,
            'min_value': 1,
            'max_value': 50,
            'required': False
        },
        'guidance_scale': {
            'type': (int, float),
            'min_value': 0.1,
            'max_value': 20.0,
            'required': False
        },
        'width': {
            'type': int,
            'min_value': 64,
            'max_value': 2048,
            'multiple_of': 8,
            'required': False
        },
        'height': {
            'type': int,
            'min_value': 64,
            'max_value': 2048,
            'multiple_of': 8,
            'required': False
        }
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RequestValidator instance.

        Args:
            config: Optional configuration dictionary for customization
        """
        self.config = config or {}

    @classmethod
    def validate_request_format(cls, request_data: Any) -> Dict[str, Any]:
        """
        Validate basic request format and structure.

        Args:
            request_data: Raw request data

        Returns:
            Validated request data dictionary

        Raises:
            ValidationError: If request format is invalid
        """
        if not isinstance(request_data, dict):
            raise ValidationError(
                'request',
                type(request_data).__name__,
                "Request must be a JSON object/dictionary"
            )

        if not request_data:
            raise ValidationError(
                'request',
                {},
                "Request cannot be empty"
            )

        return request_data

    @classmethod
    def validate_parameters(cls, request_data: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """
        Validate individual parameters according to their rules and modality requirements.

        Args:
            request_data: Request data to validate
            modality: Target modality for validation

        Returns:
            Validated and normalized request data

        Raises:
            ValidationError: If any parameter is invalid or required parameters are missing
        """
        # Step 1: Check modality-specific required parameters
        cls._validate_required_parameters(request_data, modality)

        # Step 2: Validate and normalize individual parameters
        validated_data = {}
        for param_name, value in request_data.items():
            # Skip validation for unknown parameters (pass-through)
            if param_name not in cls.COMMON_VALIDATIONS:
                validated_data[param_name] = value
                continue

            validation_rule = cls.COMMON_VALIDATIONS[param_name]
            validated_value = cls._validate_parameter(param_name, value, validation_rule)
            validated_data[param_name] = validated_value

        return validated_data

    @classmethod
    def _validate_required_parameters(cls, request_data: Dict[str, Any], modality: str):
        """
        Validate that required parameters for the specified modality are present.

        Args:
            request_data: Request data to check
            modality: Target modality

        Raises:
            ValidationError: If required parameters are missing
        """
        if modality not in ModalityDetector.MODALITY_SIGNATURES:
            raise ValidationError(
                'modality',
                modality,
                f"Unsupported modality. Supported: {list(ModalityDetector.MODALITY_SIGNATURES.keys())}"
            )

        signature = ModalityDetector.MODALITY_SIGNATURES[modality]
        required_any = signature.get('required_any', [])
        required_all = signature.get('required_all', [])

        # Check if at least one of the required parameters is present
        if required_any:
            has_required = any(param in request_data for param in required_any)
            if not has_required:
                # Report the first required parameter as the field name for test compatibility
                missing_field = required_any[0]
                raise ValidationError(
                    missing_field,
                    'missing',
                    f"Required parameter for {modality}. Must include at least one of: {required_any}"
                )

        # Check that all required_all parameters are present
        if required_all:
            for param in required_all:
                if param not in request_data:
                    raise ValidationError(
                        param,
                        'missing',
                        f"Required parameter '{param}' for {modality} modality"
                    )

    @classmethod
    def _validate_parameter(cls, param_name: str, value: Any, rule: Dict[str, Any]) -> Any:
        """Validate a single parameter against its rule with type conversion."""

        # Type validation with conversion attempts
        expected_type = rule.get('type')
        if expected_type and not isinstance(value, expected_type):
            # Attempt type conversion for common cases
            converted_value = cls._attempt_type_conversion(value, expected_type)
            if converted_value is not None:
                value = converted_value
            else:
                raise ValidationError(
                    param_name,
                    str(value),
                    f"Expected type {expected_type.__name__}, got {type(value).__name__}"
                )

        # String validations
        if isinstance(value, str):
            min_length = rule.get('min_length')
            if min_length and len(value) < min_length:
                raise ValidationError(
                    param_name,
                    value,
                    f"String is empty, must be at least {min_length} characters in length"
                )

            max_length = rule.get('max_length')
            if max_length and len(value) > max_length:
                raise ValidationError(
                    param_name,
                    value,
                    f"Must be no more than {max_length} characters in length"
                )

        # Numeric validations
        if isinstance(value, (int, float)):
            min_value = rule.get('min_value')
            if min_value is not None and value < min_value:
                raise ValidationError(
                    param_name,
                    value,
                    f"Must be at least {min_value}"
                )

            max_value = rule.get('max_value')
            if max_value is not None and value > max_value:
                min_value = rule.get('min_value', 0)
                raise ValidationError(
                    param_name,
                    value,
                    f"Out of range. Must be between {min_value} and {max_value}"
                )

            multiple_of = rule.get('multiple_of')
            if multiple_of and value % multiple_of != 0:
                raise ValidationError(
                    param_name,
                    value,
                    f"Must be a multiple of {multiple_of}"
                )

        return value

    @classmethod
    def _attempt_type_conversion(cls, value: Any, expected_type: type) -> Any:
        """
        Attempt to convert value to expected type.

        Args:
            value: Value to convert
            expected_type: Target type

        Returns:
            Converted value if successful, None if conversion fails
        """
        try:
            # Handle tuple types (e.g., (int, float))
            if isinstance(expected_type, tuple):
                for single_type in expected_type:
                    converted = cls._attempt_type_conversion(value, single_type)
                    if converted is not None:
                        return converted
                return None

            # String to number conversions
            if expected_type == int and isinstance(value, str):
                return int(value)
            elif expected_type == float and isinstance(value, str):
                return float(value)
            elif expected_type == int and isinstance(value, float):
                return int(value) if value.is_integer() else None
            elif expected_type == float and isinstance(value, int):
                return float(value)

            # No conversion possible
            return None

        except (ValueError, TypeError):
            return None

    def validate_full_request(self, request_data: Dict[str, Any], modality: str) -> Optional[Dict[str, Any]]:
        """
        Perform complete request validation for specified modality.

        MMI-004 compliant implementation that validates parameters
        for an explicitly specified modality type.

        Args:
            request_data: Raw request data dictionary
            modality: Target modality for validation

        Returns:
            None if validation succeeds, error dict if validation fails

        Error dict format:
            {
                'field': str,     # Field name that failed validation
                'value': str,     # Field value that was invalid
                'message': str    # Human-readable error message
            }
        """
        try:
            # Step 1: Basic format validation
            validated_request = self.validate_request_format(request_data)

            # Step 2: Parameter validation for specified modality
            self.validate_parameters(validated_request, modality)

            logger.info(f"Request validation complete - modality: {modality}")
            return None  # Success

        except ValidationError as e:
            # Convert ValidationError to error dict
            error_dict = {
                'field': e.field,
                'value': e.value,
                'message': e.reason
            }
            logger.warning(f"Validation failed for modality {modality}: {error_dict}")
            return error_dict

        except Exception as e:
            # Handle unexpected errors
            error_dict = {
                'field': 'request',
                'value': 'unknown',
                'message': f"Validation system error: {str(e)}"
            }
            logger.error(f"Unexpected validation error for modality {modality}: {error_dict}")
            return error_dict