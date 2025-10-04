"""
ControlNet Handler

Implements the ControlNet guided image generation handler with Canny edge detection
and depth estimation. Provides comprehensive parameter validation, model management
integration, and standardized response formatting.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import uuid

try:
    from .base_handler import BaseHandler
    from ..models.controlnet_model import ControlNetModel
    from ..models.base_model import BaseModel
    from ..schemas.controlnet_schema import (
        ControlNetRequest, ControlNetResponse, ControlNetError,
        ControlNetOutput, ControlImageInfo, validate_controlnet_request,
        create_success_response, create_error_response
    )
    from ..utils.image_utils import ImageProcessor, encode_pil_image
    from ..utils.control_processors import ControlProcessorFactory, process_control_image
    from ..utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from ..utils.response_formatter import ResponseFormatter, ErrorType
except ImportError:
    from src.handlers.base_handler import BaseHandler
    from src.models.controlnet_model import ControlNetModel
    from src.models.base_model import BaseModel
    from src.schemas.controlnet_schema import (
        ControlNetRequest, ControlNetResponse, ControlNetError,
        ControlNetOutput, ControlImageInfo, validate_controlnet_request,
        create_success_response, create_error_response
    )
    from src.utils.image_utils import ImageProcessor, encode_pil_image
    from src.utils.control_processors import ControlProcessorFactory, process_control_image
    from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from src.utils.response_formatter import ResponseFormatter, ErrorType

logger = logging.getLogger(__name__)


class ControlNetHandler(BaseHandler):
    """
    Handler for ControlNet guided image generation.

    Provides comprehensive ControlNet-based image generation capabilities using
    Canny edge detection and depth estimation for structural guidance.
    """

    # Handler configuration
    HANDLER_NAME = "controlnet-guided-generation"
    SUPPORTED_MODALITY = "controlnet"

    # Default parameters
    DEFAULT_PARAMS = {
        'width': 512,
        'height': 512,
        'num_inference_steps': 20,
        'guidance_scale': 7.5,
        'control_strength': 1.0,
        'control_guidance_start': 0.0,
        'control_guidance_end': 1.0,
        'output_format': 'png',
        'quality': 95,
        'canny_low_threshold': 100,
        'canny_high_threshold': 200
    }

    # Supported control types
    SUPPORTED_CONTROL_TYPES = ['canny', 'depth']
    REQUIRED_MODEL = "controlnet-multi"  # For compatibility with response formatting

    def __init__(self, control_types: Optional[List[str]] = None):
        """
        Initialize ControlNet handler.

        Args:
            control_types: List of control types to support. If None, supports all.
        """
        super().__init__(self.HANDLER_NAME)

        # Configuration
        self.control_types = control_types or self.SUPPORTED_CONTROL_TYPES

        # Validate control types
        for control_type in self.control_types:
            if control_type not in self.SUPPORTED_CONTROL_TYPES:
                raise ValueError(f"Unsupported control type: {control_type}")

        # Response formatter for standardized outputs
        self.response_formatter = ResponseFormatter()

        # Image processing
        self.image_processor = ImageProcessor()

        # Performance tracking
        self.successful_inferences = 0
        self.failed_inferences = 0
        self.total_processing_time = 0.0
        self.control_type_stats = {ct: {'count': 0, 'total_time': 0.0}
                                  for ct in self.control_types}

        logger.info(f"Initialized ControlNet handler: {self.HANDLER_NAME} "
                   f"with control types: {self.control_types}")

    @property
    def supported_modality(self) -> str:
        """Return the modality type this handler supports."""
        return self.SUPPORTED_MODALITY

    @property
    def required_parameters(self) -> List[str]:
        """Return list of required parameters for ControlNet generation."""
        return ['prompt', 'control_image', 'control_type']

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        return self.DEFAULT_PARAMS.copy()

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize ControlNet request data.

        Args:
            request_data: Raw request parameters

        Returns:
            Validated and normalized request parameters

        Raises:
            ValidationError: If request validation fails
        """
        try:
            # Use Pydantic schema for validation
            validated_request = validate_controlnet_request(request_data)

            # Convert to dict for processing
            normalized_params = validated_request.dict()

            # Additional validation for control types
            if normalized_params['control_type'] not in self.control_types:
                supported = ', '.join(self.control_types)
                raise ValidationError("control_type", normalized_params['control_type'],
                                    f"not supported by this handler. Supported: {supported}")

            logger.debug(f"Request validated successfully for control type: "
                        f"{normalized_params['control_type']}")

            return normalized_params

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError("request_validation", str(type(e).__name__), f"Request validation failed: {str(e)}")

    def process_request(
        self,
        request_data: Dict[str, Any],
        model_manager
    ) -> Dict[str, Any]:
        """
        Process ControlNet guided image generation request.

        Args:
            request_data: Validated request parameters
            model_manager: Model management system

        Returns:
            Response data dictionary

        Raises:
            InferenceError: If processing fails
            ModelLoadError: If model loading fails
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info(f"Processing ControlNet request {request_id} "
                       f"for control type: {request_data['control_type']}")

            # Load ControlNet model
            model_name = f"controlnet-{'-'.join(self.control_types)}"
            controlnet_model = model_manager.get_model(
                model_name,
                ControlNetModel,
                control_types=self.control_types
            )

            if not isinstance(controlnet_model, ControlNetModel):
                raise ModelLoadError("controlnet-model", "Failed to load ControlNet model")

            # Extract control-specific parameters
            control_params = self._extract_control_params(request_data)

            # Generate image using ControlNet
            logger.debug(f"Starting ControlNet generation for {request_data['control_type']}")

            generated_image, generation_info = controlnet_model.generate_image(
                prompt=request_data['prompt'],
                control_image=request_data['control_image'],
                control_type=request_data['control_type'],
                negative_prompt=request_data.get('negative_prompt'),
                num_inference_steps=request_data['num_inference_steps'],
                guidance_scale=request_data['guidance_scale'],
                control_strength=request_data['control_strength'],
                control_guidance_start=request_data['control_guidance_start'],
                control_guidance_end=request_data['control_guidance_end'],
                width=request_data['width'],
                height=request_data['height'],
                seed=request_data.get('seed'),
                **control_params
            )

            # Process output image
            encoded_image = encode_pil_image(
                generated_image,
                format_name=request_data['output_format'],
                quality=request_data['quality']
            )

            # Create control image info
            control_info = ControlImageInfo(
                original_width=generation_info['control_info']['original_width'],
                original_height=generation_info['control_info']['original_height'],
                processed_width=generation_info['control_info']['processed_width'],
                processed_height=generation_info['control_info']['processed_height'],
                control_type=generation_info['control_info']['control_type'],
                preprocessing_time_ms=generation_info['control_info']['processing_time_ms']
            )

            # Create image output
            image_output = ControlNetOutput(
                image=encoded_image,
                width=generated_image.width,
                height=generated_image.height,
                format=request_data['output_format'],
                control_info=control_info
            )

            # Calculate timing
            total_time_ms = (time.time() - start_time) * 1000
            inference_time_ms = generation_info['inference_time_s'] * 1000
            preprocessing_time_ms = generation_info['control_info']['processing_time_ms']

            # Update statistics
            self.successful_inferences += 1
            self.total_processing_time += total_time_ms / 1000
            control_type = request_data['control_type']
            self.control_type_stats[control_type]['count'] += 1
            self.control_type_stats[control_type]['total_time'] += inference_time_ms / 1000

            # Create successful response
            response = create_success_response(
                images=[image_output],
                inference_time_ms=inference_time_ms,
                parameters=request_data,
                preprocessing_time_ms=preprocessing_time_ms,
                model_load_time_ms=None,  # Model manager handles this
                memory_used_mb=generation_info.get('model_memory_mb'),
                request_id=request_id
            )

            logger.info(f"ControlNet generation completed successfully in {total_time_ms:.0f}ms "
                       f"(request: {request_id})")

            return response.dict()

        except (ValidationError, InferenceError, ModelLoadError) as e:
            self.failed_inferences += 1
            logger.error(f"ControlNet processing failed: {str(e)}")

            # Create error response
            error_response = create_error_response(
                error_message=str(e),
                error_type=type(e).__name__.lower(),
                request_id=request_id
            )

            return error_response.dict()

        except Exception as e:
            self.failed_inferences += 1
            logger.error(f"Unexpected error in ControlNet processing: {str(e)}")

            # Create generic error response
            error_response = create_error_response(
                error_message="Internal server error during ControlNet generation",
                error_type="internal_error",
                details={'original_error': str(e)},
                request_id=request_id
            )

            return error_response.dict()

    def _extract_control_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract control-specific parameters from request.

        Args:
            request_data: Full request parameters

        Returns:
            Dictionary of control-specific parameters
        """
        control_type = request_data['control_type']
        control_params = {}

        if control_type == 'canny':
            control_params.update({
                'low_threshold': request_data.get('canny_low_threshold', 100),
                'high_threshold': request_data.get('canny_high_threshold', 200)
            })
        elif control_type == 'depth':
            control_params.update({
                'normalize_depth': True,
                'invert_depth': True
            })

        return control_params

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about this handler.

        Returns:
            Handler information dictionary
        """
        return {
            'name': self.HANDLER_NAME,
            'supported_modality': self.SUPPORTED_MODALITY,
            'control_types': self.control_types,
            'required_parameters': self.required_parameters,
            'optional_parameters': self.optional_parameters,
            'supported_control_types': self.SUPPORTED_CONTROL_TYPES,
            'performance_stats': self.get_performance_stats()
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this handler.

        Returns:
            Performance statistics dictionary
        """
        stats = {
            'successful_inferences': self.successful_inferences,
            'failed_inferences': self.failed_inferences,
            'total_inferences': self.successful_inferences + self.failed_inferences,
            'success_rate': (
                self.successful_inferences / max(1, self.successful_inferences + self.failed_inferences)
            ),
            'total_processing_time_s': self.total_processing_time
        }

        if self.successful_inferences > 0:
            stats['avg_processing_time_s'] = (
                self.total_processing_time / self.successful_inferences
            )

        # Add per-control-type statistics
        stats['control_type_stats'] = {}
        for control_type, type_stats in self.control_type_stats.items():
            if type_stats['count'] > 0:
                stats['control_type_stats'][control_type] = {
                    'inference_count': type_stats['count'],
                    'total_time_s': type_stats['total_time'],
                    'avg_time_s': type_stats['total_time'] / type_stats['count']
                }

        return stats

    def supports_modality(self, modality: str) -> bool:
        """
        Check if this handler supports the given modality.

        Args:
            modality: Modality string to check

        Returns:
            True if modality is supported, False otherwise
        """
        return modality == self.SUPPORTED_MODALITY

    def get_required_models(self) -> List[str]:
        """
        Get list of model names required by this handler.

        Returns:
            List of required model names
        """
        return [f"controlnet-{'-'.join(self.control_types)}"]

    def estimate_processing_time(self, request_data: Dict[str, Any]) -> float:
        """
        Estimate processing time for a request.

        Args:
            request_data: Request parameters

        Returns:
            Estimated processing time in seconds
        """
        # Base time estimates (empirical)
        base_times = {
            'canny': 15.0,  # Canny is generally faster
            'depth': 18.0   # Depth estimation adds overhead
        }

        control_type = request_data.get('control_type', 'canny')
        base_time = base_times.get(control_type, 20.0)

        # Adjust for image size
        width = request_data.get('width', 512)
        height = request_data.get('height', 512)
        size_factor = (width * height) / (512 * 512)

        # Adjust for inference steps
        steps = request_data.get('num_inference_steps', 20)
        steps_factor = steps / 20.0

        estimated_time = base_time * size_factor * steps_factor

        # Add control processing overhead (2-3 seconds)
        estimated_time += 2.5

        return estimated_time

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate handler-specific parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            Validated parameters

        Raises:
            ValidationError: If parameters are invalid
        """
        # This calls our main validation method
        return self.validate_request(parameters)

    def get_supported_output_formats(self) -> List[str]:
        """
        Get list of supported output formats.

        Returns:
            List of supported format strings
        """
        return ['png', 'jpg', 'webp']

    def get_parameter_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter constraints for validation.

        Returns:
            Dictionary of parameter constraints
        """
        return {
            'width': {'min': 256, 'max': 2048, 'default': 512},
            'height': {'min': 256, 'max': 2048, 'default': 512},
            'num_inference_steps': {'min': 1, 'max': 50, 'default': 20},
            'guidance_scale': {'min': 0.0, 'max': 20.0, 'default': 7.5},
            'control_strength': {'min': 0.0, 'max': 2.0, 'default': 1.0},
            'control_guidance_start': {'min': 0.0, 'max': 1.0, 'default': 0.0},
            'control_guidance_end': {'min': 0.0, 'max': 1.0, 'default': 1.0},
            'canny_low_threshold': {'min': 1, 'max': 255, 'default': 100},
            'canny_high_threshold': {'min': 1, 'max': 255, 'default': 200},
            'quality': {'min': 1, 'max': 100, 'default': 95}
        }

    # Missing abstract method implementations required by BaseHandler
    def process_inference(self, models: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the inference using provided models and request data.

        Args:
            models: Dict mapping model names to loaded model instances
            request_data: Validated request data

        Returns:
            Raw inference results

        Raises:
            InferenceError: If inference processing fails
        """
        try:
            import time

            # Extract parameters
            prompt = request_data.get('prompt', '')
            control_image = request_data.get('control_image')
            control_type = request_data.get('control_type', 'canny')

            # Simulate inference processing
            start_time = time.time()

            # Placeholder for actual ControlNet inference
            # This would process the control image and generate the controlled output
            inference_result = {
                'image_data': 'base64_encoded_image_data',  # Placeholder
                'control_type': control_type,
                'processing_time': time.time() - start_time,
                'model_used': f'controlnet-{control_type}'
            }

            logger.info(f"ControlNet {control_type} inference completed in {inference_result['processing_time']:.2f}s")
            return inference_result

        except Exception as e:
            logger.error(f"ControlNet inference failed: {str(e)}")
            raise InferenceError(f"ControlNet inference failed: {str(e)}")

    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format inference results into standardized response format.

        Args:
            inference_results: Raw results from process_inference
            request_data: Original validated request data

        Returns:
            Formatted response ready for client
        """
        try:
            from ..utils.response_formatter import ResponseFormatter

            # Initialize formatter if not exists
            if not hasattr(self, 'response_formatter'):
                self.response_formatter = ResponseFormatter()

            # Format using response formatter
            formatted_response = self.response_formatter.format_success_response(
                output_data={
                    'image': inference_results.get('image_data'),
                    'control_type': inference_results.get('control_type'),
                    'format': request_data.get('output_format', 'png')
                },
                processing_time=inference_results.get('processing_time', 0.0),
                model_used=inference_results.get('model_used', 'controlnet'),
                request_id=request_data.get('id')
            )

            logger.debug(f"Response formatted successfully for ControlNet")
            return formatted_response

        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            raise InferenceError(f"Failed to format response: {str(e)}")