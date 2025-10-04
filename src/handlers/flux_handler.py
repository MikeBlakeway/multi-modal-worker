"""
FLUX.1 Text-to-Image Handler

Implements the text-to-image generation handler using FLUX.1 Schnell fp8 model.
Provides comprehensive parameter validation, model management integration,
and standardized response formatting.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

try:
    from .base_handler import BaseHandler
    from ..models.flux_model import FluxModel
    from ..models.base_model import BaseModel
    from ..schemas.text_to_image_schema import (
        TextToImageRequest, TextToImageResponse, TextToImageError,
        ImageOutput, validate_text_to_image_request,
        create_success_response, create_error_response
    )
    from ..utils.image_utils import ImageProcessor, encode_pil_image
    from ..utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from ..utils.response_formatter import ResponseFormatter, ErrorType
except ImportError:
    from src.handlers.base_handler import BaseHandler
    from src.models.flux_model import FluxModel
    from src.models.base_model import BaseModel
    from src.schemas.text_to_image_schema import (
        TextToImageRequest, TextToImageResponse, TextToImageError,
        ImageOutput, validate_text_to_image_request,
        create_success_response, create_error_response
    )
    from src.utils.image_utils import ImageProcessor, encode_pil_image
    from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from src.utils.response_formatter import ResponseFormatter, ErrorType

logger = logging.getLogger(__name__)


class FluxHandler(BaseHandler):
    """
    Handler for FLUX.1 Schnell text-to-image generation.

    Provides comprehensive text-to-image generation capabilities using the
    FLUX.1 Schnell fp8 model with optimized memory usage and fast inference.
    """

    # Handler configuration
    HANDLER_NAME = "flux-text-to-image"
    SUPPORTED_MODALITY = "text-to-image"

    # Default parameters
    DEFAULT_PARAMS = {
        'width': 1024,
        'height': 1024,
        'num_inference_steps': 4,
        'guidance_scale': 0.0,  # FLUX Schnell default
        'output_format': 'png',
        'quality': 95
    }

    # Required model name
    FLUX_MODEL_NAME = "flux-1-schnell-fp8"
    REQUIRED_MODEL = "flux-1-schnell-fp8"  # For compatibility with response formatting

    def __init__(self):
        """Initialize FLUX.1 text-to-image handler."""
        super().__init__(self.HANDLER_NAME)

        # Response formatter for standardized outputs
        self.response_formatter = ResponseFormatter()

        # Performance tracking
        self.successful_inferences = 0
        self.failed_inferences = 0
        self.total_processing_time = 0.0

        logger.info(f"Initialized FLUX.1 handler: {self.HANDLER_NAME}")

    @property
    def supported_modality(self) -> str:
        """Return the modality type this handler supports."""
        return self.SUPPORTED_MODALITY

    @property
    def required_parameters(self) -> List[str]:
        """Return list of required parameters for text-to-image generation."""
        return ['prompt']

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        return self.DEFAULT_PARAMS.copy()

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize text-to-image request data.

        Args:
            request_data: Raw request data from client

        Returns:
            Normalized and validated request data

        Raises:
            ValidationError: If request data is invalid
        """
        try:
            # Use Pydantic schema for validation
            validated_request = validate_text_to_image_request(request_data)

            # Convert to dict for processing using Pydantic V2 method
            validated_data = validated_request.model_dump()

            # Add handler-specific metadata
            validated_data['modality'] = self.SUPPORTED_MODALITY
            validated_data['handler'] = self.HANDLER_NAME
            validated_data['timestamp'] = datetime.utcnow().isoformat() + "Z"

            logger.debug(f"Text-to-image request validated: {validated_data['prompt'][:50]}...")

            return validated_data

        except Exception as e:
            logger.error(f"Text-to-image validation failed: {e}")
            # Create ValidationError with proper constructor (field, value, reason)
            raise ValidationError("request", str(request_data), str(e))

    def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
        """
        Determine which models are needed for text-to-image generation.

        Args:
            request_data: Validated request data

        Returns:
            List of model names required for processing
        """
        return [self.FLUX_MODEL_NAME]

    def process_inference(self, models: Dict[str, BaseModel], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute FLUX.1 text-to-image inference using provided models.

        Args:
            models: Dict mapping model names to loaded model instances
            request_data: Validated request data

        Returns:
            Raw inference results

        Raises:
            InferenceError: If inference processing fails
        """
        try:
            # Get FLUX model
            flux_model = models.get(self.FLUX_MODEL_NAME)
            if not flux_model or not isinstance(flux_model, FluxModel):
                raise InferenceError(f"FLUX model not available: {self.FLUX_MODEL_NAME}")

            if not flux_model.is_loaded:
                raise InferenceError("FLUX model not loaded")

            # Prepare inference inputs
            inference_inputs = {
                'prompt': request_data['prompt'],
                'width': request_data.get('width', self.DEFAULT_PARAMS['width']),
                'height': request_data.get('height', self.DEFAULT_PARAMS['height']),
                'num_inference_steps': request_data.get('num_inference_steps', self.DEFAULT_PARAMS['num_inference_steps']),
                'guidance_scale': request_data.get('guidance_scale', self.DEFAULT_PARAMS['guidance_scale']),
                'seed': request_data.get('seed')
            }

            logger.info(f"Starting FLUX.1 inference for prompt: {inference_inputs['prompt'][:100]}...")

            # Perform inference
            inference_start = time.perf_counter()
            inference_result = flux_model.infer(inference_inputs)
            inference_time = time.perf_counter() - inference_start

            # Update performance tracking
            self.successful_inferences += 1
            self.total_processing_time += inference_time

            logger.info(f"FLUX.1 inference completed in {inference_time:.2f}s")

            return {
                'generated_image': inference_result['image'],
                'inference_time': inference_result['inference_time'],
                'parameters_used': inference_result['parameters'],
                'memory_usage_mb': inference_result.get('memory_usage_mb', 0),
                'model_info': inference_result.get('model_info', {}),
                'handler_processing_time': inference_time,
                'success': True
            }

        except Exception as e:
            self.failed_inferences += 1
            logger.error(f"FLUX.1 inference failed: {e}")

            if isinstance(e, (ValidationError, InferenceError)):
                raise
            else:
                raise InferenceError(f"Text-to-image inference failed: {e}")

    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format FLUX.1 inference results into standardized response format.

        Args:
            inference_results: Raw results from process_inference
            request_data: Original validated request data

        Returns:
            Formatted response ready for client
        """
        try:
            if not inference_results.get('success', False):
                # Handle inference failure
                return self.response_formatter.format_error_response(
                    ErrorType.INFERENCE_ERROR,
                    "Text-to-image generation failed",
                    {"details": "Inference did not complete successfully"}
                )

            # Extract generated image
            generated_image = inference_results['generated_image']

            # Get output format and quality from request
            output_format = request_data.get('output_format', self.DEFAULT_PARAMS['output_format'])
            quality = request_data.get('quality', self.DEFAULT_PARAMS['quality'])

            # Encode image to base64
            base64_data, file_size = encode_pil_image(generated_image, output_format, quality)

            # Create image output object
            image_output = ImageOutput(
                image_data=base64_data,
                format=output_format.lower(),
                width=generated_image.width,
                height=generated_image.height,
                file_size=file_size,
                seed_used=request_data.get('seed')
            )

            # Create success response
            response = create_success_response(
                images=[image_output],
                prompt=request_data['prompt'],
                parameters=inference_results['parameters_used'],
                inference_time=inference_results['inference_time'],
                model_info=inference_results.get('model_info', {}),
                peak_memory_mb=inference_results.get('memory_usage_mb')
            )

            # Convert to dict for response formatter using Pydantic V2 method
            response_dict = response.model_dump()

            # Add handler-specific metadata
            metadata = {
                'handler': self.HANDLER_NAME,
                'modality': self.SUPPORTED_MODALITY,
                'processing_stats': {
                    'successful_inferences': self.successful_inferences,
                    'failed_inferences': self.failed_inferences,
                    'average_processing_time': self.total_processing_time / max(1, self.successful_inferences),
                    'handler_processing_time': inference_results.get('handler_processing_time', 0)
                }
            }

            logger.info(f"FLUX.1 response formatted successfully, image size: {file_size} bytes")

            return self.response_formatter.format_success_response(
                output_data=response_dict,
                request_id=request_data.get('request_id', 'unknown'),
                modality=self.SUPPORTED_MODALITY,
                processing_time_ms=inference_results.get('inference_time', 0) * 1000,
                models_used=[self.REQUIRED_MODEL],
                additional_metadata=metadata
            )

        except Exception as e:
            logger.error(f"Response formatting failed: {e}")

            return self.response_formatter.format_error_response(
                error_message="Failed to format text-to-image response",
                error_type=ErrorType.INTERNAL_ERROR,
                request_id=request_data.get('request_id', 'unknown'),
                modality=self.SUPPORTED_MODALITY,
                details={"error": str(e), "handler": self.HANDLER_NAME}
            )

    def get_handler_stats(self) -> Dict[str, Any]:
        """Get comprehensive handler statistics."""
        return {
            'handler_name': self.HANDLER_NAME,
            'supported_modality': self.SUPPORTED_MODALITY,
            'successful_inferences': self.successful_inferences,
            'failed_inferences': self.failed_inferences,
            'total_inferences': self.successful_inferences + self.failed_inferences,
            'success_rate': self.successful_inferences / max(1, self.successful_inferences + self.failed_inferences),
            'average_processing_time': self.total_processing_time / max(1, self.successful_inferences),
            'total_processing_time': self.total_processing_time,
            'required_models': [self.FLUX_MODEL_NAME],
            'default_parameters': self.DEFAULT_PARAMS
        }

    def validate_model_compatibility(self, model: BaseModel) -> bool:
        """
        Validate that a model is compatible with this handler.

        Args:
            model: Model instance to validate

        Returns:
            True if model is compatible
        """
        return isinstance(model, FluxModel) and model.model_name == self.FLUX_MODEL_NAME

    def get_parameter_info(self) -> Dict[str, Any]:
        """Get detailed information about supported parameters."""
        return {
            'required_parameters': {
                'prompt': {
                    'type': 'string',
                    'description': 'Text description of the image to generate',
                    'min_length': 1,
                    'max_length': 2000,
                    'example': 'A beautiful sunset over mountains with realistic lighting'
                }
            },
            'optional_parameters': {
                'width': {
                    'type': 'integer',
                    'description': 'Width of generated image in pixels',
                    'default': 1024,
                    'minimum': 256,
                    'maximum': 2048,
                    'multiple_of': 8
                },
                'height': {
                    'type': 'integer',
                    'description': 'Height of generated image in pixels',
                    'default': 1024,
                    'minimum': 256,
                    'maximum': 2048,
                    'multiple_of': 8
                },
                'num_inference_steps': {
                    'type': 'integer',
                    'description': 'Number of denoising steps (more steps = better quality, slower)',
                    'default': 4,
                    'minimum': 1,
                    'maximum': 50,
                    'recommended': [4, 8, 16]
                },
                'guidance_scale': {
                    'type': 'float',
                    'description': 'How closely to follow the prompt (FLUX Schnell typically uses 0.0)',
                    'default': 0.0,
                    'minimum': 0.0,
                    'maximum': 20.0,
                    'recommended': 0.0
                },
                'seed': {
                    'type': 'integer',
                    'description': 'Random seed for reproducible generation (optional)',
                    'minimum': 0,
                    'maximum': 4294967295,
                    'example': 42
                },
                'output_format': {
                    'type': 'string',
                    'description': 'Output image format',
                    'default': 'png',
                    'options': ['png', 'jpeg', 'webp'],
                    'recommended': 'png'
                },
                'quality': {
                    'type': 'integer',
                    'description': 'Output quality for JPEG/WebP (1-100, ignored for PNG)',
                    'default': 95,
                    'minimum': 1,
                    'maximum': 100
                }
            }
        }

    def estimate_processing_time(self, request_data: Dict[str, Any]) -> float:
        """
        Estimate processing time for a request based on parameters.

        Args:
            request_data: Request parameters

        Returns:
            Estimated processing time in seconds
        """
        # Base time for FLUX.1 Schnell
        base_time = 8.0  # seconds for 1024x1024, 4 steps

        # Scale by resolution
        width = request_data.get('width', 1024)
        height = request_data.get('height', 1024)
        resolution_factor = (width * height) / (1024 * 1024)

        # Scale by steps
        steps = request_data.get('num_inference_steps', 4)
        steps_factor = steps / 4

        estimated_time = base_time * resolution_factor * steps_factor

        # Add overhead for first inference (model warmup)
        if self.successful_inferences == 0:
            estimated_time += 5.0

        return estimated_time