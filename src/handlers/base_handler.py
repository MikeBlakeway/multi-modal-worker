"""
Base Handler Abstract Class

Defines the interface that all modality-specific handlers must implement.
Provides common functionality for request processing, validation, and response formatting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging

try:
    # Try relative imports first (when running as package)
    from ..models import BaseModel, model_manager
    from ..utils import ValidationError, InferenceError, config
except ImportError:
    # Fall back to absolute imports (when running as script or in tests)
    from src.models import BaseModel, model_manager
    from src.utils import ValidationError, InferenceError, config

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """
    Abstract base class for all modality handlers.

    Each modality (text-to-image, image-to-video, etc.) should implement
    this interface to provide consistent request processing across the system.
    """

    def __init__(self, handler_name: str):
        """
        Initialize the base handler.

        Args:
            handler_name: Unique identifier for this handler type
        """
        self.handler_name = handler_name
        self.model_manager = model_manager
        self._request_count = 0
        self._total_processing_time = 0.0

        logger.info(f"Initialized {self.handler_name} handler")

    @property
    @abstractmethod
    def supported_modality(self) -> str:
        """Return the modality type this handler supports."""
        pass

    @property
    @abstractmethod
    def required_parameters(self) -> List[str]:
        """Return list of required parameters for this modality."""
        pass

    @property
    @abstractmethod
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        pass

    @abstractmethod
    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize request data for this modality.

        Args:
            request_data: Raw request data from client

        Returns:
            Normalized and validated request data

        Raises:
            ValidationError: If request data is invalid
        """
        pass

    @abstractmethod
    def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
        """
        Determine which models are needed for this request.

        Args:
            request_data: Validated request data

        Returns:
            List of model names required for processing
        """
        pass

    @abstractmethod
    def process_inference(self, models: Dict[str, BaseModel], request_data: Dict[str, Any]) -> Dict[str, Any]:
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
        pass

    @abstractmethod
    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format inference results into standardized response format.

        Args:
            inference_results: Raw results from process_inference
            request_data: Original validated request data

        Returns:
            Formatted response ready for client
        """
        pass

    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main request processing pipeline that orchestrates the entire flow.

        Args:
            request_data: Raw request data from client

        Returns:
            Complete formatted response
        """
        start_time = time.perf_counter()
        request_id = request_data.get('id', f"{self.handler_name}_{self._request_count}")

        try:
            logger.info(f"[{request_id}] Processing {self.supported_modality} request")

            # Step 1: Validate request
            validated_data = self.validate_request(request_data)
            logger.debug(f"[{request_id}] Request validation completed")

            # Step 2: Determine required models
            required_models = self.get_required_models(validated_data)
            logger.debug(f"[{request_id}] Required models: {required_models}")

            # Step 3: Load models
            loaded_models = {}
            for model_name in required_models:
                try:
                    loaded_models[model_name] = self.model_manager.get_model(model_name)
                    logger.debug(f"[{request_id}] Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"[{request_id}] Failed to load model {model_name}: {e}")
                    raise InferenceError(f"Model loading failed: {model_name} - {str(e)}")

            # Step 4: Process inference
            inference_results = self.process_inference(loaded_models, validated_data)
            logger.debug(f"[{request_id}] Inference completed")

            # Step 5: Format response
            formatted_response = self.format_response(inference_results, validated_data)

            # Add metadata
            processing_time = time.perf_counter() - start_time
            formatted_response['metadata'] = {
                'request_id': request_id,
                'handler': self.handler_name,
                'modality': self.supported_modality,
                'processing_time_ms': round(processing_time * 1000, 2),
                'models_used': list(loaded_models.keys())
            }

            # Update statistics
            self._request_count += 1
            self._total_processing_time += processing_time

            logger.info(f"[{request_id}] Request completed in {processing_time:.3f}s")
            return formatted_response

        except ValidationError as e:
            logger.warning(f"[{request_id}] Validation error: {e}")
            return {
                'error': f"Validation failed: {str(e)}",
                'error_type': 'validation_error',
                'request_id': request_id,
                'handler': self.handler_name
            }

        except InferenceError as e:
            logger.error(f"[{request_id}] Inference error: {e}")
            return {
                'error': f"Inference failed: {str(e)}",
                'error_type': 'inference_error',
                'request_id': request_id,
                'handler': self.handler_name
            }

        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
            return {
                'error': f"Internal error: {str(e)}",
                'error_type': 'internal_error',
                'request_id': request_id,
                'handler': self.handler_name
            }

    def get_handler_stats(self) -> Dict[str, Any]:
        """Return performance statistics for this handler."""
        avg_processing_time = (
            self._total_processing_time / self._request_count
            if self._request_count > 0 else 0.0
        )

        return {
            'handler_name': self.handler_name,
            'supported_modality': self.supported_modality,
            'request_count': self._request_count,
            'total_processing_time_s': round(self._total_processing_time, 3),
            'average_processing_time_s': round(avg_processing_time, 3),
            'required_parameters': self.required_parameters,
            'optional_parameters': list(self.optional_parameters.keys())
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._request_count = 0
        self._total_processing_time = 0.0
        logger.info(f"Reset statistics for {self.handler_name} handler")