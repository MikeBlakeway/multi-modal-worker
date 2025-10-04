"""
Response Formatting Utilities

Provides standardized response formatting across all modalities to ensure
consistent output format for clients and proper integration with RunPod.
"""

import time
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResponseStatus(Enum):
    """Standardized response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"


class ErrorType(Enum):
    """Categorized error types for client handling."""
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    INFERENCE_ERROR = "inference_error"
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    INTERNAL_ERROR = "internal_error"


class ResponseFormatter:
    """
    Formats responses into standardized format across all modalities.

    Ensures consistent structure, error handling, and metadata inclusion
    for both successful and failed inference requests.
    """

    @staticmethod
    def format_success_response(
        output_data: Any,
        request_id: str,
        modality: str,
        processing_time_ms: float,
        models_used: List[str],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format successful inference response.

        Args:
            output_data: Primary inference output (image URLs, video URLs, etc.)
            request_id: Unique request identifier
            modality: Modality type that processed the request
            processing_time_ms: Total processing time in milliseconds
            models_used: List of models that were used for inference
            additional_metadata: Optional extra metadata to include

        Returns:
            Standardized success response dictionary
        """
        metadata = {
            'request_id': request_id,
            'modality': modality,
            'processing_time_ms': round(processing_time_ms, 2),
            'models_used': models_used,
            'timestamp': time.time(),
            'status': ResponseStatus.SUCCESS.value
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        response = {
            'status': ResponseStatus.SUCCESS.value,
            'output': output_data,
            'metadata': metadata
        }

        logger.debug(f"[{request_id}] Formatted success response")
        return response

    @staticmethod
    def format_error_response(
        error_message: str,
        error_type: ErrorType,
        request_id: str,
        modality: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Format error response with detailed information.

        Args:
            error_message: Human-readable error description
            error_type: Categorized error type
            request_id: Unique request identifier
            modality: Modality type (if determined)
            details: Additional error context and details
            suggestions: Suggested actions for resolving the error

        Returns:
            Standardized error response dictionary
        """
        error_info = {
            'message': error_message,
            'type': error_type.value,
            'timestamp': time.time(),
            'request_id': request_id
        }

        if modality:
            error_info['modality'] = modality

        if details:
            error_info['details'] = details

        if suggestions:
            error_info['suggestions'] = suggestions

        response = {
            'status': ResponseStatus.ERROR.value,
            'error': error_info
        }

        logger.debug(f"[{request_id}] Formatted error response: {error_type.value}")
        return response

    @staticmethod
    def format_validation_error(
        field_name: str,
        field_value: Any,
        validation_message: str,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Format validation error with specific field information.

        Args:
            field_name: Name of the invalid field
            field_value: Value that failed validation
            validation_message: Specific validation failure message
            request_id: Unique request identifier

        Returns:
            Standardized validation error response
        """
        suggestions = [
            f"Check the '{field_name}' parameter value",
            "Refer to API documentation for valid parameter formats",
            "Verify all required parameters are provided"
        ]

        details = {
            'field': field_name,
            'provided_value': str(field_value),
            'validation_rule': validation_message
        }

        return ResponseFormatter.format_error_response(
            error_message=f"Validation failed for parameter '{field_name}': {validation_message}",
            error_type=ErrorType.VALIDATION_ERROR,
            request_id=request_id,
            details=details,
            suggestions=suggestions
        )

    @staticmethod
    def format_modality_not_supported_error(
        requested_modality: str,
        supported_modalities: List[str],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Format error for unsupported modality requests.

        Args:
            requested_modality: The modality that was requested
            supported_modalities: List of currently supported modalities
            request_id: Unique request identifier

        Returns:
            Standardized modality error response
        """
        suggestions = [
            f"Use one of the supported modalities: {', '.join(supported_modalities)}",
            "Check spelling of the modality parameter",
            "Refer to API documentation for current modality support"
        ]

        details = {
            'requested_modality': requested_modality,
            'supported_modalities': supported_modalities
        }

        return ResponseFormatter.format_error_response(
            error_message=f"Modality '{requested_modality}' is not currently supported",
            error_type=ErrorType.VALIDATION_ERROR,
            request_id=request_id,
            details=details,
            suggestions=suggestions
        )

    @staticmethod
    def format_model_loading_error(
        model_name: str,
        error_details: str,
        request_id: str,
        modality: str
    ) -> Dict[str, Any]:
        """
        Format error for model loading failures.

        Args:
            model_name: Name of the model that failed to load
            error_details: Detailed error message
            request_id: Unique request identifier
            modality: Modality that required the model

        Returns:
            Standardized model loading error response
        """
        suggestions = [
            "Try the request again (model loading is retried automatically)",
            "Check if sufficient GPU memory is available",
            "Contact support if the problem persists"
        ]

        details = {
            'model_name': model_name,
            'loading_error': error_details,
            'modality': modality
        }

        return ResponseFormatter.format_error_response(
            error_message=f"Failed to load required model '{model_name}': {error_details}",
            error_type=ErrorType.MODEL_ERROR,
            request_id=request_id,
            modality=modality,
            details=details,
            suggestions=suggestions
        )

    @staticmethod
    def format_inference_error(
        error_details: str,
        request_id: str,
        modality: str,
        models_attempted: List[str]
    ) -> Dict[str, Any]:
        """
        Format error for inference processing failures.

        Args:
            error_details: Detailed error message
            request_id: Unique request identifier
            modality: Modality that was being processed
            models_attempted: List of models that were attempted

        Returns:
            Standardized inference error response
        """
        suggestions = [
            "Verify input parameters are within acceptable ranges",
            "Try reducing complexity of the request",
            "Check if all required parameters are provided"
        ]

        details = {
            'inference_error': error_details,
            'models_used': models_attempted
        }

        return ResponseFormatter.format_error_response(
            error_message=f"Inference processing failed: {error_details}",
            error_type=ErrorType.INFERENCE_ERROR,
            request_id=request_id,
            modality=modality,
            details=details,
            suggestions=suggestions
        )

    @staticmethod
    def add_runpod_compatibility(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure response is compatible with RunPod serverless format.

        Args:
            response: Standardized response dictionary

        Returns:
            RunPod-compatible response format
        """
        # RunPod expects specific field names for outputs and errors
        if response.get('status') == ResponseStatus.SUCCESS.value:
            # For successful responses, RunPod expects 'output' field
            runpod_response = {
                'output': response.get('output'),
                'metadata': response.get('metadata', {}),
                'status': 'success'
            }
        else:
            # For error responses, RunPod expects 'error' field
            runpod_response = {
                'error': response.get('error', {}),
                'status': 'error'
            }

        return runpod_response

    @staticmethod
    def format_system_status_response(
        system_stats: Dict[str, Any],
        supported_modalities: List[str],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Format system status and health check response.

        Args:
            system_stats: Current system statistics and health metrics
            supported_modalities: List of currently supported modalities
            request_id: Unique request identifier

        Returns:
            Formatted system status response
        """
        output_data = {
            'system_status': 'healthy',
            'supported_modalities': supported_modalities,
            'system_stats': system_stats,
            'capabilities': {
                'concurrent_requests': True,
                'model_management': True,
                'automatic_eviction': True,
                'gpu_monitoring': system_stats.get('gpu_available', False)
            }
        }

        return ResponseFormatter.format_success_response(
            output_data=output_data,
            request_id=request_id,
            modality='system-status',
            processing_time_ms=0.0,
            models_used=[]
        )