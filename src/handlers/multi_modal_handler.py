"""
Multi-Modal Handler Routing System

Main routing class that orchestrates request flow through validation,
modality detection, handler selection, and response formatting.
"""

import uuid
import time
from typing import Dict, Any, List, Optional
import logging

try:
    # Try relative imports first (when run as package)
    from ..utils.request_validator import RequestValidator, ModalityDetector
    from ..utils.response_formatter import ResponseFormatter, ErrorType
    from ..utils.logging_config import LoggingConfig, get_request_logger, log_request_start, log_request_complete
    from ..models.model_manager import ModelManager
    from .base_handler import BaseHandler
    from .flux_handler import FluxHandler  # MMI-005: FLUX.1 Text-to-Image Handler
except ImportError:
    # Fall back to absolute imports (when run from tests)
    from utils.request_validator import RequestValidator, ModalityDetector
    from utils.response_formatter import ResponseFormatter, ErrorType
    from utils.logging_config import LoggingConfig, get_request_logger, log_request_start, log_request_complete
    from models.model_manager import ModelManager
    from handlers.base_handler import BaseHandler
    from handlers.flux_handler import FluxHandler  # MMI-005: FLUX.1 Text-to-Image Handler

# Additional handler imports (MMI-006, MMI-007, MMI-008)
try:
    from .controlnet_handler import ControlNetHandler  # MMI-006: ControlNet Integration
    from .animatediff_handler import AnimateDiffHandler  # MMI-007: AnimateDiff Integration
    from .ltx_video_handler import LTXVideoHandler  # MMI-008: LTX-Video Text-to-Video
except ImportError:
    from handlers.controlnet_handler import ControlNetHandler  # MMI-006: ControlNet Integration
    from handlers.animatediff_handler import AnimateDiffHandler  # MMI-007: AnimateDiff Integration
    from handlers.ltx_video_handler import LTXVideoHandler  # MMI-008: LTX-Video Text-to-Video

# Future handlers (commented out for now)
# from .inpainting_handler import InpaintingHandler
# from .camera_control_handler import CameraControlHandler


class MultiModalHandler:
    """
    Central routing system for multi-modal inference requests.

    Manages the complete request lifecycle from initial validation through
    response formatting, providing a unified interface for all modalities.
    """

    def __init__(self, model_manager: ModelManager, auto_initialize: bool = True):
        """
        Initialize the multi-modal handler.

        Args:
            model_manager: Model management system instance
            auto_initialize: Whether to automatically initialize handlers (default: True, False for testing)
        """
        self.model_manager = model_manager
        self.request_validator = RequestValidator()
        self.modality_detector = ModalityDetector()
        self.response_formatter = ResponseFormatter()

        # Initialize logger first (needed by _initialize_handlers)
        self.logger = get_request_logger()

        # Handler registry - will be populated as handlers are implemented
        self.handlers: Dict[str, BaseHandler] = {}

        # Initialize available handlers if requested
        if auto_initialize:
            self._initialize_handlers()

        # Initialize supported modalities list
        self.supported_modalities = list(self.handlers.keys())

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0

    def _initialize_handlers(self):
        """Initialize all available handlers."""
        try:
            # Initialize FLUX.1 text-to-image handler (MMI-005)
            flux_handler = FluxHandler()
            self.register_handler(flux_handler.supported_modality, flux_handler)

            # Initialize ControlNet handler (MMI-006)
            controlnet_handler = ControlNetHandler()
            self.register_handler(controlnet_handler.supported_modality, controlnet_handler)

            # Initialize AnimateDiff handler (MMI-007)
            animatediff_handler = AnimateDiffHandler()
            self.register_handler(animatediff_handler.supported_modality, animatediff_handler)

            # Initialize LTX-Video handler (MMI-008)
            ltx_video_handler = LTXVideoHandler()
            self.register_handler(ltx_video_handler.supported_modality, ltx_video_handler)

            # Future handlers will be added here as they are implemented

            self.logger.info(f"Initialized {len(self.handlers)} handlers")

        except Exception as e:
            self.logger.error(f"Failed to initialize some handlers: {e}")
            # Continue with whatever handlers loaded successfully

    def register_handler(self, modality: str, handler: BaseHandler):
        """
        Register a handler for a specific modality.

        Args:
            modality: Modality name (e.g., 'text-to-image')
            handler: Handler instance implementing BaseHandler
        """
        self.handlers[modality] = handler
        self.supported_modalities = list(self.handlers.keys())
        self.logger.info(f"Registered handler for modality: {modality}")

    def get_supported_modalities(self) -> List[str]:
        """
        Get list of currently supported modalities.

        Returns:
            List of supported modality names
        """
        return self.supported_modalities.copy()

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete multi-modal inference request.

        Args:
            request_data: Complete request parameters

        Returns:
            Formatted response (success or error)
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            with LoggingConfig.request_context(request_id):
                self.request_count += 1

                # Step 1: Detect modality from request parameters
                detected_modality = self._detect_modality(request_data, request_id)
                if not detected_modality:
                    return self._handle_modality_detection_failure(request_data, request_id)

                # Update logging context with detected modality
                with LoggingConfig.request_context(request_id, detected_modality):
                    log_request_start(request_id, detected_modality, request_data)

                    # Step 2: Validate request parameters
                    validation_result = self._validate_request(
                        request_data, detected_modality, request_id
                    )
                    if not validation_result['valid']:
                        return validation_result['error_response']

                    # Step 3: Check if handler is available for modality
                    if detected_modality not in self.handlers:
                        return self._handle_unsupported_modality(
                            detected_modality, request_id
                        )

                    # Step 4: Process request through appropriate handler
                    handler = self.handlers[detected_modality]
                    processing_result = self._process_with_handler(
                        handler, request_data, request_id, detected_modality
                    )

                    # Step 5: Format and return response
                    processing_time_ms = (time.time() - start_time) * 1000
                    self.total_processing_time += processing_time_ms

                    log_request_complete(request_id, True, processing_time_ms)
                    return processing_result

        except Exception as e:
            # Handle unexpected errors
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"[{request_id}] Unexpected error: {str(e)}")
            log_request_complete(request_id, False, processing_time_ms)

            return self.response_formatter.format_error_response(
                error_message=f"Internal processing error: {str(e)}",
                error_type=ErrorType.INTERNAL_ERROR,
                request_id=request_id,
                modality=request_data.get('modality'),
                details={'exception_type': type(e).__name__}
            )

    def _detect_modality(self, request_data: Dict[str, Any], request_id: str) -> Optional[str]:
        """
        Detect modality from request parameters.

        Args:
            request_data: Request parameters
            request_id: Request identifier

        Returns:
            Detected modality name or None if detection failed
        """
        try:
            # Check for explicit modality parameter first
            if 'modality' in request_data:
                explicit_modality = request_data['modality']
                if explicit_modality in self.supported_modalities:
                    self.logger.info(f"[{request_id}] Using explicit modality: {explicit_modality}")
                    return explicit_modality
                else:
                    self.logger.warning(f"[{request_id}] Explicit modality '{explicit_modality}' not supported")

            # Use automatic detection
            detected = self.modality_detector.detect_modality(request_data)
            if detected:
                self.logger.info(f"[{request_id}] Auto-detected modality: {detected}")
            else:
                self.logger.warning(f"[{request_id}] Could not auto-detect modality from parameters")

            return detected

        except Exception as e:
            self.logger.error(f"[{request_id}] Error detecting modality: {str(e)}")
            return None

    def _validate_request(
        self,
        request_data: Dict[str, Any],
        modality: str,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Validate request parameters for the detected modality.

        Args:
            request_data: Request parameters
            modality: Detected modality
            request_id: Request identifier

        Returns:
            Validation result with 'valid' flag and optional 'error_response'
        """
        try:
            validation_error = self.request_validator.validate_full_request(
                request_data, modality
            )

            if validation_error:
                error_response = self.response_formatter.format_validation_error(
                    field_name=validation_error.get('field', 'unknown'),
                    field_value=validation_error.get('value', 'unknown'),
                    validation_message=validation_error.get('message', 'Validation failed'),
                    request_id=request_id
                )
                return {'valid': False, 'error_response': error_response}

            self.logger.info(f"[{request_id}] Request validation passed")
            return {'valid': True}

        except Exception as e:
            self.logger.error(f"[{request_id}] Validation error: {str(e)}")
            error_response = self.response_formatter.format_error_response(
                error_message=f"Validation system error: {str(e)}",
                error_type=ErrorType.INTERNAL_ERROR,
                request_id=request_id,
                modality=modality
            )
            return {'valid': False, 'error_response': error_response}

    def _handle_modality_detection_failure(
        self,
        request_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Handle cases where modality detection fails.

        Args:
            request_data: Original request parameters
            request_id: Request identifier

        Returns:
            Error response for modality detection failure
        """
        suggestions = [
            "Include a 'modality' parameter in your request",
            f"Supported modalities: {', '.join(self.supported_modalities)}",
            "Check parameter names match expected format for automatic detection"
        ]

        details = {
            'provided_parameters': list(request_data.keys()),
            'supported_modalities': self.supported_modalities,
            'auto_detection_failed': True
        }

        return self.response_formatter.format_error_response(
            error_message="Could not determine request modality from parameters",
            error_type=ErrorType.VALIDATION_ERROR,
            request_id=request_id,
            details=details,
            suggestions=suggestions
        )

    def _handle_unsupported_modality(
        self,
        modality: str,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Handle requests for unsupported modalities.

        Args:
            modality: Requested modality
            request_id: Request identifier

        Returns:
            Error response for unsupported modality
        """
        return self.response_formatter.format_modality_not_supported_error(
            requested_modality=modality,
            supported_modalities=self.supported_modalities,
            request_id=request_id
        )

    def _process_with_handler(
        self,
        handler: BaseHandler,
        request_data: Dict[str, Any],
        request_id: str,
        modality: str
    ) -> Dict[str, Any]:
        """
        Process request using the appropriate modality handler.

        Args:
            handler: Handler instance for the modality
            request_data: Request parameters
            request_id: Request identifier
            modality: Processing modality

        Returns:
            Formatted response from handler processing
        """
        try:
            self.logger.info(f"[{request_id}] Processing with {modality} handler")

            # Ensure request_id is in request_data for handler
            if 'id' not in request_data:
                request_data['id'] = request_id

            # Process through handler
            result = handler.handle_request(request_data)

            self.logger.info(f"[{request_id}] Handler processing completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"[{request_id}] Handler processing failed: {str(e)}")

            return self.response_formatter.format_error_response(
                error_message=f"Processing failed in {modality} handler: {str(e)}",
                error_type=ErrorType.INFERENCE_ERROR,
                request_id=request_id,
                modality=modality,
                details={'handler_error': str(e)}
            )

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics.

        Returns:
            System status information
        """
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0.0
        )

        # Get memory and model stats from ModelManager
        manager_status = self.model_manager.get_manager_status()
        memory_summary = manager_status.get('memory_summary', {})
        memory_stats = memory_summary.get('stats', {})

        model_stats = {
            'loaded_models': manager_status.get('loaded_count', 0),
            'available_vram': memory_stats.get('gpu_free_mb', 0) / 1024.0,  # Convert MB to GB
            'total_vram': memory_stats.get('gpu_total_mb', 0) / 1024.0  # Convert MB to GB
        }

        status = {
            'service': 'multi-modal-inference-worker',
            'status': 'healthy',
            'supported_modalities': self.supported_modalities,
            'statistics': {
                'total_requests': self.request_count,
                'average_processing_time_ms': round(avg_processing_time, 2)
            },
            'system': {
                'memory': memory_stats,
                'models': model_stats
            }
        }

        return status

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return formatted response.

        Returns:
            Health check response
        """
        request_id = f"health-{int(time.time())}"
        system_stats = self.get_system_status()

        return self.response_formatter.format_system_status_response(
            system_stats=system_stats,
            supported_modalities=self.supported_modalities,
            request_id=request_id
        )