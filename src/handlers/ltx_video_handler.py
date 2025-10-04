"""
LTX-Video Handler

Implements the LTX-Video text-to-video generation handler with DiT-based architecture.
Provides comprehensive parameter validation, model management integration,
and standardized response formatting for video generation workflows.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import uuid

try:
    from .base_handler import BaseHandler
    from ..models.ltx_video_model import LTXVideoModel
    from ..schemas.text_to_video_schema import (
        TextToVideoRequest, TextToVideoResponse, VideoInfo,
        validate_text_to_video_request, create_success_response,
        create_error_response
    )
    from ..utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from ..utils.response_formatter import ResponseFormatter, ErrorType
    from ..models.memory_monitor import MemoryMonitor
except ImportError:
    from src.handlers.base_handler import BaseHandler
    from src.models.ltx_video_model import LTXVideoModel
    from src.schemas.text_to_video_schema import (
        TextToVideoRequest, TextToVideoResponse, VideoInfo,
        validate_text_to_video_request, create_success_response,
        create_error_response
    )
    from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from src.utils.response_formatter import ResponseFormatter, ErrorType
    from src.models.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class LTXVideoHandler(BaseHandler):
    """
    Handler for LTX-Video text-to-video generation.

    Provides comprehensive text-to-video generation capabilities using LTX-Video 2B
    distilled model with real-time generation and DiT-based architecture.
    """

    HANDLER_NAME = "ltx-video-text-to-video"
    SUPPORTED_MODALITY = "text-to-video"

    def __init__(self,
                 model_id: str = None,
                 auto_load: bool = False):
        """
        Initialize LTX-Video handler.

        Args:
            model_id: Custom model ID for LTX-Video model
            auto_load: Whether to automatically load model on initialization
        """
        super().__init__(self.HANDLER_NAME)

        # Store constructor parameters
        self._model_id = model_id

        # Initialize model
        self.model = LTXVideoModel()

        # Response formatter for standardized outputs
        self.response_formatter = ResponseFormatter()

        # Memory monitor for resource management
        self.memory_monitor = MemoryMonitor()

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.successful_requests = 0
        self.failed_requests = 0

        # Auto-load model if requested
        if auto_load:
            self.ensure_model_loaded()

        logger.info(f"LTXVideoHandler initialized with model: {self.model.MODEL_ID}")

    @property
    def supported_modality(self) -> str:
        """Return the modality this handler supports."""
        return self.SUPPORTED_MODALITY

    @property
    def required_parameters(self) -> List[str]:
        """Return list of required parameters for this modality."""
        return ['prompt']

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        return {
            'width': 720,
            'height': 1280,
            'num_frames': 25,  # (8*3)+1 for LTX-Video
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'fps': 8
        }

    def get_required_models(self, request_data: Dict[str, Any]) -> List[str]:
        """
        Determine which models are needed for this request.

        Args:
            request_data: Validated request data

        Returns:
            List of model names required for processing
        """
        # For LTX-Video, we only need the LTX-Video model itself
        return ['ltx-video']

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
            # For LTX-Video, we use the integrated model directly
            # The base handler pattern expects external model management,
            # but our implementation uses an integrated approach

            # Ensure our model is loaded
            if not self.model.is_loaded:
                self.model.load_model()

            # Preprocess prompt
            enhanced_prompt = self.preprocess_prompt(request_data['prompt'])

            # Generate video
            video_data = self.model.generate_video(
                prompt=enhanced_prompt,
                width=request_data['width'],
                height=request_data['height'],
                num_frames=request_data['num_frames'],
                num_inference_steps=request_data['num_inference_steps'],
                guidance_scale=request_data['guidance_scale'],
                fps=request_data['fps']
            )

            return {
                'video_data': video_data,
                'video_info': {
                    'width': request_data['width'],
                    'height': request_data['height'],
                    'num_frames': request_data['num_frames'],
                    'fps': request_data['fps'],
                    'duration': request_data['num_frames'] / request_data['fps']
                }
            }

        except Exception as e:
            raise InferenceError(f"Video generation failed: {str(e)}")

    def format_response(self, inference_results: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format inference results into standardized response format.

        Args:
            inference_results: Raw results from process_inference
            request_data: Original validated request data

        Returns:
            Formatted response ready for client
        """
        return {
            'success': True,
            'data': {
                'video_data': inference_results['video_data'],
                'video_info': inference_results['video_info'],
                'metadata': {
                    'model_type': 'ltx-video-2b',
                    'handler_version': '1.0.0',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        }

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize request data for LTX-Video generation.

        Args:
            request_data: Raw request data from client

        Returns:
            Normalized and validated request data

        Raises:
            ValidationError: If request data is invalid
        """
        # Check required parameters
        for param in self.required_parameters:
            if param not in request_data or not request_data[param]:
                raise ValidationError(param, str(request_data.get(param, 'None')), "Missing required parameter")

        # Validate prompt
        prompt = request_data.get('prompt', '').strip()
        if not prompt:
            raise ValidationError('prompt', prompt, "Empty or invalid prompt")

        # Apply defaults and validate optional parameters
        validated_data = request_data.copy()
        for param, default_value in self.optional_parameters.items():
            if param not in validated_data:
                validated_data[param] = default_value

        # Validate and optimize parameters for LTX-Video constraints
        validated_data = self.optimize_parameters(validated_data)

        # Additional validation using schema
        try:
            validate_text_to_video_request(validated_data)
        except Exception as e:
            raise ValidationError('schema', str(validated_data), f"Schema validation failed: {str(e)}")

        return validated_data

    def optimize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters for LTX-Video constraints and performance.

        Args:
            params: Input parameters

        Returns:
            Optimized parameters
        """
        optimized = params.copy()

        # Optimize dimensions to be divisible by 32 (only if not already valid)
        width = params.get('width', 720)
        height = params.get('height', 1280)

        if width % 32 != 0:
            optimized['width'] = ((width + 31) // 32) * 32
        if height % 32 != 0:
            optimized['height'] = ((height + 31) // 32) * 32

        # Validate and correct frame count to (8*n)+1 pattern
        num_frames = params.get('num_frames', 25)
        if (num_frames - 1) % 8 != 0:
            # Find closest valid frame count (round up to next valid value)
            base_frames = (num_frames - 1 + 7) // 8  # Round up
            optimized['num_frames'] = (base_frames * 8) + 1

        # Ensure reasonable ranges
        optimized['width'] = max(256, min(optimized['width'], 1280))
        optimized['height'] = max(256, min(optimized['height'], 1280))
        optimized['num_frames'] = max(9, min(optimized['num_frames'], 257))  # 8*n+1 format

        return optimized

    def preprocess_prompt(self, prompt: str) -> str:
        """
        Preprocess prompt for optimal LTX-Video results.

        Args:
            prompt: Input text prompt

        Returns:
            Enhanced prompt with detailed descriptions
        """
        if len(prompt) < 20:
            # Enhance short prompts with temporal and visual details
            enhanced = f"{prompt}, detailed cinematic video, smooth motion, high quality"
            return enhanced
        return prompt

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a text-to-video generation request using BaseHandler pattern.

        Args:
            request_data: Raw request data

        Returns:
            Complete response dictionary
        """
        # Use the BaseHandler's orchestrated workflow
        return self.handle_request(request_data)

    def format_response(self,
                       success: bool = True,
                       video_data: Optional[str] = None,
                       request_params: Optional[Dict[str, Any]] = None,
                       processing_time: Optional[float] = None,
                       error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Alternative format_response for backward compatibility with tests.

        Args:
            success: Whether generation was successful
            video_data: Base64 encoded video data
            request_params: Original request parameters
            processing_time: Time taken for processing
            error_message: Error message if failed

        Returns:
            Formatted response dictionary
        """
        if success and video_data and request_params:
            video_info = {
                'width': request_params['width'],
                'height': request_params['height'],
                'num_frames': request_params['num_frames'],
                'fps': request_params['fps'],
                'duration': request_params['num_frames'] / request_params['fps']
            }

            metadata = {
                'model_type': 'ltx-video-2b',
                'processing_time': processing_time,
                'handler_version': '1.0.0',
                'timestamp': datetime.utcnow().isoformat()
            }

            return {
                'success': True,
                'data': {
                    'video_data': video_data,
                    'video_info': video_info,
                    'metadata': metadata
                }
            }
        else:
            return {
                'success': False,
                'error': {
                    'message': error_message or "Unknown error",
                    'timestamp': datetime.utcnow().isoformat(),
                    'handler': self.HANDLER_NAME
                }
            }

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get handler capabilities and supported parameters.

        Returns:
            Dictionary describing handler capabilities
        """
        return {
            'modality': self.SUPPORTED_MODALITY,
            'model_type': 'ltx-video-2b',
            'max_duration': 32.0,  # seconds
            'max_frames': 257,
            'resolution_support': {
                'min_resolution': 256,
                'max_resolution': 1280,
                'constraint': 'divisible_by_32'
            },
            'frame_rates': [8, 12, 16, 24],
            'inference_time_target': '<45s',
            'memory_footprint': '6-10GB'
        }

    def supports_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Check if this handler supports the given request.

        Args:
            request_data: Request parameters to check

        Returns:
            True if supported, False otherwise
        """
        # Must have prompt (text-to-video)
        if 'prompt' not in request_data:
            return False

        # Must NOT have image (that would be image-to-video)
        if 'image' in request_data or 'input_image' in request_data:
            return False

        return True

    def ensure_model_loaded(self) -> float:
        """
        Ensure the LTX-Video model is loaded and ready.

        Returns:
            Model loading time in milliseconds

        Raises:
            ModelLoadError: If model loading fails
        """
        if self.model.is_loaded:
            logger.debug("LTX-Video model already loaded")
            return 0.0

        start_time = time.perf_counter()

        try:
            logger.info("Loading LTX-Video model...")
            self.model.load_model()

            load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"LTX-Video model loaded in {load_time_ms:.1f}ms")
            return load_time_ms

        except Exception as e:
            logger.error(f"Failed to load LTX-Video model: {e}")
            raise ModelLoadError("ltx-video-model", f"Failed to load LTX-Video model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information and statistics."""
        base_info = self.model.get_model_info()

        # Add handler-specific statistics
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0.0
        )

        success_rate = (
            self.successful_requests / self.request_count
            if self.request_count > 0 else 0.0
        )

        handler_stats = {
            'handler_type': 'LTXVideoHandler',
            'supported_modality': self.supported_modality,
            'request_count': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(success_rate, 3),
            'average_processing_time': round(avg_processing_time, 3),
            'total_processing_time': round(self.total_processing_time, 3)
        }

        return {**base_info, **handler_stats}