"""
AnimateDiff Handler

Implements the AnimateDiff image-to-video generation handler with motion adaptation.
Provides comprehensive parameter validation, model management integration,
and standardized response formatting for video generation workflows.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time
import uuid

try:
    from .base_handler import BaseHandler
    from ..models.animatediff_model import AnimateDiffModel
    from ..schemas.image_to_video_schema import (
        ImageToVideoRequest, ImageToVideoResponse, VideoInfo,
        validate_image_to_video_request, create_success_response,
        create_error_response
    )
    from ..utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from ..utils.response_formatter import ResponseFormatter, ErrorType
except ImportError:
    from src.handlers.base_handler import BaseHandler
    from src.models.animatediff_model import AnimateDiffModel
    from src.schemas.image_to_video_schema import (
        ImageToVideoRequest, ImageToVideoResponse, VideoInfo,
        validate_image_to_video_request, create_success_response,
        create_error_response
    )
    from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
    from src.utils.response_formatter import ResponseFormatter, ErrorType

logger = logging.getLogger(__name__)


class AnimateDiffHandler(BaseHandler):
    """
    Handler for AnimateDiff image-to-video generation.

    Provides comprehensive AnimateDiff-based image-to-video generation capabilities
    with motion adaptation and temporal consistency for creating smooth animations.
    """

    def __init__(self,
                 motion_adapter_id: str = None,
                 base_model_id: str = None,
                 auto_load: bool = False):
        """
        Initialize AnimateDiff handler.

        Args:
            motion_adapter_id: Custom motion adapter model ID
            base_model_id: Custom base diffusion model ID
            auto_load: Whether to automatically load model on initialization
        """
        super().__init__("animatediff-image-to-video")

        # Store constructor parameters (different names to avoid property conflicts)
        self._motion_adapter_id = motion_adapter_id
        self._base_model_id = base_model_id

        # Initialize model
        self.model = AnimateDiffModel(
            motion_adapter_id=self._motion_adapter_id,
            base_model_id=self._base_model_id
        )

        # Response formatter for standardized outputs
        self.response_formatter = ResponseFormatter()

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.successful_requests = 0
        self.failed_requests = 0

        # Auto-load model if requested
        if auto_load:
            self.ensure_model_loaded()

        logger.info(f"AnimateDiffHandler initialized with motion adapter: {self.model.motion_adapter_id}")

    @property
    def supported_modality(self) -> str:
        """Return the modality this handler supports."""
        return "image-to-video"

    @property
    def model_id(self) -> str:
        """Return the base model ID for compatibility."""
        return self.model.base_model_id

    @property
    def motion_adapter_id(self) -> str:
        """Return the motion adapter ID for compatibility."""
        return self.model.motion_adapter_id

    @property
    def model_info(self) -> Dict[str, Any]:
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
            'handler_type': 'AnimateDiffHandler',
            'supported_modality': self.supported_modality,
            'request_count': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(success_rate, 3),
            'average_processing_time': round(avg_processing_time, 3),
            'total_processing_time': round(self.total_processing_time, 3)
        }

        return {**base_info, **handler_stats}

    def ensure_model_loaded(self) -> float:
        """
        Ensure the AnimateDiff model is loaded and ready.

        Returns:
            Model loading time in milliseconds

        Raises:
            ModelLoadError: If model loading fails
        """
        if self.model.is_loaded:
            logger.debug("AnimateDiff model already loaded")
            return 0.0

        start_time = time.perf_counter()

        try:
            logger.info("Loading AnimateDiff model...")
            self.model.load_model()

            load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"AnimateDiff model loaded successfully in {load_time_ms:.1f}ms")

            return load_time_ms

        except Exception as e:
            logger.error(f"Failed to load AnimateDiff model: {e}")
            raise ModelLoadError("animatediff-model", f"Failed to load AnimateDiff model: {e}")

    def process_request(self, request_data: Dict[str, Any]) -> ImageToVideoResponse:
        """
        Process an image-to-video generation request.

        Args:
            request_data: Raw request data dictionary

        Returns:
            ImageToVideoResponse with generation results or error information
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            # Track request
            self.request_count += 1
            logger.info(f"Processing image-to-video request {request_id}")

            # Validate request
            try:
                request = validate_image_to_video_request(request_data)
                logger.debug(f"Request validation successful for {request_id}")
            except ValidationError as e:
                logger.warning(f"Request validation failed for {request_id}: {e}")
                self.failed_requests += 1
                return create_error_response(
                    error_message=f"Invalid request: {str(e)}",
                    error_code="VALIDATION_ERROR",
                    request_id=request_id
                )

            # Ensure model is loaded
            try:
                model_load_time_ms = self.ensure_model_loaded()
            except ModelLoadError as e:
                logger.error(f"Model loading failed for {request_id}: {e}")
                self.failed_requests += 1
                return create_error_response(
                    error_message=f"Model loading failed: {str(e)}",
                    error_code="MODEL_LOAD_ERROR",
                    request_id=request_id
                )

            # Validate video parameters
            try:
                self._validate_video_parameters(request)
            except ValidationError as e:
                logger.warning(f"Video parameter validation failed for {request_id}: {e}")
                self.failed_requests += 1
                return create_error_response(
                    error_message=f"Invalid video parameters: {str(e)}",
                    error_code="PARAMETER_ERROR",
                    request_id=request_id
                )

            # Generate video
            try:
                generation_start = time.perf_counter()

                base64_video, generation_info = self.model.generate_video(
                    input_image=request.input_image,
                    motion_prompt=request.motion_prompt,
                    num_frames=request.num_frames,
                    fps=request.fps,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    motion_strength=request.motion_strength,
                    seed=request.seed,
                    context_batch_size=request.context_batch_size,
                    enable_loop=request.enable_loop,
                    enable_smooth=request.enable_smooth
                )

                inference_time = time.perf_counter() - generation_start

                logger.info(f"Video generation completed for {request_id} in {inference_time:.2f}s")

            except InferenceError as e:
                logger.error(f"Video generation failed for {request_id}: {e}")
                self.failed_requests += 1
                return create_error_response(
                    error_message=f"Video generation failed: {str(e)}",
                    error_code="INFERENCE_ERROR",
                    request_id=request_id
                )

            # Create video info
            video_info = VideoInfo(
                **generation_info['video_info']
            )

            # Prepare generation parameters for response
            generation_params = {
                'motion_prompt': request.motion_prompt,
                'num_frames': request.num_frames,
                'fps': request.fps,
                'num_inference_steps': request.num_inference_steps,
                'guidance_scale': request.guidance_scale,
                'motion_strength': request.motion_strength,
                'seed': request.seed,
                'context_batch_size': request.context_batch_size,
                'enable_loop': request.enable_loop,
                'enable_smooth': request.enable_smooth,
                'model_info': generation_info.get('model_name', 'AnimateDiff')
            }

            # Calculate total processing time
            total_time = time.perf_counter() - start_time
            self.total_processing_time += total_time
            self.successful_requests += 1

            logger.info(f"Request {request_id} completed successfully in {total_time:.2f}s")

            # Create success response
            return create_success_response(
                video_base64=base64_video,
                video_info=video_info,
                generation_params=generation_params,
                inference_time=inference_time,
                model_load_time_ms=model_load_time_ms,
                request_id=request_id
            )

        except Exception as e:
            # Handle any unexpected errors
            logger.error(f"Unexpected error processing request {request_id}: {e}")
            self.failed_requests += 1

            total_time = time.perf_counter() - start_time
            self.total_processing_time += total_time

            return create_error_response(
                error_message=f"Internal error: {str(e)}",
                error_code="INTERNAL_ERROR",
                request_id=request_id
            )

    def _validate_video_parameters(self, request: ImageToVideoRequest) -> None:
        """
        Validate video-specific parameters for feasibility.

        Args:
            request: Validated request object

        Raises:
            ValidationError: If parameters are invalid or infeasible
        """
        # Check frame count vs inference time constraints
        estimated_time_per_frame = 1.2  # Conservative estimate in seconds
        estimated_total_time = request.num_frames * estimated_time_per_frame

        if estimated_total_time > 30:  # 25 second target + buffer
            raise ValidationError(
                f"Frame count too high: {request.num_frames} frames would take ~{estimated_total_time:.1f}s "
                f"(exceeds 25s target). Try reducing num_frames."
            )

        # Validate FPS for reasonable video duration
        min_duration = 0.5  # Minimum 0.5 second video
        max_duration = 5.0   # Maximum 5 second video

        duration = request.num_frames / request.fps
        if duration < min_duration:
            raise ValidationError(
                f"Video too short: {duration:.2f}s (minimum {min_duration}s). "
                f"Increase num_frames or decrease fps."
            )

        if duration > max_duration:
            raise ValidationError(
                f"Video too long: {duration:.2f}s (maximum {max_duration}s). "
                f"Decrease num_frames or increase fps."
            )

        # Validate motion prompt if provided
        if request.motion_prompt:
            # Check for potentially problematic prompts
            problematic_words = ['static', 'still', 'frozen', 'motionless']
            prompt_lower = request.motion_prompt.lower()

            if any(word in prompt_lower for word in problematic_words):
                logger.warning(
                    f"Motion prompt contains static-suggesting words: '{request.motion_prompt}'. "
                    f"This may result in minimal motion."
                )

        logger.debug(f"Video parameters validated: {request.num_frames} frames, {request.fps} fps, {duration:.2f}s duration")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status of the handler.

        Returns:
            Dictionary containing health information
        """
        return {
            'handler_type': 'AnimateDiffHandler',
            'modality': self.supported_modality,
            'model_loaded': self.model.is_loaded,
            'model_info': self.model_info,
            'performance_stats': {
                'request_count': self.request_count,
                'success_rate': (
                    self.successful_requests / self.request_count
                    if self.request_count > 0 else 0.0
                ),
                'average_processing_time': (
                    self.total_processing_time / self.request_count
                    if self.request_count > 0 else 0.0
                )
            },
            'capabilities': {
                'image_to_video': True,
                'motion_prompts': True,
                'seamless_loops': True,
                'frame_interpolation': True,
                'max_frames': 32,
                'max_duration_seconds': 5.0,
                'supported_formats': ['mp4', 'gif']
            }
        }

    def cleanup(self) -> None:
        """Clean up resources and unload model."""
        logger.info("Cleaning up AnimateDiff handler")

        if self.model and self.model.is_loaded:
            self.model.unload_model()

        logger.info("AnimateDiff handler cleanup completed")

    # Abstract method implementations required by BaseHandler
    @property
    def required_parameters(self) -> list[str]:
        """Return list of required parameters for image-to-video generation."""
        return ['input_image', 'prompt']

    @property
    def optional_parameters(self) -> Dict[str, Any]:
        """Return dict of optional parameters with their default values."""
        return {
            'num_frames': 16,
            'fps': 8,
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'seed': None,
            'motion_strength': 0.8,
            'output_format': 'mp4'
        }

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize request data for image-to-video generation.

        Args:
            request_data: Raw request data from client

        Returns:
            Normalized and validated request data

        Raises:
            ValidationError: If request data is invalid
        """
        try:
            # Create and validate ImageToVideoRequest schema
            request = ImageToVideoRequest(**request_data)

            # Convert back to dict for processing
            validated_data = request.model_dump()

            logger.debug(f"Request validation successful for AnimateDiff")
            return validated_data

        except Exception as e:
            logger.error(f"Request validation failed: {str(e)}")
            raise ValidationError(f"Invalid image-to-video request: {str(e)}")

    def get_required_models(self, request_data: Dict[str, Any]) -> list[str]:
        """
        Determine which models are needed for this request.

        Args:
            request_data: Validated request data

        Returns:
            List of model names required for processing
        """
        return ['animatediff-v3-adapter', 'flux-1-schnell-fp8']

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
            # Extract the AnimateDiff model - for now, use the first available model
            # In a complete implementation, this would load and use the actual AnimateDiff model

            # Simulate inference processing
            start_time = time.time()

            # Placeholder for actual AnimateDiff inference
            # This would call self.model.infer() with proper parameters
            inference_result = {
                'video_data': 'base64_encoded_video_data',  # Placeholder
                'frames_generated': request_data.get('num_frames', 16),
                'format': request_data.get('output_format', 'mp4'),
                'processing_time': time.time() - start_time,
                'model_used': 'animatediff-v3-adapter'
            }

            logger.info(f"AnimateDiff inference completed in {inference_result['processing_time']:.2f}s")
            return inference_result

        except Exception as e:
            logger.error(f"AnimateDiff inference failed: {str(e)}")
            raise InferenceError(f"Image-to-video inference failed: {str(e)}")

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
            # Create video metadata
            video_info = {
                'format': inference_results.get('format', 'mp4'),
                'num_frames': inference_results.get('frames_generated', 16),
                'fps': request_data.get('fps', 8),
                'width': request_data.get('width', 512),
                'height': request_data.get('height', 512),
                'duration': inference_results.get('frames_generated', 16) / request_data.get('fps', 8),
                'size_bytes': len(inference_results.get('video_data', '')) * 3 // 4  # Estimate from base64
            }

            # Format using response formatter
            formatted_response = self.response_formatter.format_success_response(
                output_data={
                    'video': inference_results.get('video_data'),
                    'video_info': video_info
                },
                processing_time=inference_results.get('processing_time', 0.0),
                model_used=inference_results.get('model_used', 'animatediff'),
                request_id=request_data.get('id')
            )

            logger.debug(f"Response formatted successfully for AnimateDiff")
            return formatted_response

        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            raise InferenceError(f"Failed to format response: {str(e)}")

    def __repr__(self) -> str:
        """String representation of the handler."""
        model_status = "loaded" if self.model.is_loaded else "unloaded"
        return (f"AnimateDiffHandler(modality='{self.supported_modality}', "
                f"model_status='{model_status}', requests={self.request_count})")