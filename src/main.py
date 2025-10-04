"""
Multi-Modal Inference Worker Entry Point

RunPod serverless worker that provides multi-modal AI inference capabilities.
Supports text-to-image, image-to-video, text-to-video, inpainting,
ControlNet guidance, and camera control functionality.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional


class ModalityType(Enum):
    """Enumeration of supported modality types."""
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_VIDEO = "image-to-video"
    TEXT_TO_VIDEO = "text-to-video"
    CONTROL_NET = "controlnet"
    INPAINTING = "inpainting"
    CAMERA_CONTROL = "camera-control"

try:
    # Try relative imports first (when running as package)
    from .models.model_manager import ModelManager
    from .handlers.multi_modal_handler import MultiModalHandler
    from .utils.logging_config import LoggingConfig, get_system_logger
    from .utils.response_formatter import ResponseFormatter
    from .utils import config, UnsupportedModalityError, ValidationError, InferenceError
except ImportError:
    # Fall back to absolute imports (when running as script or in tests)
    from src.models.model_manager import ModelManager
    from src.handlers.multi_modal_handler import MultiModalHandler
    from src.utils.logging_config import LoggingConfig, get_system_logger
    from src.utils.response_formatter import ResponseFormatter
    from src.utils import config, UnsupportedModalityError, ValidationError, InferenceError


# Global instances
_model_manager = None
_multi_modal_handler = None
_response_formatter = ResponseFormatter()


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_multi_modal_handler() -> MultiModalHandler:
    """Get or create the global multi-modal handler instance."""
    global _multi_modal_handler
    if _multi_modal_handler is None:
        model_manager = get_model_manager()
        _multi_modal_handler = MultiModalHandler(model_manager)
    return _multi_modal_handler

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.

    Args:
        event: RunPod event containing input parameters

    Returns:
        Dict containing inference results or error information
    """
    try:
        # Initialize logging
        debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        LoggingConfig.setup_logging(debug_mode)
        logger = get_system_logger()

        logger.info(f"Processing request: {event.get('id', 'unknown')}")

        # Extract input data
        input_data = event.get("input", {})

        # Handle health check requests
        if input_data.get("health_check"):
            handler = get_multi_modal_handler()
            response = handler.health_check()
            return _response_formatter.add_runpod_compatibility(response)

        # Handle system status requests
        if input_data.get("system_status"):
            handler = get_multi_modal_handler()
            status = handler.get_system_status()
            response = _response_formatter.format_system_status_response(
                system_stats=status,
                supported_modalities=handler.get_supported_modalities(),
                request_id=f"status-{event.get('id', 'unknown')}"
            )
            return _response_formatter.add_runpod_compatibility(response)

        # Process regular inference request through routing system
        handler = get_multi_modal_handler()
        response = handler.process_request(input_data)

        # Convert to RunPod-compatible format
        return _response_formatter.add_runpod_compatibility(response)

    except Exception as e:
        # Fallback error handling without logger if logging setup failed
        return {
            "error": f"Critical handler error: {str(e)}",
            "status": "error"
        }


def runpod_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod entry point function.

    This function is called by RunPod for each inference request.
    """
    return handler(event)


if __name__ == "__main__":
    # For local testing
    test_event = {
        "id": "test-request",
        "input": {
            "health_check": True
        }
    }

    print("Testing health check...")
    result = runpod_handler(test_event)
    print(json.dumps(result, indent=2))

    print("\nTesting inference request...")
    test_event = {
        "id": "test-inference",
        "input": {
            "prompt": "A beautiful sunset over mountains",
            "steps": 4,
            "guidance_scale": 1.0
        }
    }

    result = runpod_handler(test_event)
    print(json.dumps(result, indent=2))