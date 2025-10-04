"""
Multi-Modal Inference Worker

A RunPod serverless worker that provides multi-modal AI inference capabilities
including text-to-image, image-to-video, text-to-video, inpainting,
ControlNet guidance, and camera control functionality.
"""

__version__ = "0.1.0"
__author__ = "Media Labs Team"

# Package exports
from .main import MultiModalHandler

__all__ = [
    "MultiModalHandler",
]