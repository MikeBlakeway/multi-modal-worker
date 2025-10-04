"""
Test configuration and setup for MMI-004 routing infrastructure tests.

This file configures pytest and provides common test utilities.
"""

import pytest
import sys
import os
from unittest.mock import Mock

# Add source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
pytest_plugins = []

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Ensure clean test environment
    import logging
    logging.getLogger().handlers.clear()

    # Set test configuration
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'

    yield

    # Cleanup after tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']

@pytest.fixture
def mock_model_manager():
    """Fixture providing a mock ModelManager."""
    from models.model_manager import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_manager_status.return_value = {
        "loaded_models": [],
        "loaded_count": 0,
        "registered_count": 3,
        "max_models": 5,
        "memory_summary": {
            "stats": {
                "gpu_free_mb": 16000,
                "gpu_total_mb": 24000,
                "gpu_utilization": 30.0
            },
            "thresholds": {
                "warning_percent": 80,
                "eviction_percent": 90,
                "warning_exceeded": False,
                "eviction_needed": False
            },
            "available_memory_mb": 16000
        },
        "statistics": {},
        "configuration": {
            "max_models": 5,
            "model_timeout_seconds": 300,
            "protect_duration_minutes": 5
        }
    }
    mock_manager._loaded_models = {}

    return mock_manager

@pytest.fixture
def sample_text_to_image_request():
    """Fixture providing a sample text-to-image request."""
    return {
        "prompt": "A beautiful sunset over mountains with purple clouds",
        "steps": 4,
        "guidance_scale": 1.0,
        "width": 1024,
        "height": 1024
    }

@pytest.fixture
def sample_image_to_video_request():
    """Fixture providing a sample image-to-video request."""
    return {
        "image_url": "https://example.com/test-image.jpg",
        "duration": 4,
        "fps": 24
    }

@pytest.fixture
def invalid_request():
    """Fixture providing an invalid request for error testing."""
    return {
        "steps": 100,  # Invalid: out of range
        "guidance_scale": 1.0
        # Missing required prompt
    }

# ============================================================================
# Enhanced Test Fixtures (to reduce boilerplate)
# ============================================================================

@pytest.fixture
def sample_request_data():
    """Enhanced request data for testing."""
    return {
        "prompt": "a beautiful landscape",
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "seed": 42
    }


@pytest.fixture
def concrete_handler():
    """Create a concrete handler for testing."""
    # Import here to avoid circular imports
    import sys
    import os
    from unittest.mock import Mock

    # Create a minimal concrete handler that matches BaseHandler interface
    class TestConcreteHandler:
        def __init__(self):
            self.handler_name = "test-handler"
            self._request_count = 0
            self._total_processing_time = 0.0

        @property
        def supported_modality(self):
            return "test-modality"

        @property
        def required_parameters(self):
            return ["prompt"]

        @property
        def optional_parameters(self):
            return {"width": 512, "height": 512}

        def validate_request(self, request_data):
            return request_data

        def get_required_models(self, request_data):
            return ["test-model"]

        def process_inference(self, models, request_data):
            return {"result": "test-output"}

        def format_response(self, inference_results, request_data):
            return {"status": "success", "output": inference_results}

    return TestConcreteHandler()


# ============================================================================
# Performance Test Helpers
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0

    return Timer


# ============================================================================
# Parametrization Helpers (reduce test duplication)
# ============================================================================

# Common test cases for modality detection
MODALITY_TEST_CASES = [
    ("text-to-image", {"prompt": "test"}, True),
    ("image-to-video", {"input_image": "base64", "motion_prompt": "test"}, True),
    ("controlnet", {"prompt": "test", "control_image": "base64"}, True),
    ("invalid-modality", {}, False),
]

IMAGE_DIMENSIONS = [
    (512, 512),
    (768, 768),
    (1024, 1024),
]

# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names/paths."""

    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # Mark slow tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)

        # Mark model tests
        if "model" in item.nodeid and "load" in item.nodeid:
            item.add_marker(pytest.mark.model)


# Test markers
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")