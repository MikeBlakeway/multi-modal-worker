"""
Integration test for module imports.

Validates that all modules can be imported correctly and the Python
package structure is working as expected.
"""

import unittest
import sys
from pathlib import Path


class TestModuleImports(unittest.TestCase):
    """Test cases for validating module import functionality."""

    def setUp(self):
        """Set up test fixtures and Python path."""
        # Add src directory to Python path for testing
        worker_root = Path(__file__).parent.parent.parent
        src_dir = worker_root / "src"

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    def test_main_module_import(self):
        """Test that main module can be imported."""
        try:
            import main
            self.assertIsNotNone(main, "Main module should be importable")
        except ImportError as e:
            self.fail(f"Failed to import main module: {e}")

    def test_main_handler_class_exists(self):
        """Test that MultiModalHandler class exists and is importable."""
        try:
            from main import MultiModalHandler
            self.assertIsNotNone(
                MultiModalHandler,
                "MultiModalHandler class should be importable"
            )
        except ImportError as e:
            self.fail(f"Failed to import MultiModalHandler: {e}")

    def test_modality_type_enum_exists(self):
        """Test that ModalityType enum exists and is importable."""
        try:
            from main import ModalityType
            self.assertIsNotNone(
                ModalityType,
                "ModalityType enum should be importable"
            )
        except ImportError as e:
            self.fail(f"Failed to import ModalityType: {e}")

    def test_runpod_handler_function_exists(self):
        """Test that runpod_handler function exists and is importable."""
        try:
            from main import runpod_handler
            self.assertIsNotNone(
                runpod_handler,
                "runpod_handler function should be importable"
            )
        except ImportError as e:
            self.fail(f"Failed to import runpod_handler: {e}")

    def test_handlers_package_import(self):
        """Test that handlers package can be imported."""
        try:
            import handlers
            self.assertIsNotNone(handlers, "Handlers package should be importable")
        except ImportError as e:
            self.fail(f"Failed to import handlers package: {e}")

    def test_models_package_import(self):
        """Test that models package can be imported."""
        try:
            import models
            self.assertIsNotNone(models, "Models package should be importable")
        except ImportError as e:
            self.fail(f"Failed to import models package: {e}")

    def test_utils_package_import(self):
        """Test that utils package can be imported."""
        try:
            import utils
            self.assertIsNotNone(utils, "Utils package should be importable")
        except ImportError as e:
            self.fail(f"Failed to import utils package: {e}")

    def test_handler_class_instantiation(self):
        """Test that MultiModalHandler can be instantiated."""
        try:
            from main import MultiModalHandler

            # Test with default parameters
            handler = MultiModalHandler()
            self.assertIsNotNone(handler, "Handler should be instantiable")

            # Test with custom cache directory
            custom_handler = MultiModalHandler(model_cache_dir="/tmp/test")
            self.assertIsNotNone(
                custom_handler,
                "Handler should be instantiable with custom parameters"
            )

        except Exception as e:
            self.fail(f"Failed to instantiate MultiModalHandler: {e}")

    def test_modality_type_values(self):
        """Test that ModalityType enum has expected values."""
        try:
            from main import ModalityType

            expected_modalities = {
                "TEXT_TO_IMAGE",
                "IMAGE_TO_VIDEO",
                "TEXT_TO_VIDEO",
                "CONTROL_NET",
                "INPAINTING",
                "CAMERA_CONTROL"
            }

            actual_modalities = {member.name for member in ModalityType}

            self.assertEqual(
                expected_modalities,
                actual_modalities,
                f"ModalityType should have expected values: {expected_modalities}"
            )

        except Exception as e:
            self.fail(f"Failed to validate ModalityType values: {e}")

    def test_handler_method_exists(self):
        """Test that handler method exists on MultiModalHandler."""
        try:
            from main import MultiModalHandler

            handler = MultiModalHandler()

            self.assertTrue(
                hasattr(handler, 'handler'),
                "MultiModalHandler should have handler method"
            )

            self.assertTrue(
                callable(getattr(handler, 'handler')),
                "handler method should be callable"
            )

        except Exception as e:
            self.fail(f"Failed to validate handler method: {e}")

    def test_basic_handler_functionality(self):
        """Test basic handler functionality with test input."""
        try:
            from main import MultiModalHandler

            handler = MultiModalHandler()

            # Test with valid modality that's not implemented yet
            # Should return error since no modality implementations exist yet
            test_event = {
                "id": "test-request",
                "input": {
                    "modality": "text-to-image",
                    "prompt": "test prompt"
                }
            }

            result = handler.handler(test_event)

            self.assertIsInstance(result, dict, "Handler should return a dict")
            # Since no modalities are implemented yet, should get an error
            self.assertIn("error", result, "Result should contain error field for unimplemented modality")
            self.assertIn("supported_modalities", result, "Result should list supported modalities")

            # Test with invalid modality
            invalid_event = {
                "id": "test-invalid",
                "input": {
                    "modality": "invalid-modality"
                }
            }

            invalid_result = handler.handler(invalid_event)

            self.assertIsInstance(
                invalid_result, dict,
                "Handler should return dict for invalid input"
            )
            self.assertIn(
                "error", invalid_result,
                "Invalid input should return error field"
            )

        except Exception as e:
            self.fail(f"Failed to test basic handler functionality: {e}")


if __name__ == "__main__":
    unittest.main()