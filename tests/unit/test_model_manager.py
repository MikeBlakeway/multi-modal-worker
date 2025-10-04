"""
Unit tests for ModelManager class.

Tests LRU eviction logic, thread safety, memory management,
error handling, and concurrent access patterns.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Import the modules to test
from src.models.model_manager import ModelManager
from src.models.base_model import BaseModel, ModelMetadata
from src.utils.exceptions import ModelLoadError, ModelNotFoundError, MemoryError
from src.utils.config import ModelConfig


class MockModel(BaseModel):
    """Mock model for testing purposes."""

    def __init__(self, model_name: str, model_path: Path, priority: int = 50,
                 memory_usage: int = 100, load_time: float = 0.1, **kwargs):
        super().__init__(model_name, model_path, priority)
        self._memory_usage = memory_usage
        self._load_time = load_time
        self._should_fail_load = kwargs.get('should_fail_load', False)

    def load(self) -> None:
        if self._should_fail_load:
            raise ModelLoadError(self.model_name, "Mock load failure")

        time.sleep(self._load_time)  # Simulate load time
        self.is_loaded = True
        from datetime import datetime
        self.load_time = datetime.now()
        self.memory_usage_mb = self._memory_usage
        self._model = Mock()  # Mock model object

    def unload(self) -> None:
        self.is_loaded = False
        self.memory_usage_mb = 0
        self._model = None

    def infer(self, inputs):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return {"result": f"inference from {self.model_name}"}

    def get_memory_usage(self) -> int:
        return self.memory_usage_mb

    def validate_inputs(self, inputs) -> bool:
        return True


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test models
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock model files
        (self.temp_dir / "model1").mkdir()
        (self.temp_dir / "model2").mkdir()
        (self.temp_dir / "model3").mkdir()

        # Create a fresh ModelManager for each test
        with patch('src.models.model_manager.config') as mock_config:
            mock_config.max_models_in_memory = 2
            mock_config.model_timeout_seconds = 30
            mock_config.protect_recently_used_minutes = 1
            mock_config.max_concurrent_loads = 2

            self.manager = ModelManager()

    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown manager
        if hasattr(self, 'manager'):
            self.manager.shutdown()

        # Remove temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_model_registration(self):
        """Test model registration functionality."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="test_model",
            model_class=MockModel,
            model_path=model_path,
            priority=75,
            estimated_memory_mb=200
        )

        registered = self.manager.get_registered_models()
        self.assertIn("test_model", registered)

    def test_model_loading_and_caching(self):
        """Test model loading and caching behavior."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="cached_model",
            model_class=MockModel,
            model_path=model_path
        )

        # First access should load model
        model1 = self.manager.get_model("cached_model")
        self.assertTrue(model1.is_loaded)

        # Second access should return cached model
        model2 = self.manager.get_model("cached_model")
        self.assertIs(model1, model2)

        # Check statistics
        loaded_models = self.manager.get_loaded_models()
        self.assertIn("cached_model", loaded_models)

    def test_lru_eviction(self):
        """Test LRU eviction when model limit is reached."""
        # Register 3 models with limit of 2
        for i in range(3):
            model_path = self.temp_dir / f"model{i+1}"
            self.manager.register_model(
                model_name=f"model_{i}",
                model_class=MockModel,
                model_path=model_path
            )

        # Load first two models
        model_0 = self.manager.get_model("model_0")
        model_1 = self.manager.get_model("model_1")

        # Both should be loaded
        self.assertTrue(model_0.is_loaded)
        self.assertTrue(model_1.is_loaded)
        self.assertEqual(len(self.manager.get_loaded_models()), 2)

        # Load third model - should evict oldest (model_0)
        model_2 = self.manager.get_model("model_2")

        # model_0 should be evicted, model_1 and model_2 should remain
        self.assertFalse(model_0.is_loaded)
        self.assertTrue(model_1.is_loaded)
        self.assertTrue(model_2.is_loaded)
        self.assertEqual(len(self.manager.get_loaded_models()), 2)

    def test_priority_based_eviction(self):
        """Test that higher priority models are protected from eviction."""
        # Register models with different priorities
        high_priority_path = self.temp_dir / "model1"
        low_priority_path = self.temp_dir / "model2"

        self.manager.register_model(
            model_name="high_priority",
            model_class=MockModel,
            model_path=high_priority_path,
            priority=90
        )

        self.manager.register_model(
            model_name="low_priority",
            model_class=MockModel,
            model_path=low_priority_path,
            priority=10
        )

        # Load both models
        high_model = self.manager.get_model("high_priority")
        low_model = self.manager.get_model("low_priority")

        # Wait to ensure different timestamps
        time.sleep(0.1)

        # Touch high priority model to make it recently used
        high_model.mark_used()

        # Register and load third model
        third_path = self.temp_dir / "model3"
        self.manager.register_model(
            model_name="third_model",
            model_class=MockModel,
            model_path=third_path
        )

        third_model = self.manager.get_model("third_model")

        # Low priority model should be evicted first
        self.assertTrue(high_model.is_loaded)
        self.assertFalse(low_model.is_loaded)
        self.assertTrue(third_model.is_loaded)

    def test_concurrent_model_loading(self):
        """Test thread safety during concurrent model loading."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="concurrent_model",
            model_class=MockModel,
            model_path=model_path,
            load_time=0.5  # Longer load time
        )

        results = []
        errors = []

        def load_model():
            try:
                model = self.manager.get_model("concurrent_model")
                results.append(model)
            except Exception as e:
                errors.append(e)

        # Start multiple threads loading the same model
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=load_model)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors and all results should be the same model instance
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)

        # All results should be the same model instance
        first_model = results[0]
        for model in results:
            self.assertIs(model, first_model)

    def test_model_load_failure_handling(self):
        """Test error handling when model loading fails."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="failing_model",
            model_class=MockModel,
            model_path=model_path,
            should_fail_load=True
        )

        # Loading should raise ModelLoadError
        with self.assertRaises(ModelLoadError):
            self.manager.get_model("failing_model")

        # Model should not be in loaded models
        loaded_models = self.manager.get_loaded_models()
        self.assertNotIn("failing_model", loaded_models)

    def test_manual_eviction(self):
        """Test manual model eviction."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="manual_evict",
            model_class=MockModel,
            model_path=model_path
        )

        # Load model
        model = self.manager.get_model("manual_evict")
        self.assertTrue(model.is_loaded)

        # Manually evict
        result = self.manager.evict_model("manual_evict")
        self.assertTrue(result)
        self.assertFalse(model.is_loaded)

        # Evicting non-existent model should return False
        result = self.manager.evict_model("non_existent")
        self.assertFalse(result)

    def test_clear_all_models(self):
        """Test clearing all models."""
        # Load multiple models
        for i in range(2):
            model_path = self.temp_dir / f"model{i+1}"
            self.manager.register_model(
                model_name=f"clear_test_{i}",
                model_class=MockModel,
                model_path=model_path
            )
            self.manager.get_model(f"clear_test_{i}")

        # Should have 2 loaded models
        self.assertEqual(len(self.manager.get_loaded_models()), 2)

        # Clear all
        evicted = self.manager.clear_all_models()
        self.assertEqual(len(evicted), 2)
        self.assertEqual(len(self.manager.get_loaded_models()), 0)

    def test_model_status_and_monitoring(self):
        """Test model status reporting and monitoring features."""
        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="status_test",
            model_class=MockModel,
            model_path=model_path
        )

        # Check status before loading
        status = self.manager.get_model_status("status_test")
        self.assertIsNone(status)

        # Load model and check status
        model = self.manager.get_model("status_test")
        status = self.manager.get_model_status("status_test")

        self.assertIsNotNone(status)
        self.assertEqual(status["model_name"], "status_test")
        self.assertTrue(status["is_loaded"])

        # Check manager status
        manager_status = self.manager.get_manager_status()
        self.assertIn("loaded_models", manager_status)
        self.assertIn("statistics", manager_status)
        self.assertEqual(manager_status["loaded_count"], 1)

    @patch('src.models.model_manager.memory_monitor')
    def test_memory_pressure_handling(self, mock_memory_monitor):
        """Test handling of memory pressure events."""
        # Set up mock memory monitor
        mock_memory_monitor.can_load_model.return_value = False
        mock_memory_monitor.estimate_available_memory_mb.return_value = 50

        model_path = self.temp_dir / "model1"

        self.manager.register_model(
            model_name="memory_test",
            model_class=MockModel,
            model_path=model_path,
            estimated_memory_mb=100  # More than available
        )

        # Should raise MemoryError due to insufficient memory
        with self.assertRaises(MemoryError):
            self.manager.get_model("memory_test")


if __name__ == '__main__':
    unittest.main()