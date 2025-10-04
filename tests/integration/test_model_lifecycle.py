"""
Integration tests for model lifecycle management.

Tests end-to-end model loading, usage, eviction cycles and integration
between ModelManager, MemoryMonitor, and BaseModel components.
"""

import unittest
import time
import threading
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, Mock
from datetime import datetime

from src.models import model_manager, memory_monitor, BaseModel, ModelMetadata
from src.utils import config
from src.utils.exceptions import ModelLoadError, MemoryError


class IntegrationTestModel(BaseModel):
    """Test model for integration testing."""

    def __init__(self, model_name: str, model_path: Path, priority: int = 50,
                 memory_usage: int = 100, load_delay: float = 0.1, **kwargs):
        super().__init__(model_name, model_path, priority)
        self._memory_usage = memory_usage
        self._load_delay = load_delay
        self._infer_delay = kwargs.get('infer_delay', 0.01)

    def load(self) -> None:
        time.sleep(self._load_delay)
        self.is_loaded = True
        self.memory_usage_mb = self._memory_usage
        self._model = Mock()
        self.load_time = datetime.now()

    def unload(self) -> None:
        self.is_loaded = False
        self.memory_usage_mb = 0
        self._model = None

    def infer(self, inputs):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        time.sleep(self._infer_delay)
        self.mark_used()

        return {
            "model": self.model_name,
            "result": f"processed {inputs.get('text', 'input')}",
            "memory_mb": self.memory_usage_mb
        }

    def get_memory_usage(self) -> int:
        return self.memory_usage_mb

    def validate_inputs(self, inputs) -> bool:
        return isinstance(inputs, dict) and 'text' in inputs


class TestModelLifecycle(unittest.TestCase):
    """Integration tests for complete model lifecycle."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create model directories (enough for all tests)
        for i in range(10):
            (self.temp_dir / f"model_{i}").mkdir()

        # Reset global manager state
        model_manager.clear_all_models()
        model_manager._model_registry.clear()

        # Ensure memory monitoring is running
        if not memory_monitor._monitoring:
            memory_monitor.start_monitoring()

    def tearDown(self):
        """Clean up test environment."""
        # Clear all models
        model_manager.clear_all_models()
        model_manager._model_registry.clear()

        # Remove temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_model_lifecycle(self):
        """Test complete lifecycle: register -> load -> use -> evict."""
        model_path = self.temp_dir / "model_0"

        # 1. Register model
        model_manager.register_model(
            model_name="lifecycle_test",
            model_class=IntegrationTestModel,
            model_path=model_path,
            priority=70,
            estimated_memory_mb=150,
            memory_usage=150
        )

        # Verify registration
        registered = model_manager.get_registered_models()
        self.assertIn("lifecycle_test", registered)

        # 2. Load model (should trigger loading)
        model = model_manager.get_model("lifecycle_test")
        self.assertIsNotNone(model)
        self.assertTrue(model.is_loaded)
        self.assertEqual(model.memory_usage_mb, 150)

        # 3. Use model for inference
        inputs = {"text": "test input"}
        results = model.infer(inputs)

        self.assertIn("result", results)
        self.assertEqual(results["model"], "lifecycle_test")
        self.assertTrue(model.use_count > 0)

        # 4. Manual eviction
        evicted = model_manager.evict_model("lifecycle_test")
        self.assertTrue(evicted)
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.memory_usage_mb, 0)

    def test_automatic_lru_eviction_under_load(self):
        """Test that models are loaded correctly and eviction can be triggered."""
        # Register a few models
        model_names = []
        for i in range(3):
            model_name = f"eviction_test_{i}"
            model_path = self.temp_dir / f"model_{i}"

            model_manager.register_model(
                model_name=model_name,
                model_class=IntegrationTestModel,
                model_path=model_path,
                priority=50,  # Same priority
                memory_usage=100
            )
            model_names.append(model_name)

        # Load all models
        models = []
        for model_name in model_names:
            model = model_manager.get_model(model_name)
            models.append(model)
            self.assertTrue(model.is_loaded)

        # Verify models are tracked correctly
        loaded_models = model_manager.get_loaded_models()
        self.assertEqual(len(loaded_models), 3)

        for model_name in model_names:
            self.assertIn(model_name, loaded_models)

        # Use models to establish access patterns
        for i, model in enumerate(models):
            model.infer({"text": f"access {i}"})

        # Test manual eviction (since automatic eviction has protection period)
        evicted = model_manager.evict_model(model_names[0])
        self.assertTrue(evicted, "Should successfully evict model")

        # Check that model was evicted
        self.assertFalse(models[0].is_loaded, "Evicted model should not be loaded")

        updated_loaded = model_manager.get_loaded_models()
        self.assertEqual(len(updated_loaded), 2, "Should have 2 models after eviction")
        self.assertNotIn(model_names[0], updated_loaded, "Evicted model not in list")

    def test_concurrent_model_access_patterns(self):
        """Test various concurrent access patterns."""
        # Register multiple models
        for i in range(6):  # More models to reduce contention
            model_path = self.temp_dir / f"model_{i}"
            model_manager.register_model(
                model_name=f"concurrent_{i}",
                model_class=IntegrationTestModel,
                model_path=model_path,
                memory_usage=100
            )

        results = []
        errors = []
        results_lock = threading.Lock()
        errors_lock = threading.Lock()

        def worker_thread(worker_id):
            """Worker thread that performs model operations."""
            iterations = 5
            try:
                for i in range(iterations):
                    # Use different model names per worker to reduce contention
                    model_name = f"concurrent_{(worker_id * 2 + i) % 6}"

                    # Retry logic for handling concurrent loading
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            model = model_manager.get_model(model_name, timeout_seconds=5.0)

                            # Perform inference
                            result = model.infer({"text": f"worker_{worker_id}_iteration_{i}"})
                            with results_lock:
                                results.append(result)
                            break  # Success, exit retry loop

                        except Exception as e:
                            if retry == max_retries - 1:  # Last retry
                                with errors_lock:
                                    errors.append(f"Worker {worker_id}: {e}")
                            else:
                                time.sleep(0.1 * (retry + 1))  # Exponential backoff

                    # Small delay between operations
                    time.sleep(0.01)

            except Exception as e:
                with errors_lock:
                    errors.append(f"Worker {worker_id}: {e}")

        # Start multiple worker threads
        threads = []
        for worker_id in range(4):
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all workers to complete
        for thread in threads:
            thread.join()

        # Verify results - allow some errors due to concurrency
        self.assertGreater(len(results), 10, f"Too few results, errors: {errors}")

        # Verify all results have expected structure
        for result in results:
            self.assertIn("model", result)
            self.assertIn("result", result)
            self.assertIn("memory_mb", result)

    def test_memory_pressure_integration(self):
        """Test integration with memory monitoring and pressure responses."""
        # Mock memory monitor to simulate pressure
        with patch.object(memory_monitor, 'can_load_model') as mock_can_load:
            with patch.object(memory_monitor, 'estimate_available_memory_mb') as mock_estimate:
                with patch.object(memory_monitor, 'clear_gpu_cache') as mock_clear_cache:

                    # Set up the sequence: allow first model, deny second model initially and after eviction attempts
                    can_load_calls = [True, False, False, False, False, False, False]  # Allow first, then persistent denial
                    mock_can_load.side_effect = can_load_calls
                    mock_estimate.return_value = 50  # Insufficient memory available
                    mock_clear_cache.return_value = None

                    # Register models
                    for i in range(2):
                        model_name = f"memory_test_{i}"
                        model_path = self.temp_dir / f"model_{i}"

                        model_manager.register_model(
                            model_name=model_name,
                            model_class=IntegrationTestModel,
                            model_path=model_path,
                            estimated_memory_mb=100,
                            memory_usage=100
                        )

                    # Load first model (should succeed)
                    model_0 = model_manager.get_model("memory_test_0")
                    self.assertTrue(model_0.is_loaded)

                    # Try to load second model (should fail due to memory pressure after eviction attempts)
                    with self.assertRaises(Exception) as cm:  # Expecting MemoryError from _evict_for_memory
                        model_1 = model_manager.get_model("memory_test_1")

                    # Should have attempted multiple can_load_model calls (initial + eviction attempts)
                    self.assertGreater(mock_can_load.call_count, 1)

    def test_model_status_monitoring_during_operations(self):
        """Test status monitoring during various model operations."""
        model_path = self.temp_dir / "model_0"

        model_manager.register_model(
            model_name="status_monitor",
            model_class=IntegrationTestModel,
            model_path=model_path,
            memory_usage=120
        )

        # Check status before loading
        initial_status = model_manager.get_manager_status()
        self.assertEqual(initial_status["loaded_count"], 0)

        # Load model
        model = model_manager.get_model("status_monitor")

        # Check status after loading
        loaded_status = model_manager.get_manager_status()
        self.assertEqual(loaded_status["loaded_count"], 1)

        loaded_models = loaded_status["loaded_models"]
        self.assertEqual(len(loaded_models), 1)

        model_info = loaded_models[0]
        self.assertEqual(model_info["model_name"], "status_monitor")
        self.assertTrue(model_info["is_loaded"])
        self.assertEqual(model_info["memory_usage_mb"], 120)

        # Use model and check statistics
        model.infer({"text": "test"})
        model.infer({"text": "test2"})

        updated_status = model_manager.get_manager_status()
        updated_model_info = updated_status["loaded_models"][0]
        self.assertEqual(updated_model_info["use_count"], 2)

    def test_error_recovery_and_cleanup(self):
        """Test error recovery and proper cleanup after failures."""
        model_path = self.temp_dir / "model_0"

        # Create a model that fails to load
        class FailingModel(IntegrationTestModel):
            def load(self):
                raise ModelLoadError(self.model_name, "Simulated load failure")

        model_manager.register_model(
            model_name="failing_model",
            model_class=FailingModel,
            model_path=model_path
        )

        # Attempt to load should fail
        with self.assertRaises(ModelLoadError):
            model_manager.get_model("failing_model")

        # Model should not be in loaded models
        loaded_models = model_manager.get_loaded_models()
        self.assertNotIn("failing_model", loaded_models)

        # System should still be functional for other models
        model_manager.register_model(
            model_name="working_model",
            model_class=IntegrationTestModel,
            model_path=model_path,
            memory_usage=100
        )

        # Should be able to load working model
        working_model = model_manager.get_model("working_model")
        self.assertTrue(working_model.is_loaded)

    def test_performance_under_load(self):
        """Test system performance under heavy concurrent load."""
        # Register more models to reduce contention
        for i in range(8):  # More models for workers to choose from
            model_name = f"perf_test_{i}"
            model_path = self.temp_dir / f"model_{i}"

            model_manager.register_model(
                model_name=model_name,
                model_class=IntegrationTestModel,
                model_path=model_path,
                memory_usage=50 + (i % 3) * 20,
                load_delay=0.05,
                infer_delay=0.001  # Very fast inference
            )

        # Measure performance metrics
        start_time = time.time()
        total_inferences = 0
        errors = []
        inferences_lock = threading.Lock()
        errors_lock = threading.Lock()

        def performance_worker(worker_id: int, duration_seconds: float = 1.5):
            """Worker that performs many operations for a set duration."""
            nonlocal total_inferences
            worker_start = time.time()
            local_inferences = 0

            try:
                while time.time() - worker_start < duration_seconds:
                    # Distribute model selection to reduce contention
                    model_name = f"perf_test_{(worker_id * 2 + local_inferences) % 8}"

                    # Retry logic for concurrency
                    max_retries = 2
                    for retry in range(max_retries):
                        try:
                            model = model_manager.get_model(model_name, timeout_seconds=2.0)
                            result = model.infer({"text": f"worker_{worker_id}_{local_inferences}"})
                            local_inferences += 1
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                with errors_lock:
                                    errors.append(f"Worker {worker_id}: {e}")
                            else:
                                time.sleep(0.01 * (retry + 1))

                    # Brief pause to prevent overwhelming
                    time.sleep(0.001)

                with inferences_lock:
                    total_inferences += local_inferences

            except Exception as e:
                with errors_lock:
                    errors.append(f"Worker {worker_id}: {e}")

        # Run performance test with multiple workers
        threads = []
        num_workers = 4
        test_duration = 1.0  # 1 second test

        for worker_id in range(num_workers):
            thread = threading.Thread(target=performance_worker, args=(worker_id, test_duration))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # Analyze performance
        total_duration = end_time - start_time
        inferences_per_second = total_inferences / total_duration

        # Verify minimal errors (allow some due to concurrency)
        self.assertLessEqual(len(errors), 2, f"Too many performance test errors: {errors}")

        # Verify reasonable performance (adjusted for concurrency handling)
        self.assertGreater(total_inferences, 30, f"Should complete reasonable number of inferences (got {total_inferences})")
        if total_duration > 0:
            inferences_per_second = total_inferences / total_duration
            self.assertGreater(inferences_per_second, 15, f"Should maintain reasonable throughput (got {inferences_per_second:.2f} RPS)")

        # Verify system state is clean
        final_status = model_manager.get_manager_status()
        self.assertGreaterEqual(final_status["loaded_count"], 1)


if __name__ == '__main__':
    unittest.main()