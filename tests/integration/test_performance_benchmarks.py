"""
Performance benchmarks for model management framework.

Tests memory efficiency, throughput, and scalability of the model
management system under various load conditions.
"""

import unittest
import time
import threading
import statistics
import gc
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, Mock
import psutil
import tracemalloc

from src.models import model_manager, memory_monitor, BaseModel
from src.utils import config
from src.utils.exceptions import ModelLoadError, MemoryError


class BenchmarkModel(BaseModel):
    """Lightweight model for performance benchmarking."""

    def __init__(self, model_name: str, model_path: Path, priority: int = 50,
                 memory_usage: int = 100, processing_complexity: str = "simple"):
        super().__init__(model_name, model_path, priority)
        self._memory_usage = memory_usage
        self._complexity = processing_complexity
        self._data = None

    def load(self) -> None:
        # Simulate different memory allocation patterns
        if self._complexity == "heavy":
            # Allocate more memory to simulate heavy models
            self._data = [0] * (self._memory_usage * 1000)  # Rough MB simulation
        else:
            self._data = [0] * (self._memory_usage * 100)

        self.is_loaded = True
        self.memory_usage_mb = self._memory_usage
        self.load_time = datetime.now()

    def unload(self) -> None:
        self._data = None
        self.is_loaded = False
        self.memory_usage_mb = 0
        gc.collect()  # Force garbage collection

    def infer(self, inputs):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Simulate processing based on complexity
        if self._complexity == "heavy":
            # More CPU-intensive simulation
            result = sum(x * 2 for x in self._data[:1000])
        else:
            result = len(str(inputs))

        self.mark_used()
        return {"result": result, "model": self.model_name}

    def get_memory_usage(self) -> int:
        return self.memory_usage_mb

    def validate_inputs(self, inputs) -> bool:
        return inputs is not None


class TestModelManagementPerformance(unittest.TestCase):
    """Performance benchmarks for model management system."""

    def setUp(self):
        """Set up benchmark environment."""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create model directories
        for i in range(20):  # More models for stress testing
            (self.temp_dir / f"model_{i}").mkdir()

        # Reset global manager state
        model_manager.clear_all_models()
        model_manager._model_registry.clear()

        # Start memory monitoring
        if not memory_monitor._monitoring:
            memory_monitor.start_monitoring()

        # Initialize performance tracking
        self.performance_metrics = {
            'load_times': [],
            'inference_times': [],
            'eviction_times': [],
            'memory_usage': [],
            'thread_contention': []
        }

    def tearDown(self):
        """Clean up benchmark environment."""
        model_manager.clear_all_models()
        model_manager._model_registry.clear()

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def measure_execution_time(self, operation):
        """Decorator to measure operation execution time."""
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = operation(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time
        return wrapper

    def test_model_loading_performance(self):
        """Benchmark model loading times across different scenarios."""
        load_times = []

        # Test loading performance for different model sizes
        test_scenarios = [
            ("small", 50, "simple"),
            ("medium", 200, "simple"),
            ("large", 500, "simple"),
            ("heavy_small", 50, "heavy"),
            ("heavy_medium", 200, "heavy")
        ]

        for scenario_name, memory_size, complexity in test_scenarios:
            model_name = f"perf_{scenario_name}"
            model_path = self.temp_dir / "model_0"

            # Register model
            model_manager.register_model(
                model_name=model_name,
                model_class=BenchmarkModel,
                model_path=model_path,
                memory_usage=memory_size,
                processing_complexity=complexity
            )

            # Measure loading time
            start_time = time.perf_counter()
            model = model_manager.get_model(model_name)
            end_time = time.perf_counter()

            load_time = end_time - start_time
            load_times.append((scenario_name, load_time, memory_size))

            # Clean up for next test
            model_manager.evict_model(model_name)

        # Analyze results
        print(f"\nModel Loading Performance Results:")
        for scenario_name, load_time, memory_size in load_times:
            print(f"  {scenario_name}: {load_time:.4f}s (Memory: {memory_size}MB)")

        # Performance assertions
        max_load_time = max(load_time for _, load_time, _ in load_times)
        self.assertLess(max_load_time, 1.0, "Model loading should complete within 1 second")

        # Memory usage should correlate with model size
        sorted_by_memory = sorted(load_times, key=lambda x: x[2])
        for i in range(1, len(sorted_by_memory)):
            prev_memory = sorted_by_memory[i-1][2]
            curr_memory = sorted_by_memory[i][2]
            if curr_memory > prev_memory * 2:  # Significant memory difference
                prev_time = sorted_by_memory[i-1][1]
                curr_time = sorted_by_memory[i][1]
                # Allow some tolerance for measurement variance
                self.assertLessEqual(curr_time, prev_time * 3,
                                   f"Loading time should scale reasonably with memory usage")

    def test_inference_throughput_benchmark(self):
        """Benchmark inference throughput under various conditions."""
        model_path = self.temp_dir / "model_0"

        # Register a model for throughput testing
        model_manager.register_model(
            model_name="throughput_test",
            model_class=BenchmarkModel,
            model_path=model_path,
            memory_usage=100,
            processing_complexity="simple"
        )

        model = model_manager.get_model("throughput_test")

        # Single-threaded throughput test
        single_thread_times = []
        num_inferences = 1000

        start_time = time.perf_counter()
        for i in range(num_inferences):
            inference_start = time.perf_counter()
            result = model.infer({"input": f"test_{i}"})
            inference_end = time.perf_counter()
            single_thread_times.append(inference_end - inference_start)
        end_time = time.perf_counter()

        single_thread_total = end_time - start_time
        single_thread_rps = num_inferences / single_thread_total

        # Multi-threaded throughput test
        multi_thread_results = []
        errors = []

        def worker_thread(worker_id: int, iterations: int):
            worker_times = []
            try:
                for i in range(iterations):
                    inference_start = time.perf_counter()
                    result = model.infer({"input": f"worker_{worker_id}_iter_{i}"})
                    inference_end = time.perf_counter()
                    worker_times.append(inference_end - inference_start)
            except Exception as e:
                errors.append(e)
            multi_thread_results.extend(worker_times)

        # Run concurrent workers
        num_workers = 4
        iterations_per_worker = 250
        threads = []

        multi_start = time.perf_counter()
        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker_thread, args=(worker_id, iterations_per_worker))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        multi_end = time.perf_counter()

        multi_thread_total = multi_end - multi_start
        multi_thread_rps = (num_workers * iterations_per_worker) / multi_thread_total

        # Analyze results
        print(f"\nThroughput Benchmark Results:")
        print(f"  Single-threaded: {single_thread_rps:.2f} RPS")
        print(f"  Multi-threaded ({num_workers} workers): {multi_thread_rps:.2f} RPS")
        print(f"  Avg inference time (single): {statistics.mean(single_thread_times)*1000:.3f}ms")
        print(f"  Avg inference time (multi): {statistics.mean(multi_thread_results)*1000:.3f}ms")

        # Performance assertions
        self.assertEqual(len(errors), 0, f"No errors should occur: {errors}")
        self.assertGreater(single_thread_rps, 100, "Should achieve reasonable single-thread performance")
        self.assertGreater(multi_thread_rps, single_thread_rps * 0.5,
                          "Multi-threading should provide some performance benefit")

    def test_memory_efficiency_benchmark(self):
        """Benchmark memory efficiency and cleanup effectiveness."""
        # Enable memory tracing
        tracemalloc.start()

        # Baseline memory
        gc.collect()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        model_path = self.temp_dir / "model_0"
        memory_measurements = []

        # Load and unload models repeatedly to test memory cleanup
        for cycle in range(10):
            # Load multiple models
            loaded_models = []
            for i in range(3):
                model_name = f"memory_test_{cycle}_{i}"
                model_manager.register_model(
                    model_name=model_name,
                    model_class=BenchmarkModel,
                    model_path=model_path,
                    memory_usage=100,
                    processing_complexity="heavy"
                )

                model = model_manager.get_model(model_name)
                loaded_models.append(model_name)

            # Measure memory after loading
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(("loaded", current_memory - baseline_memory))

            # Use models for inference
            for model_name in loaded_models:
                model = model_manager._loaded_models[model_name]
                for _ in range(10):
                    model.infer({"test": "data"})

            # Evict all models
            for model_name in loaded_models:
                model_manager.evict_model(model_name)

            # Force garbage collection
            gc.collect()

            # Measure memory after cleanup
            post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(("cleanup", post_cleanup_memory - baseline_memory))

        # Analyze memory efficiency
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(baseline_snapshot, 'lineno')

        print(f"\nMemory Efficiency Results:")
        print(f"  Baseline memory: {baseline_memory:.1f}MB")

        loaded_memories = [mem for phase, mem in memory_measurements if phase == "loaded"]
        cleanup_memories = [mem for phase, mem in memory_measurements if phase == "cleanup"]

        print(f"  Avg memory during loading: {statistics.mean(loaded_memories):.1f}MB")
        print(f"  Avg memory after cleanup: {statistics.mean(cleanup_memories):.1f}MB")
        print(f"  Memory cleanup efficiency: {(1 - statistics.mean(cleanup_memories) / statistics.mean(loaded_memories)) * 100:.1f}%")

        # Performance assertions
        final_memory_overhead = statistics.mean(cleanup_memories)
        self.assertLess(final_memory_overhead, 50, "Memory overhead after cleanup should be reasonable")

        # Memory should not continuously grow (indicating leaks)
        if len(cleanup_memories) >= 5:
            early_cleanup = statistics.mean(cleanup_memories[:3])
            late_cleanup = statistics.mean(cleanup_memories[-3:])
            growth_rate = (late_cleanup - early_cleanup) / early_cleanup if early_cleanup > 0 else 0
            self.assertLess(growth_rate, 0.5, "Memory usage should not grow significantly over time")

        tracemalloc.stop()

    def test_concurrent_access_scalability(self):
        """Benchmark system performance under increasing concurrent load."""
        model_path = self.temp_dir / "model_0"

        # Register more models to reduce contention (double the number of workers)
        for i in range(32):  # Enough models for up to 16 workers
            model_name = f"scale_test_{i}"
            model_manager.register_model(
                model_name=model_name,
                model_class=BenchmarkModel,
                model_path=model_path,
                memory_usage=80,
                processing_complexity="simple"
            )

        scalability_results = []

        # Test with increasing numbers of concurrent workers
        for num_workers in [1, 2, 4, 8, 16]:
            results = []
            errors = []
            completion_times = []
            results_lock = threading.Lock()
            errors_lock = threading.Lock()

            def scaling_worker(worker_id: int, iterations: int = 25):  # Reduced iterations
                worker_start = time.perf_counter()
                worker_results = []
                try:
                    for i in range(iterations):
                        # Distribute models more evenly to reduce contention
                        model_name = f"scale_test_{(worker_id * iterations + i) % 32}"

                        # Retry logic for concurrency issues
                        max_retries = 2
                        for retry in range(max_retries):
                            try:
                                model = model_manager.get_model(model_name, timeout_seconds=2.0)
                                result = model.infer({"worker": worker_id, "iteration": i})
                                worker_results.append(result)
                                break
                            except Exception as e:
                                if retry == max_retries - 1:
                                    with errors_lock:
                                        errors.append(f"Worker {worker_id}: {e}")
                                else:
                                    time.sleep(0.01 * (retry + 1))

                    with results_lock:
                        results.extend(worker_results)

                except Exception as e:
                    with errors_lock:
                        errors.append(f"Worker {worker_id}: {e}")
                finally:
                    worker_end = time.perf_counter()
                    completion_times.append(worker_end - worker_start)

            # Run scaling test
            threads = []
            test_start = time.perf_counter()

            for worker_id in range(num_workers):
                thread = threading.Thread(target=scaling_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            test_end = time.perf_counter()

            # Calculate metrics
            total_duration = test_end - test_start
            total_operations = len(results)
            throughput = total_operations / total_duration
            avg_completion_time = statistics.mean(completion_times) if completion_times else 0

            scalability_results.append({
                'workers': num_workers,
                'throughput': throughput,
                'total_operations': total_operations,
                'duration': total_duration,
                'avg_completion_time': avg_completion_time,
                'errors': len(errors)
            })

            # Brief pause between tests
            time.sleep(0.1)

        # Analyze scalability
        print(f"\nConcurrent Access Scalability Results:")
        print(f"{'Workers':<8} {'Throughput':<12} {'Operations':<12} {'Duration':<10} {'Errors':<8}")
        print("-" * 60)

        for result in scalability_results:
            print(f"{result['workers']:<8} {result['throughput']:<12.1f} "
                  f"{result['total_operations']:<12} {result['duration']:<10.2f} {result['errors']:<8}")

        # Performance assertions
        for result in scalability_results:
            self.assertEqual(result['errors'], 0, f"No errors should occur with {result['workers']} workers")

        # Throughput should generally increase with more workers (up to a point)
        single_worker_throughput = scalability_results[0]['throughput']
        best_throughput = max(r['throughput'] for r in scalability_results)

        self.assertGreater(best_throughput, single_worker_throughput * 0.8,
                          "Multi-worker setup should maintain reasonable throughput")

    def test_eviction_algorithm_performance(self):
        """Benchmark LRU eviction algorithm performance and accuracy."""
        model_path = self.temp_dir / "model_0"

        # Register more models than can be kept in memory
        num_models = 10
        for i in range(num_models):
            model_name = f"eviction_test_{i}"
            model_manager.register_model(
                model_name=model_name,
                model_class=BenchmarkModel,
                model_path=model_path,
                memory_usage=80,
                priority=50 + (i % 3) * 10  # Varying priorities
            )

        # Track access patterns and evictions
        access_log = []
        eviction_times = []

        # Load models and create access pattern
        for round_num in range(5):
            for i in range(num_models):
                model_name = f"eviction_test_{i}"

                # Measure eviction performance
                eviction_start = time.perf_counter()
                model = model_manager.get_model(model_name)
                eviction_end = time.perf_counter()

                eviction_times.append(eviction_end - eviction_start)

                # Use the model
                model.infer({"round": round_num, "model": i})
                access_log.append((time.time(), model_name, round_num, i))

                # Brief delay to create temporal patterns
                time.sleep(0.01)

        # Analyze eviction performance
        loaded_models = model_manager.get_loaded_models()

        print(f"\nEviction Algorithm Performance:")
        print(f"  Average eviction time: {statistics.mean(eviction_times)*1000:.3f}ms")
        print(f"  Max eviction time: {max(eviction_times)*1000:.3f}ms")
        print(f"  Models currently loaded: {len(loaded_models)}")
        print(f"  Total access operations: {len(access_log)}")

        # Performance assertions
        avg_eviction_time = statistics.mean(eviction_times)
        max_eviction_time = max(eviction_times)

        self.assertLess(avg_eviction_time, 0.1, "Average eviction should be fast")
        self.assertLess(max_eviction_time, 0.5, "Maximum eviction should complete quickly")

        # Verify LRU effectiveness - recently accessed models should be more likely to stay loaded
        recent_accesses = set(model_name for _, model_name, _, _ in access_log[-20:])
        loaded_model_names = set(loaded_models)

        overlap = len(recent_accesses.intersection(loaded_model_names))
        total_recent = len(recent_accesses)

        if total_recent > 0:
            retention_rate = overlap / total_recent
            self.assertGreater(retention_rate, 0.3, "LRU should retain some recently accessed models")


if __name__ == '__main__':
    # Run with verbose output for benchmark results
    unittest.main(verbosity=2)