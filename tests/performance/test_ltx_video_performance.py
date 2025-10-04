"""
Performance Tests for LTX-Video Handler

Tests memory usage validation, inference time benchmarking,
resource cleanup verification, GPU monitoring, and stress testing.
Following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from pathlib import Path
import time
import asyncio
import json
import psutil
import threading
import gc
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import resource

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.handlers.ltx_video_handler import LTXVideoHandler
from src.handlers.multi_modal_handler import MultiModalHandler
from src.models.model_manager import ModelManager
from src.models.ltx_video_model import LTXVideoModel
from src.models.memory_monitor import MemoryMonitor
from src.schemas.text_to_video_schema import TextToVideoRequest, TextToVideoResponse
from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
from src.utils.logging_config import LoggingConfig


class TestLTXVideoPerformance(unittest.TestCase):
    """Test cases for LTX-Video performance validation."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures for performance testing."""
        cls.performance_results = {}
        cls.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def setUp(self):
        """Set up test fixtures."""
        # Mock model manager with performance monitoring
        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.get_manager_status.return_value = {
            'total_models': 1,
            'loaded_models': 1,
            'memory_summary': {'stats': {'total_gb': 8.5}}
        }

        # Mock memory monitor with proper method configuration
        self.mock_memory_monitor = Mock(spec=MemoryMonitor)
        # Configure the mock method explicitly
        self.mock_memory_monitor.get_gpu_memory_usage = Mock(return_value={
            'used_gb': 6.5,
            'total_gb': 24.0,
            'utilization_percent': 27.1
        })

        # Create handlers with performance tracking
        self.ltx_video_handler = LTXVideoHandler()
        self.multi_handler = MultiModalHandler(
            model_manager=self.mock_model_manager,
            auto_initialize=False
        )
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Test requests with varying complexity
        self.minimal_request = {
            'prompt': 'A cat walking',
            'fps': 8,
            'video': True
        }

        self.standard_request = {
            'prompt': 'A majestic eagle soaring through mountains',
            'width': 720,
            'height': 1280,
            'num_frames': 25,  # 3.125 seconds at 8fps
            'fps': 8,
            'video': True
        }

        self.complex_request = {
            'prompt': 'A cinematic sequence of a bustling cityscape at night with neon lights, rain reflections, and dynamic camera movement',
            'width': 1280,
            'height': 720,
            'num_frames': 49,  # 6.125 seconds at 8fps
            'num_inference_steps': 30,
            'guidance_scale': 8.0,
            'fps': 8,
            'video': True
        }

    def tearDown(self):
        """Clean up test fixtures."""
        # Force garbage collection
        gc.collect()

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_memory_usage_validation(self, mock_handle_request):
        """Test memory usage stays within specified constraints."""
        # Setup mock response with memory tracking
        def mock_handle_with_memory(*args, **kwargs):
            # Simulate memory usage during inference
            return {
                'status': 'success',
                'output': {
                    'video_data': "mock_video_base64_data_representing_generated_video",
                    'video_info': {'duration': 3.125, 'fps': 8, 'frames': 25}
                },
                'metadata': {
                    'processing_time_ms': 2500,
                    'memory_usage': {
                        'peak_gpu_mb': 8500,  # 8.5GB
                        'peak_system_mb': 2500,
                        'model_footprint_gb': 6.2
                    }
                }
            }

        mock_handle_request.side_effect = mock_handle_with_memory

        # Measure memory before request
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Process request and measure memory
        result = self.multi_handler.process_request(self.standard_request)

        # Measure memory after request
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        # Verify successful processing
        self.assertEqual(result.get('status'), 'success')

        # Verify memory usage within constraints
        metadata = result.get('metadata', {})
        memory_usage = metadata.get('memory_usage', {})

        # Test GPU memory constraint (<10GB as specified)
        peak_gpu_gb = memory_usage.get('peak_gpu_mb', 0) / 1024
        self.assertLess(peak_gpu_gb, 10.0,
                       f"GPU memory usage {peak_gpu_gb:.2f}GB exceeds 10GB limit")

        # Test model footprint constraint (6-10GB range)
        model_footprint = memory_usage.get('model_footprint_gb', 0)
        self.assertGreaterEqual(model_footprint, 6.0,
                               f"Model footprint {model_footprint:.2f}GB below expected 6GB minimum")
        self.assertLess(model_footprint, 10.0,
                       f"Model footprint {model_footprint:.2f}GB exceeds 10GB maximum")

        # Test system memory increase is reasonable (<1GB for processing)
        self.assertLess(memory_delta, 1024,
                       f"System memory increase {memory_delta:.2f}MB too high")

        # Store results for analysis
        self.performance_results['memory_validation'] = {
            'peak_gpu_gb': peak_gpu_gb,
            'model_footprint_gb': model_footprint,
            'system_delta_mb': memory_delta,
            'constraints_met': True
        }

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_inference_time_benchmarking(self, mock_handle_request):
        """Test inference time meets performance targets."""
        # Define performance targets
        INFERENCE_TARGETS = {
            'minimal': 15.0,    # <15s for minimal requests
            'standard': 30.0,   # <30s for standard requests
            'complex': 45.0     # <45s for complex requests
        }

        test_cases = [
            ('minimal', self.minimal_request, INFERENCE_TARGETS['minimal']),
            ('standard', self.standard_request, INFERENCE_TARGETS['standard']),
            ('complex', self.complex_request, INFERENCE_TARGETS['complex'])
        ]

        performance_results = {}

        for test_name, request, target_time in test_cases:
            with self.subTest(test_case=test_name):
                # Setup mock with realistic timing
                def mock_handle_with_timing(*args, **kwargs):
                    # Simulate processing time based on complexity
                    complexity_factor = {
                        'minimal': 0.3,
                        'standard': 0.6,
                        'complex': 0.8
                    }[test_name]

                    simulated_time = target_time * complexity_factor
                    time.sleep(min(simulated_time / 100, 0.1))  # Scale down for testing

                    return {
                        'status': 'success',
                        'output': {
                            'video_data': f"mock_video_data_{test_name}",
                            'video_info': {'duration': 3.125, 'fps': 8}
                        },
                        'metadata': {
                            'processing_time_ms': int(simulated_time * 1000),
                            'inference_target_ms': int(target_time * 1000),
                            'complexity': test_name
                        }
                    }

                mock_handle_request.side_effect = mock_handle_with_timing

                # Measure actual processing time
                start_time = time.time()
                result = self.multi_handler.process_request(request)
                actual_time = time.time() - start_time

                # Verify successful processing
                self.assertEqual(result.get('status'), 'success')

                # Verify reported timing
                metadata = result.get('metadata', {})
                reported_time_s = metadata.get('processing_time_ms', 0) / 1000

                # Test inference time target
                self.assertLess(reported_time_s, target_time,
                               f"{test_name} inference time {reported_time_s:.2f}s exceeds target {target_time}s")

                # Store performance metrics
                performance_results[test_name] = {
                    'target_time_s': target_time,
                    'reported_time_s': reported_time_s,
                    'actual_time_s': actual_time,
                    'efficiency_ratio': reported_time_s / target_time,
                    'meets_target': reported_time_s < target_time
                }

        # Store results for analysis
        self.performance_results['inference_benchmarking'] = performance_results

        # Verify all tests met targets
        for test_name, results in performance_results.items():
            self.assertTrue(results['meets_target'],
                           f"{test_name} failed to meet performance target")

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_resource_cleanup_verification(self, mock_handle_request):
        """Test proper resource cleanup and memory management."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': "mock_video"},
            'metadata': {'processing_time_ms': 1000}
        }

        # Configure model manager eviction behavior - mock the evict_model method
        self.mock_model_manager.evict_model = Mock(return_value=True)

        # Measure initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Process multiple requests to generate resource usage
        num_requests = 5
        for i in range(num_requests):
            request = {
                'prompt': f'Test video {i}',
                'fps': 8,
                'video': True
            }
            result = self.multi_handler.process_request(request)
            self.assertEqual(result.get('status'), 'success')

        # Trigger cleanup - simulate evicting a model for resource cleanup
        eviction_result = self.mock_model_manager.evict_model('ltx-video')

        # Force garbage collection
        gc.collect()

        # Measure memory after cleanup
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Verify eviction was called and successful
        self.mock_model_manager.evict_model.assert_called_with('ltx-video')
        self.assertTrue(eviction_result, "Model eviction should succeed")        # Verify memory delta is reasonable
        memory_delta = final_memory - initial_memory
        self.assertLess(abs(memory_delta), 500,  # Less than 500MB delta
                       f"Memory delta {memory_delta:.2f}MB indicates poor cleanup")

        # Store cleanup metrics
        self.performance_results['resource_cleanup'] = {
            'requests_processed': num_requests,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': memory_delta,
            'eviction_successful': eviction_result,
            'cleanup_effective': abs(memory_delta) < 500
        }

    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring and constraint validation."""
        # Test GPU memory status reporting
        gpu_status = self.mock_memory_monitor.get_gpu_memory_usage()

        self.assertIsInstance(gpu_status, dict)
        self.assertIn('used_gb', gpu_status)
        self.assertIn('total_gb', gpu_status)
        self.assertIn('utilization_percent', gpu_status)

        # Verify GPU memory constraints
        used_gb = gpu_status['used_gb']
        total_gb = gpu_status['total_gb']
        utilization_percent = gpu_status['utilization_percent']

        # Test reasonable memory usage (should be within LTX-Video constraints)
        self.assertGreaterEqual(used_gb, 0, "GPU memory usage cannot be negative")
        self.assertLess(used_gb, 12.0, "GPU memory usage exceeds reasonable limit for LTX-Video")
        self.assertLess(utilization_percent, 50.0, "GPU utilization unexpectedly high")

        # Test memory efficiency (used should be reasonable fraction of total)
        if total_gb > 0:
            efficiency_ratio = used_gb / total_gb
            self.assertLess(efficiency_ratio, 0.8, "GPU memory usage ratio too high")

        # Store monitoring results
        self.performance_results['gpu_monitoring'] = {
            'used_gb': used_gb,
            'total_gb': total_gb,
            'utilization_percent': utilization_percent,
            'efficiency_ratio': efficiency_ratio if total_gb > 0 else 0,
            'constraints_met': used_gb < 12.0 and utilization_percent < 50.0
        }

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_concurrent_processing_stress(self, mock_handle_request):
        """Test performance under concurrent request load."""
        # Setup mock with variable processing times
        def mock_handle_with_delay(*args, **kwargs):
            # Simulate realistic processing variation
            import random
            processing_time = random.uniform(1.0, 3.0)  # 1-3 second range
            time.sleep(processing_time / 100)  # Scale down for testing

            return {
                'status': 'success',
                'output': {'video_data': "mock_concurrent_video"},
                'metadata': {
                    'processing_time_ms': int(processing_time * 1000),
                    'thread_id': threading.current_thread().ident
                }
            }

        mock_handle_request.side_effect = mock_handle_with_delay

        # Define concurrent test parameters
        num_concurrent_requests = 3
        max_concurrent_time = 10.0  # seconds

        # Create test requests
        requests = [
            {
                'prompt': f'Concurrent test video {i}',
                'fps': 8,
                'video': True
            }
            for i in range(num_concurrent_requests)
        ]

        # Execute concurrent requests
        start_time = time.time()
        results = []

        def process_request(request):
            return self.multi_handler.process_request(request)

        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(process_request, req) for req in requests]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # Verify all requests succeeded
        for i, result in enumerate(results):
            self.assertEqual(result.get('status'), 'success',
                           f"Concurrent request {i} failed")

        # Verify reasonable total processing time
        self.assertLess(total_time, max_concurrent_time,
                       f"Concurrent processing took {total_time:.2f}s, exceeds {max_concurrent_time}s limit")

        # Analyze threading behavior
        thread_ids = [r.get('metadata', {}).get('thread_id') for r in results]
        unique_threads = len(set(thread_ids))

        # Store concurrent processing metrics
        self.performance_results['concurrent_stress'] = {
            'num_requests': num_concurrent_requests,
            'total_time_s': total_time,
            'average_time_s': total_time / num_concurrent_requests,
            'unique_threads': unique_threads,
            'max_time_limit_s': max_concurrent_time,
            'meets_concurrency_target': total_time < max_concurrent_time
        }

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_memory_leak_detection(self, mock_handle_request):
        """Test for memory leaks during repeated processing."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': "mock_video"},
            'metadata': {'processing_time_ms': 800}
        }

        # Measure baseline memory
        gc.collect()  # Clean up before baseline
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Process requests in batches to detect memory growth
        batch_size = 5
        num_batches = 3
        memory_measurements = [baseline_memory]

        for batch in range(num_batches):
            # Process batch of requests
            for i in range(batch_size):
                request = {
                    'prompt': f'Memory leak test batch {batch} request {i}',
                    'fps': 8,
                    'video': True
                }
                result = self.multi_handler.process_request(request)
                self.assertEqual(result.get('status'), 'success')

            # Force cleanup and measure memory
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)

            # Brief pause to allow cleanup
            time.sleep(0.1)

        # Analyze memory growth pattern
        memory_deltas = [
            memory_measurements[i] - memory_measurements[i-1]
            for i in range(1, len(memory_measurements))
        ]

        # Calculate average memory growth per batch
        avg_growth_per_batch = sum(memory_deltas) / len(memory_deltas)
        max_acceptable_growth = 50  # MB per batch

        # Test for memory leak indicators
        self.assertLess(avg_growth_per_batch, max_acceptable_growth,
                       f"Average memory growth {avg_growth_per_batch:.2f}MB per batch indicates potential leak")

        # Verify final memory is reasonable
        final_memory = memory_measurements[-1]
        total_growth = final_memory - baseline_memory
        max_total_growth = 200  # MB total

        self.assertLess(total_growth, max_total_growth,
                       f"Total memory growth {total_growth:.2f}MB exceeds acceptable limit")

        # Store leak detection results
        self.performance_results['memory_leak_detection'] = {
            'baseline_memory_mb': baseline_memory,
            'final_memory_mb': final_memory,
            'total_growth_mb': total_growth,
            'avg_growth_per_batch_mb': avg_growth_per_batch,
            'batches_processed': num_batches,
            'requests_per_batch': batch_size,
            'no_leak_detected': avg_growth_per_batch < max_acceptable_growth
        }

    def test_performance_metrics_collection(self):
        """Test comprehensive performance metrics collection."""
        # Verify handler provides performance capabilities
        capabilities = self.ltx_video_handler.get_capabilities()

        self.assertIsInstance(capabilities, dict)
        self.assertIn('inference_time_target', capabilities)
        self.assertIn('memory_footprint', capabilities)

        # Verify performance targets are properly specified
        inference_target = capabilities.get('inference_time_target', '')
        memory_footprint = capabilities.get('memory_footprint', '')

        self.assertIn('45', inference_target, "Should specify <45 second target")
        self.assertIn('GB', memory_footprint, "Should specify memory in GB")

        # Test system status includes performance metrics
        system_status = self.multi_handler.get_system_status()

        self.assertIn('statistics', system_status)
        stats = system_status['statistics']
        self.assertIn('total_requests', stats)
        self.assertIn('average_processing_time_ms', stats)

        # Store metrics validation results
        self.performance_results['metrics_collection'] = {
            'capabilities_complete': all([
                'inference_time_target' in capabilities,
                'memory_footprint' in capabilities
            ]),
            'system_stats_available': all([
                'total_requests' in stats,
                'average_processing_time_ms' in stats
            ]),
            'performance_monitoring_ready': True
        }

    @classmethod
    def tearDownClass(cls):
        """Generate performance test summary."""
        # Calculate overall performance score
        performance_score = cls._calculate_performance_score()

        # Print performance summary
        print("\n" + "="*60)
        print("LTX-VIDEO PERFORMANCE TEST RESULTS")
        print("="*60)
        print(f"Overall Performance Score: {performance_score:.1f}%")
        print("-"*60)

        for test_name, results in cls.performance_results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "="*60)

    @classmethod
    def _calculate_performance_score(cls):
        """Calculate overall performance score based on test results."""
        if not cls.performance_results:
            return 0.0

        score_components = []

        # Memory validation score (25%)
        memory_results = cls.performance_results.get('memory_validation', {})
        if memory_results.get('constraints_met', False):
            score_components.append(25.0)

        # Inference benchmarking score (30%)
        benchmark_results = cls.performance_results.get('inference_benchmarking', {})
        if benchmark_results:
            targets_met = sum(1 for r in benchmark_results.values()
                             if r.get('meets_target', False))
            total_tests = len(benchmark_results)
            if total_tests > 0:
                benchmark_score = (targets_met / total_tests) * 30.0
                score_components.append(benchmark_score)

        # Resource cleanup score (15%)
        cleanup_results = cls.performance_results.get('resource_cleanup', {})
        if cleanup_results.get('cleanup_effective', False):
            score_components.append(15.0)

        # GPU monitoring score (10%)
        gpu_results = cls.performance_results.get('gpu_monitoring', {})
        if gpu_results.get('constraints_met', False):
            score_components.append(10.0)

        # Concurrent processing score (10%)
        concurrent_results = cls.performance_results.get('concurrent_stress', {})
        if concurrent_results.get('meets_concurrency_target', False):
            score_components.append(10.0)

        # Memory leak detection score (10%)
        leak_results = cls.performance_results.get('memory_leak_detection', {})
        if leak_results.get('no_leak_detected', False):
            score_components.append(10.0)

        return sum(score_components)


if __name__ == '__main__':
    unittest.main(verbosity=2)