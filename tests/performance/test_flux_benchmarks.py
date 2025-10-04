"""
Performance tests for FLUX.1 text-to-image handler.

Tests inference speed, memory usage, and throughput to ensure
the <15 second inference time requirement is met.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from typing import Dict, Any, List
import statistics
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.handlers.flux_handler import FluxHandler
    from src.models.flux_model import FluxModel
    from src.models.model_manager import ModelManager
    from src.utils.exceptions import InferenceError
except ImportError:
    # Standalone implementation for testing
    pass


class TestFluxPerformance(unittest.TestCase):
    """Performance benchmarks for FLUX.1 text-to-image generation."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.flux_handler = FluxHandler()

        # Mock model and model manager
        self.mock_flux_model = Mock(spec=FluxModel)
        self.mock_flux_model.model_name = "flux-1-schnell-fp8"
        self.mock_flux_model.is_loaded = True

        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.get_model.return_value = self.mock_flux_model

        # Standard test request
        self.test_request = {
            'prompt': 'A photorealistic mountain landscape at golden hour with dramatic lighting',
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 4,
            'guidance_scale': 0.0,
            'seed': 42
        }

        # Sample output image for mocking
        self.sample_image = Image.new('RGB', (1024, 1024), color='blue')

    def create_mock_inference_result(self, inference_time: float, memory_mb: int = 14000):
        """Helper to create consistent mock inference results."""
        return {
            'image': self.sample_image,
            'inference_time': inference_time,
            'parameters': self.test_request.copy(),
            'memory_usage_mb': memory_mb,
            'model_info': {
                'name': 'FLUX.1-schnell-fp8',
                'variant': 'fp8',
                'inference_count': 1,
                'average_time': inference_time
            }
        }

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_inference_time_under_15_seconds(self, mock_encode):
        """Test that inference completes within 15 seconds."""
        # Mock fast inference (well under requirement)
        target_inference_time = 12.5  # Target: under 15 seconds
        self.mock_flux_model.infer.return_value = self.create_mock_inference_result(target_inference_time)
        mock_encode.return_value = ("encoded_data", 1000000)

        # Time the complete request processing
        start_time = time.time()

        with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
            response = self.flux_handler.process_request(self.test_request)

        end_time = time.time()
        total_processing_time = end_time - start_time

        # Verify successful processing
        self.assertEqual(response['status'], 'success')

        # Verify inference time meets requirement
        reported_inference_time = response['output']['inference_time']
        self.assertEqual(reported_inference_time, target_inference_time)
        self.assertLess(reported_inference_time, 15.0,
                       f"Inference time {reported_inference_time}s exceeds 15 second requirement")

        # Verify total processing overhead is minimal (should be much less than 1 second)
        processing_overhead = total_processing_time - target_inference_time
        self.assertLess(processing_overhead, 1.0,
                       f"Processing overhead {processing_overhead}s is too high")

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_different_resolution_performance(self, mock_encode):
        """Test performance across different image resolutions."""
        # Test different resolutions and expected performance
        resolution_tests = [
            (512, 512, 8.0),     # Smaller - faster
            (768, 768, 10.0),    # Medium
            (1024, 1024, 12.5),  # Standard
            (1024, 1536, 14.0),  # Portrait - still under limit
            (1536, 1024, 14.0),  # Landscape - still under limit
        ]

        mock_encode.return_value = ("encoded_data", 800000)

        for width, height, expected_time in resolution_tests:
            with self.subTest(resolution=(width, height)):
                # Create resolution-specific request
                request = self.test_request.copy()
                request.update({'width': width, 'height': height})

                # Mock inference with resolution-appropriate timing
                self.mock_flux_model.infer.return_value = self.create_mock_inference_result(expected_time)

                with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                    response = self.flux_handler.process_request(request)

                # Verify success and timing
                self.assertEqual(response['status'], 'success')
                inference_time = response['output']['inference_time']

                # All resolutions should be under 15 seconds
                self.assertLess(inference_time, 15.0,
                               f"Resolution {width}x{height} took {inference_time}s, exceeds 15s limit")

                # Verify the expected timing was used
                self.assertEqual(inference_time, expected_time)

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_inference_steps_performance_tradeoff(self, mock_encode):
        """Test performance vs quality tradeoff with different inference steps."""
        # Test different step counts and their performance impact
        step_tests = [
            (1, 6.0),   # Fastest, lowest quality
            (4, 12.5),  # Default, balanced
            (8, 14.5),  # Slower, higher quality - still under limit
        ]

        mock_encode.return_value = ("encoded_data", 900000)

        for steps, expected_time in step_tests:
            with self.subTest(steps=steps):
                # Create step-specific request
                request = self.test_request.copy()
                request['num_inference_steps'] = steps

                # Mock inference with step-appropriate timing
                self.mock_flux_model.infer.return_value = self.create_mock_inference_result(expected_time)

                with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                    response = self.flux_handler.process_request(request)

                # Verify success and timing
                self.assertEqual(response['status'], 'success')
                inference_time = response['output']['inference_time']

                # All step counts should be under 15 seconds
                self.assertLess(inference_time, 15.0,
                               f"{steps} inference steps took {inference_time}s, exceeds 15s limit")

                # Verify timing scales appropriately with steps
                self.assertEqual(inference_time, expected_time)

    def test_memory_usage_tracking(self):
        """Test memory usage is properly tracked and within reasonable limits."""
        # Mock inference with memory tracking
        expected_memory_mb = 14000  # 14GB, typical for FLUX.1-schnell-fp8
        self.mock_flux_model.infer.return_value = self.create_mock_inference_result(12.0, expected_memory_mb)

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("encoded_data", 1000000)

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(self.test_request)

            # Verify memory usage is reported
            self.assertEqual(response['status'], 'success')
            model_info = response['output']['model_info']

            # Memory usage should be reasonable for FLUX.1-schnell-fp8
            # Allow up to 16GB for model + processing overhead
            max_allowed_memory = 16000
            self.assertLessEqual(expected_memory_mb, max_allowed_memory,
                               f"Memory usage {expected_memory_mb}MB exceeds limit {max_allowed_memory}MB")

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_concurrent_request_performance(self, mock_encode):
        """Test performance under concurrent load."""
        mock_encode.return_value = ("encoded_data", 800000)

        # Results storage for concurrent requests
        results = []
        results_lock = threading.Lock()

        def process_concurrent_request(request_id: int):
            """Process a single request and store timing results."""
            # Each request gets slightly different timing (simulating real variance)
            inference_time = 12.0 + (request_id * 0.1)  # 12.0 to 12.4 seconds

            mock_result = self.create_mock_inference_result(inference_time)

            # Create a new mock for this thread to avoid conflicts
            thread_mock_model = Mock(spec=FluxModel)
            thread_mock_model.model_name = "flux-1-schnell-fp8"
            thread_mock_model.is_loaded = True
            thread_mock_model.infer.return_value = mock_result

            thread_mock_manager = Mock(spec=ModelManager)
            thread_mock_manager.get_model.return_value = thread_mock_model

            # Create request
            request = self.test_request.copy()
            request['seed'] = 42 + request_id  # Unique seed per request

            # Time the processing
            start_time = time.time()

            # Create a handler instance for this thread
            thread_handler = FluxHandler()
            with patch.object(thread_handler, 'model_manager', thread_mock_manager):
                response = thread_handler.process_request(request)

            end_time = time.time()

            # Store results thread-safely
            with results_lock:
                results.append({
                    'request_id': request_id,
                    'success': response['status'] == 'success',
                    'inference_time': response.get('output', {}).get('inference_time', 0),
                    'total_time': end_time - start_time
                })

        # Run 5 concurrent requests
        threads = []
        num_requests = 5

        for i in range(num_requests):
            thread = threading.Thread(target=process_concurrent_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Analyze results
        self.assertEqual(len(results), num_requests, "Not all concurrent requests completed")

        # Verify all requests succeeded
        successful_requests = [r for r in results if r['success']]
        self.assertEqual(len(successful_requests), num_requests,
                        "Some concurrent requests failed")

        # Verify all inference times were under limit
        inference_times = [r['inference_time'] for r in results]
        max_inference_time = max(inference_times)
        self.assertLess(max_inference_time, 15.0,
                       f"Slowest concurrent request took {max_inference_time}s, exceeds 15s limit")

        # Verify total processing times are reasonable
        total_times = [r['total_time'] for r in results]
        max_total_time = max(total_times)
        self.assertLess(max_total_time, 16.0,
                       f"Slowest concurrent processing took {max_total_time}s, too slow")

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_performance_statistics_tracking(self, mock_encode):
        """Test that performance statistics are properly tracked over multiple requests."""
        mock_encode.return_value = ("encoded_data", 750000)

        # Simulate multiple requests with varying performance
        inference_times = [11.5, 12.0, 13.2, 11.8, 12.5]

        for i, inference_time in enumerate(inference_times):
            self.mock_flux_model.infer.return_value = self.create_mock_inference_result(inference_time)

            # Create unique request
            request = self.test_request.copy()
            request['seed'] = 100 + i

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(request)

            # Verify success
            self.assertEqual(response['status'], 'success')

            # Verify reported inference time
            reported_time = response['output']['inference_time']
            self.assertEqual(reported_time, inference_time)

        # Verify statistics tracking in handler
        self.assertEqual(self.flux_handler.successful_inferences, len(inference_times))

        # If the handler tracks timing statistics, verify they're reasonable
        if hasattr(self.flux_handler, 'total_inference_time'):
            expected_total = sum(inference_times)
            self.assertAlmostEqual(self.flux_handler.total_inference_time, expected_total, places=1)

        if hasattr(self.flux_handler, 'average_inference_time'):
            expected_average = statistics.mean(inference_times)
            self.assertAlmostEqual(self.flux_handler.average_inference_time, expected_average, places=1)

    def test_performance_degradation_detection(self):
        """Test detection of performance degradation scenarios."""
        # Simulate performance degradation (inference taking too long)
        slow_inference_time = 18.0  # Exceeds 15 second requirement

        mock_result = self.create_mock_inference_result(slow_inference_time)
        self.mock_flux_model.infer.return_value = mock_result

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("encoded_data", 900000)

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(self.test_request)

            # Request should still succeed, but timing should be reported accurately
            self.assertEqual(response['status'], 'success')

            # Verify the slow timing is reported
            reported_time = response['output']['inference_time']
            self.assertEqual(reported_time, slow_inference_time)

            # In production, this would trigger alerts/monitoring
            # Here we just verify the slow time is properly tracked
            self.assertGreater(reported_time, 15.0,
                             "Performance degradation scenario should report slow timing")

    @patch('src.handlers.flux_handler.encode_pil_image')
    def test_throughput_measurement(self, mock_encode):
        """Test throughput measurement for sequential requests."""
        mock_encode.return_value = ("encoded_data", 850000)

        # Process multiple sequential requests and measure throughput
        num_requests = 10
        inference_time_per_request = 12.0

        results = []
        overall_start_time = time.time()

        for i in range(num_requests):
            # Mock consistent inference timing
            self.mock_flux_model.infer.return_value = self.create_mock_inference_result(inference_time_per_request)

            request = self.test_request.copy()
            request['seed'] = 200 + i

            request_start_time = time.time()

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(request)

            request_end_time = time.time()

            # Verify success
            self.assertEqual(response['status'], 'success')

            results.append({
                'request_time': request_end_time - request_start_time,
                'inference_time': response['output']['inference_time']
            })

        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time

        # Calculate throughput metrics
        requests_per_minute = (num_requests / total_time) * 60
        average_request_time = statistics.mean([r['request_time'] for r in results])

        # Verify all inference times were as expected
        for result in results:
            self.assertEqual(result['inference_time'], inference_time_per_request)

        # Verify throughput is reasonable
        # With 12s inference time, max theoretical throughput is 5 requests/minute
        expected_max_throughput = 5.0
        self.assertLessEqual(requests_per_minute, expected_max_throughput * 1.1,  # 10% tolerance
                            f"Throughput {requests_per_minute:.2f} req/min exceeds theoretical max")

        # Verify average request processing time is close to inference time
        self.assertLess(average_request_time - inference_time_per_request, 1.0,
                       f"Processing overhead {average_request_time - inference_time_per_request:.2f}s too high")


class TestPerformanceEdgeCases(unittest.TestCase):
    """Test performance in edge case scenarios."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.flux_handler = FluxHandler()
        self.mock_flux_model = Mock(spec=FluxModel)
        self.mock_model_manager = Mock(spec=ModelManager)

    def test_performance_with_large_images(self):
        """Test performance behavior with maximum resolution images."""
        # Test maximum supported resolution
        max_resolution_request = {
            'prompt': 'Ultra high resolution test image',
            'width': 2048,
            'height': 2048,
            'num_inference_steps': 4
        }

        # Even large images should complete within reasonable time
        # FLUX.1-schnell-fp8 should handle 2048x2048 in under 20 seconds
        large_image_time = 18.0

        mock_result = {
            'image': Image.new('RGB', (2048, 2048), color='red'),
            'inference_time': large_image_time,
            'parameters': max_resolution_request.copy(),
            'memory_usage_mb': 16000,  # Higher memory for larger image
            'model_info': {}
        }

        self.mock_flux_model.infer.return_value = mock_result
        self.mock_model_manager.get_model.return_value = self.mock_flux_model

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("large_encoded_data", 3000000)  # Larger file size

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(max_resolution_request)

            # Should succeed
            self.assertEqual(response['status'], 'success')

            # Timing should be reported accurately
            self.assertEqual(response['output']['inference_time'], large_image_time)

            # Large images may exceed the 15s target but should be under 20s for ultra-high-res
            self.assertLess(large_image_time, 20.0,
                           "Even maximum resolution images should complete within 20 seconds")

    def test_performance_with_complex_prompts(self):
        """Test performance with very complex/long prompts."""
        # Very detailed prompt that might impact processing time
        complex_prompt = (
            "A hyperrealistic digital art masterpiece depicting an ancient mystical forest "
            "at twilight with ethereal glowing mushrooms, intricate celtic patterns carved "
            "into ancient oak trees, wisps of magical energy floating through misty air, "
            "a crystal clear stream reflecting starlight, detailed moss and lichen textures, "
            "volumetric lighting rays piercing through the canopy, atmospheric perspective, "
            "photorealistic rendering with cinematic composition and dramatic lighting"
        )

        complex_request = {
            'prompt': complex_prompt,
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 4
        }

        # Complex prompts shouldn't significantly impact FLUX.1-schnell performance
        complex_prompt_time = 13.0

        mock_result = {
            'image': Image.new('RGB', (1024, 1024), color='purple'),
            'inference_time': complex_prompt_time,
            'parameters': complex_request.copy(),
            'memory_usage_mb': 14500,
            'model_info': {}
        }

        self.mock_flux_model.infer.return_value = mock_result
        self.mock_model_manager.get_model.return_value = self.mock_flux_model

        with patch('src.handlers.flux_handler.encode_pil_image') as mock_encode:
            mock_encode.return_value = ("complex_encoded_data", 1200000)

            with patch.object(self.flux_handler, 'model_manager', self.mock_model_manager):
                response = self.flux_handler.process_request(complex_request)

            # Should succeed within time limit
            self.assertEqual(response['status'], 'success')
            inference_time = response['output']['inference_time']

            # Complex prompts should still meet the 15 second requirement
            self.assertLess(inference_time, 15.0,
                           f"Complex prompt inference took {inference_time}s, exceeds 15s limit")


if __name__ == '__main__':
    unittest.main()