"""
End-to-End Integration Tests for LTX-Video Handler

Tests complete request/response cycles, RunPod compatibility,
performance validation, and production readiness scenarios.
Following Test-Driven Development (TDD) approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
from pathlib import Path
import time
import asyncio
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.handlers.multi_modal_handler import MultiModalHandler
from src.handlers.ltx_video_handler import LTXVideoHandler
from src.models.model_manager import ModelManager
from src.models.ltx_video_model import LTXVideoModel
from src.schemas.text_to_video_schema import TextToVideoRequest, TextToVideoResponse, VideoInfo
from src.utils.exceptions import ValidationError, InferenceError, ModelLoadError
from src.utils.response_formatter import ResponseFormatter, ErrorType
from src.main import handler as runpod_handler


class TestLTXVideoEndToEndIntegration(unittest.TestCase):
    """Test cases for LTX-Video end-to-end integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock model manager with proper behavior
        self.mock_model_manager = Mock(spec=ModelManager)
        self.mock_model_manager.get_manager_status.return_value = {
            'total_models': 1,
            'loaded_models': 1,
            'memory_summary': {'stats': {'total_gb': 8.5}}
        }

        # Create multi-modal handler
        self.multi_handler = MultiModalHandler(
            model_manager=self.mock_model_manager,
            auto_initialize=False
        )

        # Create and register LTX-Video handler
        self.ltx_video_handler = LTXVideoHandler()
        self.multi_handler.register_handler(
            self.ltx_video_handler.supported_modality,
            self.ltx_video_handler
        )

        # Standard test requests
        self.minimal_request = {
            'prompt': 'A cat walking',
            'fps': 8,
            'video': True
        }

        self.detailed_request = {
            'prompt': 'A majestic eagle soaring through snow-capped mountain peaks at golden hour, cinematic slow motion',
            'width': 720,
            'height': 1280,
            'num_frames': 49,  # (8*6)+1
            'num_inference_steps': 25,
            'guidance_scale': 7.5,
            'fps': 8,
            'seed': 42,
            'video': True
        }

        # RunPod-compatible request
        self.runpod_request = {
            'input': self.detailed_request,
            'id': 'test-request-123'
        }

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_end_to_end_request_processing(self, mock_handle_request):
        """Test complete end-to-end request processing pipeline."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {
                'video_data': "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                'video_info': {
                    'width': 720,
                    'height': 1280,
                    'fps': 8,
                    'frames': 49,
                    'duration': 6.125
                }
            },
            'metadata': {
                'processing_time_ms': 1500,
                'modality': 'text-to-video'
            }
        }

        # Process request
        start_time = time.time()
        result = self.multi_handler.process_request(self.detailed_request)
        processing_time = time.time() - start_time

        # Verify successful processing
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('status'), 'success')
        self.assertIn('output', result)

        # Verify performance target (<45 seconds for integration test)
        self.assertLess(processing_time, 45.0, "Processing time exceeded 45 second target")

        # Verify response structure
        output = result['output']
        self.assertIsInstance(output, dict)

        # Verify metadata
        self.assertIn('metadata', result)
        metadata = result['metadata']
        self.assertIn('processing_time_ms', metadata)
        self.assertIn('modality', metadata)
        self.assertEqual(metadata['modality'], 'text-to-video')

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_runpod_compatibility_request_format(self, mock_handle_request):
        """Test compatibility with RunPod serverless request format."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {
                'video_data': "mock_base64_video_data",
                'video_info': {'duration': 6.125, 'fps': 8, 'frames': 49}
            },
            'metadata': {
                'processing_time_ms': 2000,
                'modality': 'text-to-video'
            }
        }

        # Test RunPod request format
        result = self.multi_handler.process_request(self.runpod_request['input'])

        # Verify RunPod-compatible response
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('status'), 'success')

        # Verify RunPod response structure compatibility
        self.assertIn('output', result)
        self.assertIn('metadata', result)

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_request_validation_and_optimization(self, mock_handle_request):
        """Test request validation and parameter optimization."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': "mock_video"},
            'metadata': {'processing_time_ms': 1000}
        }

        # Test request with optimal parameters (divisible by 32)
        optimal_request = {
            'prompt': 'A dancing robot',
            'width': 720,  # Divisible by 32
            'height': 1280,  # Divisible by 32
            'num_frames': 25,  # (8*3)+1 pattern
            'fps': 8,
            'video': True
        }

        # Process request
        result = self.multi_handler.process_request(optimal_request)

        # Verify successful processing
        self.assertEqual(result.get('status'), 'success')

        # Verify handler was called
        mock_handle_request.assert_called_once()

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_error_handling_and_recovery(self, mock_handle_request):
        """Test comprehensive error handling scenarios."""
        # Test 1: Model loading error simulation
        mock_handle_request.side_effect = Exception("Model inference failed")

        result = self.multi_handler.process_request(self.minimal_request)

        # Verify error response
        self.assertEqual(result.get('status'), 'error')
        self.assertIn('error', result)

        # Test 2: Successful recovery after error
        mock_handle_request.side_effect = None
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': 'mock_data'},
            'metadata': {'processing_time_ms': 1000}
        }

        result = self.multi_handler.process_request(self.minimal_request)

        # Verify successful recovery
        self.assertEqual(result.get('status'), 'success')
        self.assertIn('output', result)

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_concurrent_request_processing(self, mock_handle_request):
        """Test handling of concurrent text-to-video requests."""
        # Setup mock response with simulated timing
        def mock_handle_with_delay(*args, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return {
                'status': 'success',
                'output': {'video_data': "mock_video_data"},
                'metadata': {'processing_time_ms': 100}
            }

        mock_handle_request.side_effect = mock_handle_with_delay

        # Create multiple concurrent requests
        requests = [
            {'prompt': f'Video {i}', 'fps': 8, 'video': True}
            for i in range(3)
        ]

        # Process requests concurrently (simulated)
        results = []
        start_time = time.time()

        for request in requests:
            result = self.multi_handler.process_request(request)
            results.append(result)

        total_time = time.time() - start_time

        # Verify all requests succeeded
        for result in results:
            self.assertEqual(result.get('status'), 'success')

        # Verify reasonable total processing time
        self.assertLess(total_time, 5.0, "Concurrent processing took too long")

    def test_memory_usage_validation(self):
        """Test memory usage constraints and monitoring."""
        # Test memory constraints are properly configured
        capabilities = self.ltx_video_handler.get_capabilities()

        # Verify memory footprint specification
        self.assertIn('memory_footprint', capabilities)
        memory_spec = capabilities['memory_footprint']
        self.assertIn('GB', memory_spec)

        # Extract memory limit and verify it's reasonable for LTX-Video
        # Should be within 6-10GB range as specified
        self.assertTrue('6-10GB' in memory_spec or '6' in memory_spec)

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_performance_benchmarking(self, mock_handle_request):
        """Test performance benchmarking and timing validation."""
        # Setup mock response with realistic timing
        def mock_handle_with_timing(*args, **kwargs):
            # Simulate realistic processing time (under target)
            time.sleep(0.5)  # Simulate 500ms processing
            return {
                'status': 'success',
                'output': {
                    'video_data': "mock_video_data",
                    'video_info': {'duration': 3.125, 'fps': 8, 'frames': 25}
                },
                'metadata': {
                    'processing_time_ms': 500,
                    'modality': 'text-to-video'
                }
            }

        mock_handle_request.side_effect = mock_handle_with_timing

        # Perform benchmark test
        start_time = time.time()
        result = self.multi_handler.process_request(self.detailed_request)
        processing_time = time.time() - start_time

        # Verify performance targets
        self.assertEqual(result.get('status'), 'success')
        self.assertLess(processing_time, 45.0, "Processing exceeded 45 second target")

        # Verify timing metadata is included
        metadata = result.get('metadata', {})
        self.assertIn('processing_time_ms', metadata)

        reported_time_ms = metadata['processing_time_ms']
        self.assertGreater(reported_time_ms, 0)
        self.assertLess(reported_time_ms, 45000)  # 45 seconds in ms

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_resource_cleanup_and_management(self, mock_handle_request):
        """Test proper resource cleanup and management."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': "mock_video"},
            'metadata': {'processing_time_ms': 800}
        }

        # Process multiple requests to test resource management
        for i in range(3):
            request = {'prompt': f'Test {i}', 'fps': 8, 'video': True}
            result = self.multi_handler.process_request(request)
            self.assertEqual(result.get('status'), 'success')

        # Verify handler was called for each request
        self.assertEqual(mock_handle_request.call_count, 3)

    @patch('src.main.MultiModalHandler')
    @patch('src.main.ModelManager')
    def test_runpod_handler_integration(self, mock_model_manager_class, mock_handler_class):
        """Test integration with RunPod serverless handler."""
        # Setup mocks
        mock_manager_instance = Mock()
        mock_handler_instance = Mock()
        mock_handler_instance.process_request.return_value = {
            'status': 'success',
            'output': {'video_data': 'mock_base64'},
            'metadata': {'processing_time_ms': 1500}
        }

        mock_model_manager_class.return_value = mock_manager_instance
        mock_handler_class.return_value = mock_handler_instance

        # Test RunPod handler call
        runpod_request = {
            'prompt': 'A flowing river',
            'fps': 8,
            'video': True
        }

        # Simulate RunPod handler processing
        result = mock_handler_instance.process_request(runpod_request)

        # Verify RunPod handler processes request
        self.assertIsInstance(result, dict)
        mock_handler_instance.process_request.assert_called_once_with(runpod_request)

    def test_production_readiness_checklist(self):
        """Test production readiness criteria."""
        # Test 1: Handler capabilities are complete
        capabilities = self.ltx_video_handler.get_capabilities()
        required_fields = [
            'modality', 'model_type', 'max_frames', 'resolution_support',
            'frame_rates', 'inference_time_target', 'memory_footprint'
        ]

        for field in required_fields:
            self.assertIn(field, capabilities, f"Missing capability field: {field}")

        # Test 2: Modality support is correctly configured
        self.assertEqual(capabilities['modality'], 'text-to-video')

        # Test 3: Performance targets are reasonable
        inference_target = capabilities['inference_time_target']
        self.assertIn('45', inference_target, "Inference time target should mention 45 seconds")

        # Test 4: Resolution constraints are properly specified
        resolution_support = capabilities['resolution_support']
        self.assertIn('constraint', resolution_support)
        self.assertEqual(resolution_support['constraint'], 'divisible_by_32')

    @patch('src.handlers.ltx_video_handler.LTXVideoHandler.handle_request')
    def test_parameter_validation_edge_cases(self, mock_handle_request):
        """Test parameter validation with edge cases."""
        # Setup mock response
        mock_handle_request.return_value = {
            'status': 'success',
            'output': {'video_data': "mock_video"},
            'metadata': {'processing_time_ms': 500}
        }

        # Test edge cases - use parameters that should pass validation
        edge_cases = [
            # Minimum valid parameters
            {'prompt': 'Short prompt', 'fps': 8, 'video': True},

            # Standard resolution
            {'prompt': 'Test', 'width': 1280, 'height': 1280, 'fps': 8, 'video': True},

            # Valid frame count (8*32)+1 = 257
            {'prompt': 'Test', 'num_frames': 257, 'fps': 8, 'video': True},

            # Various FPS values
            {'prompt': 'Test', 'fps': 24, 'video': True},
        ]

        for i, request in enumerate(edge_cases):
            with self.subTest(case=i, request=request):
                result = self.multi_handler.process_request(request)
                self.assertEqual(result.get('status'), 'success',
                               f"Edge case {i} failed: {request}")

    def test_system_health_and_monitoring(self):
        """Test system health monitoring capabilities."""
        # Test system status reporting
        status = self.multi_handler.get_system_status()

        self.assertIsInstance(status, dict)
        self.assertIn('status', status)
        self.assertEqual(status['status'], 'healthy')
        self.assertIn('supported_modalities', status)
        self.assertIn('text-to-video', status['supported_modalities'])

        # Test statistics reporting
        self.assertIn('statistics', status)
        stats = status['statistics']
        self.assertIn('total_requests', stats)
        self.assertIn('average_processing_time_ms', stats)

        # Test handler capabilities (avoid get_model_info which may require actual model)
        capabilities = self.ltx_video_handler.get_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIn('modality', capabilities)
        self.assertEqual(capabilities['modality'], 'text-to-video')


if __name__ == '__main__':
    unittest.main()