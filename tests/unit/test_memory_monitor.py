"""
Unit tests for MemoryMonitor class.

Tests memory tracking accuracy, threshold detection, callback triggering,
and monitoring lifecycle management.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from datetime import datetime

from src.models.memory_monitor import MemoryMonitor, MemoryStats
from src.utils.config import ModelConfig


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for MemoryMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        # Create monitor with short intervals for testing
        self.monitor = MemoryMonitor(check_interval_seconds=0.1)

        # Mock callbacks for testing
        self.warning_callback = Mock()
        self.eviction_callback = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()

    @patch('src.models.memory_monitor.torch')
    @patch('src.models.memory_monitor.psutil')
    def test_memory_stats_collection_gpu(self, mock_psutil, mock_torch):
        """Test GPU memory statistics collection."""
        # Mock CUDA availability and GPU memory
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3  # 8GB
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB
        mock_torch.cuda.memory_reserved.return_value = 3 * 1024**3   # 3GB

        # Mock system memory
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.used = 8 * 1024**3    # 8GB
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        # Create monitor with CUDA available (after mocking)
        test_monitor = MemoryMonitor(check_interval_seconds=0.1)

        # Get stats
        stats = test_monitor.get_current_stats()

        # Verify GPU stats
        self.assertEqual(stats.gpu_total_mb, 8 * 1024)  # 8GB in MB
        self.assertEqual(stats.gpu_allocated_mb, 2 * 1024)  # 2GB in MB
        self.assertEqual(stats.gpu_free_mb, 6 * 1024)  # 6GB free
        self.assertAlmostEqual(stats.gpu_utilization_percent, 25.0, places=1)  # 2GB/8GB = 25%

        # Verify system stats
        self.assertEqual(stats.system_total_mb, 16 * 1024)
        self.assertEqual(stats.system_used_mb, 8 * 1024)
        self.assertEqual(stats.system_utilization_percent, 50.0)

        # Clean up test monitor
        test_monitor.stop_monitoring()

    @patch('src.models.memory_monitor.torch')
    @patch('src.models.memory_monitor.psutil')
    def test_memory_stats_collection_cpu_only(self, mock_psutil, mock_torch):
        """Test memory statistics collection when GPU is not available."""
        # Mock no CUDA
        mock_torch.cuda.is_available.return_value = False

        # Mock system memory
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.used = 8 * 1024**3    # 8GB
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        # Get stats
        stats = self.monitor.get_current_stats()

        # GPU stats should be zero
        self.assertEqual(stats.gpu_total_mb, 0)
        self.assertEqual(stats.gpu_allocated_mb, 0)
        self.assertEqual(stats.gpu_utilization_percent, 0.0)

        # System stats should be populated
        self.assertEqual(stats.system_total_mb, 16 * 1024)
        self.assertEqual(stats.system_utilization_percent, 50.0)

    @patch('src.models.memory_monitor.psutil.virtual_memory')
    def test_memory_pressure_detection(self, mock_virtual_memory):
        """Test memory pressure threshold detection using system memory (since GPU not available)."""
        # Mock system memory with high utilization
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.used = 14 * 1024**3   # 14GB used (87.5% usage)
        mock_memory.available = 2 * 1024**3  # 2GB available
        mock_memory.percent = 87.5  # 87.5% utilization
        mock_virtual_memory.return_value = mock_memory

        # Force stats update
        self.monitor.get_current_stats()

        # Check pressure with default thresholds (warning: 75%, eviction: 85%)
        warning_exceeded, eviction_needed = self.monitor.check_memory_pressure()

        # Should trigger both warning and eviction since 87.5% > 85%
        self.assertTrue(warning_exceeded, "Should exceed warning threshold (75%)")
        self.assertTrue(eviction_needed, "Should need eviction (85%)")

    def test_available_memory_estimation(self):
        """Test available memory estimation with safety buffers."""
        with patch.object(self.monitor, 'get_current_stats') as mock_get_stats:
            mock_stats = MemoryStats()
            mock_stats.gpu_free_mb = 2000  # 2GB free
            mock_stats.system_available_mb = 4000  # 4GB available
            mock_get_stats.return_value = mock_stats

            # Test GPU available memory (with 512MB safety buffer)
            self.monitor.cuda_available = True
            available = self.monitor.estimate_available_memory_mb()
            self.assertEqual(available, 2000 - 512)  # 2GB - 512MB buffer

            # Test CPU available memory (with 1GB safety buffer)
            self.monitor.cuda_available = False
            available = self.monitor.estimate_available_memory_mb()
            self.assertEqual(available, 4000 - 1024)  # 4GB - 1GB buffer

    def test_model_loading_capacity_check(self):
        """Test checking if enough memory is available for model loading."""
        with patch.object(self.monitor, 'estimate_available_memory_mb') as mock_estimate:
            mock_estimate.return_value = 1000  # 1GB available

            # Should be able to load 800MB model
            self.assertTrue(self.monitor.can_load_model(800))

            # Should not be able to load 1200MB model
            self.assertFalse(self.monitor.can_load_model(1200))

    def test_callback_registration_and_triggering(self):
        """Test callback registration and triggering on threshold events."""
        # Register callbacks
        self.monitor.add_warning_callback(self.warning_callback)
        self.monitor.add_eviction_callback(self.eviction_callback)

        # Mock high memory usage
        with patch.object(self.monitor, 'check_memory_pressure') as mock_check:
            with patch.object(self.monitor, 'get_current_stats') as mock_stats:
                mock_check.return_value = (True, True)  # Warning and eviction
                mock_stats.return_value = MemoryStats(gpu_utilization_percent=90.0)

                # Start monitoring briefly
                self.monitor.start_monitoring()
                time.sleep(0.3)  # Let monitoring loop run a few times
                self.monitor.stop_monitoring()

        # Callbacks should have been triggered
        self.warning_callback.assert_called()
        self.eviction_callback.assert_called()

    def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        # Initially not monitoring
        self.assertFalse(self.monitor._monitoring)

        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring)
        self.assertIsNotNone(self.monitor._monitor_thread)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)

        # Should be able to restart
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring)
        self.monitor.stop_monitoring()

    @patch('src.models.memory_monitor.torch')
    def test_gpu_cache_clearing(self, mock_torch):
        """Test GPU cache clearing functionality."""
        # Set up CUDA availability and create monitor for this test
        mock_torch.cuda.is_available.return_value = True
        test_monitor = MemoryMonitor(check_interval_seconds=0.1)

        # Test successful cache clear
        test_monitor.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called_once()

        # Test cache clear with exception
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("GPU error")

        # Should not raise exception
        test_monitor.clear_gpu_cache()

        # Clean up test monitor
        test_monitor.stop_monitoring()

    def test_memory_summary_generation(self):
        """Test comprehensive memory summary generation."""
        with patch.object(self.monitor, 'get_current_stats') as mock_get_stats:
            with patch.object(self.monitor, 'check_memory_pressure') as mock_check:
                with patch.object(self.monitor, 'estimate_available_memory_mb') as mock_estimate:
                    # Mock return values
                    mock_stats = MemoryStats(gpu_utilization_percent=75.0)
                    mock_get_stats.return_value = mock_stats
                    mock_check.return_value = (True, False)  # Warning but not eviction
                    mock_estimate.return_value = 1500

                    summary = self.monitor.get_memory_summary()

                    # Verify summary structure
                    self.assertIn('stats', summary)
                    self.assertIn('thresholds', summary)
                    self.assertIn('available_memory_mb', summary)
                    self.assertIn('monitoring_active', summary)
                    self.assertIn('cuda_available', summary)

                    # Verify threshold information
                    thresholds = summary['thresholds']
                    self.assertTrue(thresholds['warning_exceeded'])
                    self.assertFalse(thresholds['eviction_needed'])

                    # Verify available memory
                    self.assertEqual(summary['available_memory_mb'], 1500)

    def test_concurrent_monitoring_access(self):
        """Test thread safety of monitoring operations."""
        results = []
        errors = []

        def get_stats_concurrently():
            try:
                for _ in range(10):
                    stats = self.monitor.get_current_stats()
                    results.append(stats)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        # Start multiple threads accessing monitor
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=get_stats_concurrently)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have no errors and multiple results
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)

    def test_stats_timestamp_and_serialization(self):
        """Test MemoryStats timestamp handling and serialization."""
        stats = MemoryStats()

        # Should have timestamp
        self.assertIsNotNone(stats.timestamp)
        self.assertIsInstance(stats.timestamp, datetime)

        # Should serialize to dict
        stats_dict = stats.to_dict()
        self.assertIsInstance(stats_dict, dict)
        self.assertIn('timestamp', stats_dict)
        self.assertIn('gpu_total_mb', stats_dict)
        self.assertIn('system_utilization_percent', stats_dict)


if __name__ == '__main__':
    unittest.main()