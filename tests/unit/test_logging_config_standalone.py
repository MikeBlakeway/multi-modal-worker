"""
Standalone tests for MMI-004 Logging Configuration functionality.

These tests validate logging setup and request context management
without requiring the full application stack.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch
from contextlib import contextmanager


# Inline LoggingConfig implementation for testing
class LoggingConfig:
    """
    Centralized logging configuration for the multi-modal inference worker.
    """

    _setup_complete = False

    @classmethod
    def setup_logging(cls, log_level: str = "INFO"):
        """
        Setup structured logging configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        if cls._setup_complete:
            return  # Avoid duplicate setup

        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger.addHandler(handler)

        cls._setup_complete = True

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name (default: "multi-modal-worker")

        Returns:
            Configured logger instance
        """
        if not cls._setup_complete:
            cls.setup_logging()

        logger_name = name if name else "multi-modal-worker"
        return logging.getLogger(logger_name)


class RequestContext:
    """
    Context manager for request tracking and performance monitoring.
    """

    def __init__(self, request_id: str, logger: logging.Logger = None):
        """
        Initialize request context.

        Args:
            request_id: Unique identifier for the request
            logger: Optional logger instance for context logging
        """
        self.request_id = request_id
        self.logger = logger
        self.start_time = None
        self.performance_data = {}

    def __enter__(self):
        """Enter the context and start timing."""
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Request started: {self.request_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and log completion."""
        if exc_type is not None:
            # Exception occurred
            if self.logger:
                self.logger.error(f"Request failed: {self.request_id} - {exc_type.__name__}: {exc_val}")
        else:
            # Successful completion
            total_time = self.get_total_time_ms()
            if self.logger:
                perf_summary = ", ".join([f"{k}: {v}ms" for k, v in self.performance_data.items()])
                self.logger.info(f"Request completed: {self.request_id} - Total: {total_time:.1f}ms - Steps: {perf_summary}")

    def add_performance_metric(self, step_name: str, duration_ms: float):
        """
        Add a performance metric for a processing step.

        Args:
            step_name: Name of the processing step
            duration_ms: Duration in milliseconds
        """
        self.performance_data[step_name] = duration_ms

    def get_total_time_ms(self) -> float:
        """
        Get total request time in milliseconds.

        Returns:
            Total time since context start in milliseconds
        """
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000.0


class TestLoggingConfig:
    """Test cases for LoggingConfig setup and configuration."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset logging configuration before each test
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
        LoggingConfig._setup_complete = False

    def test_setup_logging_default_level(self):
        """Test default logging setup."""
        LoggingConfig.setup_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_logging_debug_level(self):
        """Test logging setup with debug level."""
        LoggingConfig.setup_logging(log_level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_error_level(self):
        """Test logging setup with error level."""
        LoggingConfig.setup_logging(log_level="ERROR")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level (should default to INFO)."""
        LoggingConfig.setup_logging(log_level="INVALID")

        root_logger = logging.getLogger()
        # Should fall back to INFO level
        assert root_logger.level == logging.INFO

    def test_setup_logging_case_insensitive(self):
        """Test that log level is case insensitive."""
        LoggingConfig.setup_logging(log_level="debug")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_get_logger_with_name(self):
        """Test getting a named logger."""
        LoggingConfig.setup_logging()

        logger = LoggingConfig.get_logger("test_module")
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_without_name(self):
        """Test getting logger without specific name."""
        LoggingConfig.setup_logging()

        logger = LoggingConfig.get_logger()
        assert logger.name == "multi-modal-worker"

    def test_multiple_setup_calls(self):
        """Test that multiple setup calls don't create duplicate handlers."""
        LoggingConfig.setup_logging()
        initial_handler_count = len(logging.getLogger().handlers)

        LoggingConfig.setup_logging()
        final_handler_count = len(logging.getLogger().handlers)

        # Should not create additional handlers
        assert final_handler_count == initial_handler_count

    def test_logger_hierarchy(self):
        """Test that child loggers inherit configuration."""
        LoggingConfig.setup_logging(log_level="DEBUG")

        parent_logger = LoggingConfig.get_logger("parent")
        child_logger = LoggingConfig.get_logger("parent.child")

        # Child should inherit parent's effective level
        assert child_logger.isEnabledFor(logging.DEBUG)


class TestRequestContext:
    """Test cases for RequestContext context manager."""

    def setup_method(self):
        """Setup test fixtures."""
        LoggingConfig.setup_logging(log_level="DEBUG")
        self.logger = LoggingConfig.get_logger("test")

    def test_request_context_basic_usage(self):
        """Test basic request context usage."""
        request_id = "test-123"

        with RequestContext(request_id) as context:
            assert context.request_id == request_id
            assert context.start_time is not None
            assert context.performance_data == {}

    def test_request_context_performance_tracking(self):
        """Test performance tracking within request context."""
        with RequestContext("perf-test") as context:
            context.add_performance_metric("step1", 500)  # 500ms
            context.add_performance_metric("step2", 1000)  # 1000ms

            assert "step1" in context.performance_data
            assert "step2" in context.performance_data
            assert context.performance_data["step1"] == 500
            assert context.performance_data["step2"] == 1000

    def test_request_context_total_time_calculation(self):
        """Test that total request time is calculated correctly."""
        with patch('time.time', side_effect=[1000.0, 1002.5, 1002.5]):  # Extra value for exit time calculation
            with RequestContext("timing-test") as context:
                pass  # Context exit will calculate total time

            total_time = context.get_total_time_ms()
            assert total_time == 2500.0  # 2.5 seconds in milliseconds

    def test_request_context_logging_integration(self):
        """Test that request context integrates with logging."""
        with patch.object(self.logger, 'info') as mock_info:
            with RequestContext("log-test", logger=self.logger):
                pass

            # Verify that context completion was logged
            mock_info.assert_called()
            call_args = mock_info.call_args[0][0]
            assert "Request completed" in call_args

    def test_request_context_without_logger(self):
        """Test request context without explicit logger."""
        # Should not raise exception
        with RequestContext("no-logger-test") as context:
            assert context.request_id == "no-logger-test"

    def test_request_context_with_exception(self):
        """Test request context behavior when exception occurs."""
        with patch.object(self.logger, 'error') as mock_error:
            try:
                with RequestContext("exception-test", logger=self.logger):
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Verify that exception was logged
            mock_error.assert_called()

    def test_request_context_performance_data_isolation(self):
        """Test that performance data is isolated between contexts."""
        # First context
        with RequestContext("context1") as ctx1:
            ctx1.add_performance_metric("metric1", 100)

        # Second context
        with RequestContext("context2") as ctx2:
            ctx2.add_performance_metric("metric2", 200)

            # Should only contain its own metrics
            assert "metric1" not in ctx2.performance_data
            assert "metric2" in ctx2.performance_data

    def test_request_context_nested_usage(self):
        """Test nested request context usage."""
        with RequestContext("outer") as outer_ctx:
            outer_ctx.add_performance_metric("outer_metric", 100)

            with RequestContext("inner") as inner_ctx:
                inner_ctx.add_performance_metric("inner_metric", 50)

                # Each context maintains its own data
                assert "outer_metric" not in inner_ctx.performance_data
                assert "inner_metric" not in outer_ctx.performance_data


class TestRequestContextEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        LoggingConfig.setup_logging()

    def test_empty_request_id(self):
        """Test request context with empty request ID."""
        with RequestContext("") as context:
            assert context.request_id == ""
            assert context.start_time is not None

    def test_none_request_id(self):
        """Test request context with None request ID."""
        with RequestContext(None) as context:
            assert context.request_id is None
            assert context.start_time is not None

    def test_very_long_request_id(self):
        """Test request context with very long request ID."""
        long_id = "x" * 1000
        with RequestContext(long_id) as context:
            assert context.request_id == long_id

    def test_unicode_request_id(self):
        """Test request context with Unicode request ID."""
        unicode_id = "request-测试-🎯"
        with RequestContext(unicode_id) as context:
            assert context.request_id == unicode_id

    def test_performance_metric_with_zero_time(self):
        """Test adding performance metric with zero time."""
        with RequestContext("zero-time-test") as context:
            context.add_performance_metric("instant", 0)
            assert context.performance_data["instant"] == 0

    def test_performance_metric_with_negative_time(self):
        """Test adding performance metric with negative time."""
        with RequestContext("negative-time-test") as context:
            context.add_performance_metric("negative", -100)
            # Should handle gracefully
            assert "negative" in context.performance_data

    def test_duplicate_performance_metrics(self):
        """Test adding duplicate performance metrics."""
        with RequestContext("duplicate-test") as context:
            context.add_performance_metric("step", 100)
            context.add_performance_metric("step", 200)  # Overwrite

            # Should contain the latest value
            assert context.performance_data["step"] == 200

    def test_context_manager_exception_handling(self):
        """Test that context manager handles exceptions gracefully."""
        logger = LoggingConfig.get_logger("exception_test")

        with patch.object(logger, 'error') as mock_error:
            try:
                with RequestContext("exception-handling", logger=logger):
                    raise RuntimeError("Simulated error")
            except RuntimeError:
                pass

            # Should have logged the exception
            mock_error.assert_called()


class TestLoggingPerformance:
    """Test performance-related aspects of logging configuration."""

    def test_logger_creation_performance(self):
        """Test that logger creation is efficient."""
        LoggingConfig.setup_logging()

        # Create many loggers quickly
        start_time = time.time()
        for i in range(100):
            LoggingConfig.get_logger(f"performance_test_{i}")
        end_time = time.time()

        # Should complete quickly (less than 1 second)
        assert (end_time - start_time) < 1.0

    def test_request_context_overhead(self):
        """Test that request context has minimal overhead."""
        start_time = time.time()

        # Create many request contexts
        for i in range(100):
            with RequestContext(f"overhead_test_{i}"):
                pass

        end_time = time.time()

        # Should complete quickly (less than 1 second for 100 contexts)
        assert (end_time - start_time) < 1.0

    def test_performance_metric_addition_speed(self):
        """Test that adding performance metrics is fast."""
        with RequestContext("metric-speed-test") as context:
            start_time = time.time()

            # Add many performance metrics
            for i in range(1000):
                context.add_performance_metric(f"metric_{i}", i * 10)

            end_time = time.time()

            # Should complete quickly
            assert (end_time - start_time) < 1.0
            assert len(context.performance_data) == 1000


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def test_full_request_lifecycle_logging(self):
        """Test logging throughout a complete request lifecycle."""
        LoggingConfig.setup_logging(log_level="DEBUG")
        logger = LoggingConfig.get_logger("integration_test")

        with patch.object(logger, 'info') as mock_info, \
             patch.object(logger, 'debug') as mock_debug:

            with RequestContext("lifecycle-test", logger=logger) as context:
                # Simulate request processing steps
                logger.debug("Starting validation")
                context.add_performance_metric("validation", 100)

                logger.debug("Loading models")
                context.add_performance_metric("model_loading", 500)

                logger.debug("Processing inference")
                context.add_performance_metric("inference", 2000)

                logger.info("Request processed successfully")

            # Verify appropriate logging occurred
            assert mock_debug.call_count >= 3
            assert mock_info.call_count >= 2  # Success message + completion

    def test_error_logging_with_context(self):
        """Test error logging with request context."""
        LoggingConfig.setup_logging()
        logger = LoggingConfig.get_logger("error_test")

        with patch.object(logger, 'error') as mock_error:
            try:
                with RequestContext("error-context-test", logger=logger):
                    logger.error("Validation failed", extra={"error_code": "VAL001"})
                    raise ValueError("Processing error")
            except ValueError:
                pass

            # Should have logged both the explicit error and context error
            assert mock_error.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])