"""
Structured Logging Configuration

Provides comprehensive logging setup for the multi-modal inference worker
with request tracking, performance monitoring, and debugging capabilities.
"""

import logging
import logging.config
import sys
import json
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager


class RequestContextFilter(logging.Filter):
    """
    Adds request context to log records for request tracking.
    """

    def __init__(self):
        super().__init__()
        self.request_id = None
        self.modality = None

    def set_context(self, request_id: str, modality: Optional[str] = None):
        """Set the current request context."""
        self.request_id = request_id
        self.modality = modality

    def clear_context(self):
        """Clear the current request context."""
        self.request_id = None
        self.modality = None

    def filter(self, record):
        """Add request context to log record."""
        record.request_id = getattr(self, 'request_id', None) or 'no-request'
        record.modality = getattr(self, 'modality', None) or 'unknown'
        return True


class PerformanceLogger:
    """
    Specialized logger for performance monitoring and timing.
    """

    def __init__(self):
        self.logger = logging.getLogger('performance')
        self._timings: Dict[str, float] = {}

    @contextmanager
    def time_operation(self, operation_name: str, request_id: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            self.logger.info(f"[{request_id}] Starting {operation_name}")
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._timings[operation_name] = duration_ms
            self.logger.info(
                f"[{request_id}] Completed {operation_name} in {duration_ms:.2f}ms"
            )

    def log_inference_metrics(
        self,
        request_id: str,
        modality: str,
        total_time_ms: float,
        model_load_time_ms: float,
        inference_time_ms: float,
        models_used: list
    ):
        """Log comprehensive inference performance metrics."""
        metrics = {
            'request_id': request_id,
            'modality': modality,
            'total_time_ms': round(total_time_ms, 2),
            'model_load_time_ms': round(model_load_time_ms, 2),
            'inference_time_ms': round(inference_time_ms, 2),
            'models_used': models_used,
            'models_count': len(models_used)
        }

        self.logger.info(f"METRICS: {json.dumps(metrics)}")


class LoggingConfig:
    """
    Centralized logging configuration for the multi-modal worker.
    """

    # Global request context filter instance
    request_filter = RequestContextFilter()
    performance_logger = PerformanceLogger()

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[{asctime}] {levelname:8} [{request_id}] [{modality:12}] {name:20} | {message}',
                'style': '{'
            },
            'detailed': {
                'format': '[{asctime}] {levelname:8} [{request_id}] [{modality:12}] {name:20} | {funcName}:{lineno} | {message}',
                'style': '{'
            },
            'performance': {
                'format': '[{asctime}] PERF [{request_id}] | {message}',
                'style': '{'
            },
            'json': {
                'format': '{{"timestamp": "{asctime}", "level": "{levelname}", "request_id": "{request_id}", "modality": "{modality}", "logger": "{name}", "message": "{message}"}}',
                'style': '{'
            }
        },
        'filters': {
            'request_context': {
                '()': RequestContextFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filters': ['request_context'],
                'stream': sys.stdout
            },
            'debug_console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filters': ['request_context'],
                'stream': sys.stdout
            },
            'performance': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'performance',
                'filters': ['request_context'],
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filters': ['request_context'],
                'filename': '/tmp/multi-modal-worker.log'
            }
        },
        'loggers': {
            'request': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'model': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'inference': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'performance': {
                'level': 'INFO',
                'handlers': ['performance'],
                'propagate': False
            },
            'system': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'validation': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }

    @classmethod
    def setup_logging(cls, debug_mode: bool = False, log_level: str = None):
        """
        Setup logging configuration for the worker.

        Args:
            debug_mode: Enable debug-level logging and detailed formatting
            log_level: Specific log level string (DEBUG, INFO, WARNING, ERROR)
        """
        config = cls.LOGGING_CONFIG.copy()

        # Handle log_level parameter
        if log_level:
            log_level_upper = log_level.upper()
            if log_level_upper not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                # Fall back to INFO for invalid levels
                log_level_upper = 'INFO'

            config['root']['level'] = log_level_upper

            # Set all loggers to the specified level
            for logger_config in config['loggers'].values():
                logger_config['level'] = log_level_upper

            # Use debug console for DEBUG level
            if log_level_upper == 'DEBUG':
                config['handlers']['console'] = config['handlers']['debug_console'].copy()
        elif debug_mode:
            # Enable debug level logging
            config['handlers']['console'] = config['handlers']['debug_console'].copy()
            config['root']['level'] = 'DEBUG'

            # Set all loggers to debug level
            for logger_config in config['loggers'].values():
                logger_config['level'] = 'DEBUG'

        logging.config.dictConfig(config)

        # Store reference to the filter for request context management
        cls.request_filter = logging.getLogger().handlers[0].filters[0]

        # Log startup message
        startup_logger = logging.getLogger('system')
        startup_logger.info("Multi-Modal Inference Worker logging initialized")
        if debug_mode:
            startup_logger.info("Debug mode enabled - verbose logging active")

    @classmethod
    @contextmanager
    def request_context(cls, request_id: str, modality: Optional[str] = None):
        """
        Context manager for request-specific logging.

        Args:
            request_id: Unique request identifier
            modality: Processing modality (if determined)
        """
        try:
            cls.request_filter.set_context(request_id, modality)
            yield
        finally:
            cls.request_filter.clear_context()

    @classmethod
    def get_logger(cls, name: str = "multi-modal-worker") -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name (e.g., 'request', 'model', 'inference'), defaults to 'multi-modal-worker'

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)

    @classmethod
    def get_performance_logger(cls) -> PerformanceLogger:
        """Get the performance logger instance."""
        return cls.performance_logger


# Convenience logger instances
def get_request_logger() -> logging.Logger:
    """Get logger for request processing."""
    return logging.getLogger('request')


def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return logging.getLogger('model')


def get_inference_logger() -> logging.Logger:
    """Get logger for inference operations."""
    return logging.getLogger('inference')


def get_validation_logger() -> logging.Logger:
    """Get logger for validation operations."""
    return logging.getLogger('validation')


def get_system_logger() -> logging.Logger:
    """Get logger for system operations."""
    return logging.getLogger('system')


def log_request_start(request_id: str, modality: str, parameters: Dict[str, Any]):
    """Log the start of request processing."""
    logger = get_request_logger()
    param_summary = {k: str(v)[:100] for k, v in parameters.items()}
    logger.info(f"Processing {modality} request with parameters: {param_summary}")


def log_request_complete(request_id: str, success: bool, processing_time_ms: float):
    """Log the completion of request processing."""
    logger = get_request_logger()
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Request {status} - Total processing time: {processing_time_ms:.2f}ms")


def log_model_operation(operation: str, model_name: str, details: Optional[str] = None):
    """Log model management operations."""
    logger = get_model_logger()
    message = f"{operation}: {model_name}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_inference_step(step: str, details: Optional[str] = None):
    """Log individual inference processing steps."""
    logger = get_inference_logger()
    message = f"Step: {step}"
    if details:
        message += f" - {details}"
    logger.info(message)


def log_validation_result(field: str, result: bool, message: Optional[str] = None):
    """Log validation results."""
    logger = get_validation_logger()
    status = "VALID" if result else "INVALID"
    log_message = f"Validation {status}: {field}"
    if message:
        log_message += f" - {message}"
    logger.info(log_message)


def log_system_event(event: str, details: Optional[Dict[str, Any]] = None):
    """Log system-level events."""
    logger = get_system_logger()
    message = f"System event: {event}"
    if details:
        message += f" - {json.dumps(details)}"
    logger.info(message)


def log_error(logger_name: str, error: Exception, context: Optional[str] = None):
    """Log errors with full context."""
    logger = logging.getLogger(logger_name)
    message = f"ERROR: {type(error).__name__}: {str(error)}"
    if context:
        message = f"{context} - {message}"
    logger.error(message)


class RequestContext:
    """
    Context manager for request tracking and performance monitoring.

    Provides structured logging context and performance metrics collection
    for individual requests throughout their lifecycle.
    """

    def __init__(self, request_id: str, logger: Optional[logging.Logger] = None):
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