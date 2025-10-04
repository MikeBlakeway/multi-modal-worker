"""
Custom exceptions for multi-modal inference worker.

Provides specific error types for model management, memory handling,
and inference operations to enable proper error handling and recovery.
"""


class ModelManagementError(Exception):
    """Base exception for model management operations."""
    pass


class ModelLoadError(ModelManagementError):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class ModelNotFoundError(ModelManagementError):
    """Raised when requested model is not found."""

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        super().__init__(f"Model '{model_name}' not found at path: {model_path}")


class MemoryError(ModelManagementError):
    """Raised when GPU memory operations fail."""

    def __init__(self, operation: str, required_mb: int, available_mb: int):
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient memory for {operation}: "
            f"required {required_mb}MB, available {available_mb}MB"
        )


class ModelEvictionError(ModelManagementError):
    """Raised when model eviction fails."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to evict model '{model_name}': {reason}")


class ConcurrencyError(ModelManagementError):
    """Raised when concurrent model operations conflict."""

    def __init__(self, operation: str, model_name: str):
        self.operation = operation
        self.model_name = model_name
        super().__init__(f"Concurrent {operation} operation blocked for model '{model_name}'")


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    def __init__(self, setting: str, reason: str):
        self.setting = setting
        self.reason = reason
        super().__init__(f"Configuration error for '{setting}': {reason}")


class InferenceError(Exception):
    """Base exception for inference operations."""
    pass


class UnsupportedModalityError(InferenceError):
    """Raised when requested modality is not supported."""

    def __init__(self, modality: str, supported_modalities: list):
        self.modality = modality
        self.supported_modalities = supported_modalities
        super().__init__(
            f"Unsupported modality '{modality}'. "
            f"Supported: {', '.join(supported_modalities)}"
        )


class ValidationError(InferenceError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: str, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid {field} '{value}': {reason}")


class ProcessingError(InferenceError):
    """Raised when image or data processing operations fail."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Processing error in {operation}: {reason}")