"""
Configuration management for multi-modal inference worker.

Centralized configuration handling with environment variable support,
validation, and sensible defaults for model management and inference settings.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from .exceptions import ConfigurationError


class ModelConfig:
    """Configuration for model management and memory settings."""

    def __init__(self):
        """Initialize configuration with environment variables and defaults."""

        # Model storage paths
        self.model_cache_dir = self._get_env_path(
            "MODEL_CACHE_DIR",
            "/runpod-volume/models"
        )

        # Memory management settings
        self.memory_threshold_percent = self._get_env_int(
            "MEMORY_THRESHOLD_PERCENT",
            85,
            min_val=50,
            max_val=95
        )

        self.memory_warning_percent = self._get_env_int(
            "MEMORY_WARNING_PERCENT",
            75,
            min_val=40,
            max_val=90
        )

        # Model management settings
        self.max_models_in_memory = self._get_env_int(
            "MAX_MODELS_IN_MEMORY",
            3,
            min_val=1,
            max_val=10
        )

        self.model_timeout_seconds = self._get_env_int(
            "MODEL_TIMEOUT_SECONDS",
            300,
            min_val=30,
            max_val=1800
        )

        # LRU eviction settings
        self.protect_recently_used_minutes = self._get_env_int(
            "PROTECT_RECENTLY_USED_MINUTES",
            5,
            min_val=1,
            max_val=60
        )

        # Cache and performance settings
        self.hf_cache_dir = self._get_env_path(
            "HF_HOME",
            "/runpod-volume/cache/hf"
        )

        self.torch_cache_dir = self._get_env_path(
            "TORCH_HOME",
            "/runpod-volume/cache/torch"
        )

        # Logging and monitoring
        self.log_level = self._get_env_str(
            "LOG_LEVEL",
            "INFO",
            valid_values=["DEBUG", "INFO", "WARNING", "ERROR"]
        )

        self.memory_check_interval_seconds = self._get_env_int(
            "MEMORY_CHECK_INTERVAL",
            30,
            min_val=5,
            max_val=300
        )

        # Thread safety settings
        self.max_concurrent_loads = self._get_env_int(
            "MAX_CONCURRENT_LOADS",
            2,
            min_val=1,
            max_val=5
        )

        # Validate configuration
        self._validate_config()

    def _get_env_str(self, key: str, default: str, valid_values: Optional[list] = None) -> str:
        """Get string environment variable with validation."""
        value = os.getenv(key, default)

        if valid_values and value not in valid_values:
            raise ConfigurationError(
                key,
                f"Invalid value '{value}', must be one of: {valid_values}"
            )

        return value

    def _get_env_int(self, key: str, default: int, min_val: Optional[int] = None,
                     max_val: Optional[int] = None) -> int:
        """Get integer environment variable with validation."""
        try:
            value = int(os.getenv(key, str(default)))
        except ValueError:
            raise ConfigurationError(key, f"Must be a valid integer")

        if min_val is not None and value < min_val:
            raise ConfigurationError(key, f"Must be >= {min_val}")

        if max_val is not None and value > max_val:
            raise ConfigurationError(key, f"Must be <= {max_val}")

        return value

    def _get_env_path(self, key: str, default: str) -> Path:
        """Get path environment variable and convert to Path object."""
        path_str = os.getenv(key, default)
        return Path(path_str)

    def _validate_config(self) -> None:
        """Validate configuration consistency."""
        # Memory thresholds must be logical
        if self.memory_warning_percent >= self.memory_threshold_percent:
            raise ConfigurationError(
                "memory_thresholds",
                f"Warning threshold ({self.memory_warning_percent}%) must be < "
                f"eviction threshold ({self.memory_threshold_percent}%)"
            )

        # Cache directories should exist or be creatable (skip in test environment)
        if not os.getenv('PYTEST_CURRENT_TEST'):
            for cache_dir in [self.hf_cache_dir, self.torch_cache_dir]:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError):
                    # In development/test environments, skip directory creation errors
                    pass

    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get full path for a specific model."""
        return self.model_cache_dir / model_type / model_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "model_cache_dir": str(self.model_cache_dir),
            "memory_threshold_percent": self.memory_threshold_percent,
            "memory_warning_percent": self.memory_warning_percent,
            "max_models_in_memory": self.max_models_in_memory,
            "model_timeout_seconds": self.model_timeout_seconds,
            "protect_recently_used_minutes": self.protect_recently_used_minutes,
            "hf_cache_dir": str(self.hf_cache_dir),
            "torch_cache_dir": str(self.torch_cache_dir),
            "log_level": self.log_level,
            "memory_check_interval_seconds": self.memory_check_interval_seconds,
            "max_concurrent_loads": self.max_concurrent_loads,
        }


# Global configuration instance
config = ModelConfig()