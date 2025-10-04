#!/usr/bin/env python3
"""
Simple validation script for the model management framework.

Tests basic functionality of ModelManager, MemoryMonitor, and BaseModel
to ensure the framework is working correctly.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ModelManager, MemoryMonitor, BaseModel
from src.utils.exceptions import ModelLoadError

class SimpleTestModel(BaseModel):
    """Simple test model implementation."""

    def __init__(self, model_name: str, model_path: Path, priority: int = 50):
        super().__init__(model_name, model_path, priority)
        self._loaded_data = None

    def load(self) -> None:
        """Load the model (simulate with sleep)."""
        print(f"Loading {self.model_name}...")
        time.sleep(0.1)  # Simulate load time

        from datetime import datetime
        self.is_loaded = True
        self.load_time = datetime.now()
        self.memory_usage_mb = 150
        self._loaded_data = f"data_for_{self.model_name}"
        print(f"✓ {self.model_name} loaded successfully")

    def unload(self) -> None:
        """Unload the model."""
        print(f"Unloading {self.model_name}...")
        self.is_loaded = False
        self.memory_usage_mb = 0
        self._loaded_data = None
        print(f"✓ {self.model_name} unloaded")

    def infer(self, inputs):
        """Perform inference."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        self.mark_used()
        return {
            "model": self.model_name,
            "result": f"processed {inputs.get('text', 'input')} with {self._loaded_data}",
            "memory_mb": self.memory_usage_mb
        }

    def get_memory_usage(self) -> int:
        return self.memory_usage_mb

    def validate_inputs(self, inputs) -> bool:
        return isinstance(inputs, dict)


def test_memory_monitor():
    """Test memory monitor functionality."""
    print("\n🧪 Testing MemoryMonitor...")

    monitor = MemoryMonitor()

    # Get memory stats
    stats = monitor.get_current_stats()
    print(f"✓ Memory stats: System {stats.system_used_mb}/{stats.system_total_mb}MB")
    print(f"✓ GPU available: {'Yes' if stats.gpu_total_mb > 0 else 'No (expected in test environment)'}")

    # Test capacity checking
    can_load_100mb = monitor.can_load_model(100)
    can_load_huge = monitor.can_load_model(999999)

    print(f"✓ Can load 100MB: {can_load_100mb}")
    print(f"✓ Can load 999GB: {can_load_huge} (should be False)")

    assert can_load_100mb == True, "Should be able to load 100MB"
    assert can_load_huge == False, "Should not be able to load 999GB"


def test_model_manager():
    """Test model manager functionality."""
    print("\n🧪 Testing ModelManager...")

    manager = ModelManager()

    # Create temporary model paths
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path_1 = Path(temp_dir) / "model_1"
        model_path_2 = Path(temp_dir) / "model_2"
        model_path_1.mkdir()
        model_path_2.mkdir()

        # Register models
        print("Registering models...")
        manager.register_model(
            model_name="test_model_1",
            model_class=SimpleTestModel,
            model_path=model_path_1,
            priority=70
        )

        manager.register_model(
            model_name="test_model_2",
            model_class=SimpleTestModel,
            model_path=model_path_2,
            priority=30
        )

        # Check registration
        registered = manager.get_registered_models()
        print(f"✓ Registered models: {registered}")

        assert "test_model_1" in registered
        assert "test_model_2" in registered

        # Load and use models
        print("\nTesting model loading and inference...")
        model_1 = manager.get_model("test_model_1")

        assert model_1.is_loaded
        print(f"✓ Model 1 loaded: {model_1.model_name}")

        # Test inference
        result = model_1.infer({"text": "hello world"})
        print(f"✓ Inference result: {result}")

        assert "result" in result
        assert "test_model_1" in result["model"]

        # Load second model
        model_2 = manager.get_model("test_model_2")
        assert model_2.is_loaded
        print(f"✓ Model 2 loaded: {model_2.model_name}")

        # Check status
        status = manager.get_manager_status()
        print(f"✓ Manager status: {status['loaded_count']} models loaded")

        assert status["loaded_count"] >= 1

        # Test eviction
        print("\nTesting model eviction...")
        evicted = manager.evict_model("test_model_1")
        assert evicted
        assert not model_1.is_loaded
        print(f"✓ Model 1 evicted successfully")

        # Clear all
        manager.clear_all_models()
        final_status = manager.get_manager_status()
        print(f"✓ All models cleared: {final_status['loaded_count']} models remaining")

        assert final_status["loaded_count"] == 0


def main():
    """Run all validation tests."""
    print("🚀 Model Management Framework Validation")
    print("=" * 50)

    try:
        test_memory_monitor()
        test_model_manager()

        print("\n" + "=" * 50)
        print("✅ All validation tests passed!")
        print("🎉 Model Management Framework is working correctly")
        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)