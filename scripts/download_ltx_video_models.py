#!/usr/bin/env python3
"""
LTX-Video Model Download Script

Downloads and validates LTX-Video text-to-video generation models
for the multi-modal inference worker.

Usage:
    python download_ltx_video_models.py --cache-dir=/runpod-volume/models
    python download_ltx_video_models.py --validate-only  # Only validate existing models
    python download_ltx_video_models.py --force-download  # Force redownload
"""

import os
import sys
import argparse
import logging
import json
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    import torch
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install torch diffusers huggingface_hub")
    sys.exit(1)

# Try to import LTXVideoPipeline, fall back to generic validation
try:
    from diffusers import LTXVideoPipeline
    HAS_LTX_PIPELINE = True
except ImportError:
    HAS_LTX_PIPELINE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LTXVideoModelDownloader:
    """Manages LTX-Video model downloads with validation and integrity checks."""

    def __init__(
        self,
        cache_dir: str = "/runpod-volume/models",
        validation_mode: str = "strict"
    ):
        self.cache_dir = Path(cache_dir)
        self.validation_mode = validation_mode
        self.models_dir = self.cache_dir / "video_backbones"
        self.ltx_dir = self.models_dir / "ltx-2b"

        # Create directory structure
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized LTX-Video Model Downloader:")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  LTX-Video directory: {self.ltx_dir}")
        logger.info(f"  Validation mode: {validation_mode}")

        if not HAS_LTX_PIPELINE:
            logger.warning("LTXVideoPipeline not available, using basic validation only")

    def get_ltx_video_config(self) -> Dict[str, Any]:
        """Get LTX-Video model configuration."""
        return {
            "ltx_video": {
                "repo_id": "Lightricks/LTX-Video",
                "revision": "2b-distilled",  # Use distilled 2B model for efficiency
                "estimated_size_gb": 8.5,
                "local_dir": self.ltx_dir,
                "description": "LTX-Video 2B text-to-video diffusion transformer",
                "required_files": [
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/pytorch_model.bin",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "tokenizer/merges.txt",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "model_index.json"
                ],
                "optional_files": [
                    "README.md",
                    "feature_extractor/preprocessor_config.json",
                    ".gitattributes"
                ]
            }
        }

    def check_storage_space(self, required_gb: float) -> bool:
        """Check if there's enough storage space."""
        try:
            statvfs = os.statvfs(str(self.cache_dir))
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

            logger.info(f"Available storage space: {free_space_gb:.2f} GB")
            logger.info(f"Required space: {required_gb:.2f} GB")

            if free_space_gb < required_gb + 1:  # +1GB buffer
                logger.error(f"Insufficient storage space! Need {required_gb:.2f}GB, have {free_space_gb:.2f}GB")
                return False
            return True
        except Exception as e:
            logger.warning(f"Could not check storage space: {e}")
            return True  # Proceed if we can't check

    def validate_model_files(self, config: Dict[str, Any]) -> bool:
        """Validate that all required model files are present and valid."""
        model_config = config["ltx_video"]
        local_dir = Path(model_config["local_dir"])

        if not local_dir.exists():
            logger.error(f"Model directory does not exist: {local_dir}")
            return False

        # Check required files
        missing_files = []
        for file_path in model_config["required_files"]:
            full_path = local_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            elif full_path.stat().st_size == 0:
                missing_files.append(f"{file_path} (empty)")

        if missing_files:
            logger.error(f"Missing or empty required files: {missing_files}")
            return False

        # Validate key model files
        try:
            # Check transformer config
            config_path = local_dir / "transformer/config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    transformer_config = json.load(f)
                    if "LTXVideoTransformer2D" not in str(transformer_config.get("_class_name", "")):
                        logger.warning("Transformer config may not be for LTX-Video model")

            # Check model index
            index_path = local_dir / "model_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    model_index = json.load(f)
                    if "ltx_video" not in str(model_index).lower():
                        logger.warning("Model index may not be for LTX-Video")

            logger.info("✅ Model file validation passed")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def test_model_loading(self, config: Dict[str, Any]) -> bool:
        """Test that the model can be loaded properly."""
        if self.validation_mode != "strict":
            logger.info("Skipping model loading test (validation_mode != strict)")
            return True

        if not HAS_LTX_PIPELINE:
            logger.info("LTXVideoPipeline not available, skipping model loading test")
            return True

        model_config = config["ltx_video"]
        local_dir = Path(model_config["local_dir"])

        try:
            logger.info("Testing model loading (this may take a few minutes)...")

            # Test loading with diffusers pipeline
            pipe = LTXVideoPipeline.from_pretrained(
                str(local_dir),
                torch_dtype=torch.float16,
                device_map=None  # Don't load to GPU for validation
            )

            # Basic model structure checks
            assert hasattr(pipe, 'transformer'), "Model missing transformer component"
            assert hasattr(pipe, 'vae'), "Model missing VAE component"
            assert hasattr(pipe, 'text_encoder'), "Model missing text encoder"
            assert hasattr(pipe, 'scheduler'), "Model missing scheduler"

            # Check transformer parameters
            param_count = sum(p.numel() for p in pipe.transformer.parameters())
            logger.info(f"Transformer parameters: {param_count:,}")

            # Expected parameter range for 2B model (allowing some variance)
            if param_count < 1.5e9 or param_count > 3e9:
                logger.warning(f"Unexpected parameter count: {param_count:,} (expected ~2B)")

            logger.info("✅ Model loading test passed")
            return True

        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False

    def download_ltx_video_model(self, config: Dict[str, Any], force: bool = False) -> bool:
        """Download LTX-Video model from HuggingFace."""
        model_config = config["ltx_video"]

        # Check if already downloaded
        if not force and self.validate_model_files(config):
            logger.info("✅ LTX-Video model already downloaded and validated")
            return True

        # Check storage space
        if not self.check_storage_space(model_config["estimated_size_gb"]):
            return False

        try:
            logger.info(f"Downloading LTX-Video model from {model_config['repo_id']}...")
            logger.info(f"Revision: {model_config['revision']}")
            logger.info(f"Target directory: {model_config['local_dir']}")

            # Download model using HuggingFace Hub
            downloaded_path = snapshot_download(
                repo_id=model_config["repo_id"],
                revision=model_config["revision"],
                local_dir=str(model_config["local_dir"]),
                local_dir_use_symlinks=False,
                resume_download=True,
                cache_dir=str(self.cache_dir / ".cache"),
                # Don't include git lfs files that are too large
                ignore_patterns=["*.bin"] if not force else None
            )

            logger.info(f"✅ Download completed: {downloaded_path}")

            # Validate downloaded model
            if not self.validate_model_files(config):
                logger.error("Downloaded model failed validation")
                return False

            # Test model loading if in strict mode
            if not self.test_model_loading(config):
                logger.error("Downloaded model failed loading test")
                return False

            # Create manifest file
            self._create_manifest(config)

            logger.info("✅ LTX-Video model download and validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False

    def _create_manifest(self, config: Dict[str, Any]):
        """Create manifest file with model information."""
        model_config = config["ltx_video"]
        manifest_path = Path(model_config["local_dir"]) / "download_manifest.json"

        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(model_config["local_dir"]):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size

        manifest = {
            "model_name": "LTX-Video",
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "download_date": str(datetime.now().isoformat()),
            "total_size_gb": round(total_size / (1024**3), 2),
            "files_count": len(model_config["required_files"]),
            "validation_passed": True
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created manifest: {manifest_path}")

    def cleanup_cache(self):
        """Clean up temporary download cache."""
        cache_path = self.cache_dir / ".cache"
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cleaned up cache: {cache_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about downloaded models."""
        config = self.get_ltx_video_config()
        model_config = config["ltx_video"]
        local_dir = Path(model_config["local_dir"])

        info = {
            "model_name": "LTX-Video",
            "status": "not_downloaded",
            "path": str(local_dir),
            "size_gb": 0,
            "files_present": False
        }

        if local_dir.exists():
            # Calculate total size
            total_size = 0
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size

            info["size_gb"] = round(total_size / (1024**3), 2)
            info["files_present"] = self.validate_model_files(config)
            info["status"] = "downloaded" if info["files_present"] else "incomplete"

            # Check manifest
            manifest_path = local_dir / "download_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    info["manifest"] = manifest

        return info


def main():
    parser = argparse.ArgumentParser(description="Download LTX-Video models")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/runpod-volume/models",
        help="Model cache directory"
    )
    parser.add_argument(
        "--validation-mode",
        choices=["basic", "strict"],
        default="strict",
        help="Validation mode: basic (file checks) or strict (includes loading test)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing models, don't download"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload even if models exist"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit"
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Clean up download cache"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = LTXVideoModelDownloader(
        cache_dir=args.cache_dir,
        validation_mode=args.validation_mode
    )

    # Handle info request
    if args.info:
        info = downloader.get_model_info()
        print(json.dumps(info, indent=2))
        return 0

    # Handle cache cleanup
    if args.cleanup_cache:
        downloader.cleanup_cache()
        return 0

    # Get model configuration
    config = downloader.get_ltx_video_config()

    # Validate existing models
    if args.validate_only:
        if downloader.validate_model_files(config):
            if downloader.test_model_loading(config):
                logger.info("✅ All validations passed")
                return 0
            else:
                logger.error("❌ Model loading validation failed")
                return 1
        else:
            logger.error("❌ File validation failed")
            return 1

    # Download models
    success = downloader.download_ltx_video_model(config, force=args.force_download)

    if success:
        logger.info("🎉 LTX-Video model download completed successfully!")
        return 0
    else:
        logger.error("💥 LTX-Video model download failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())