#!/usr/bin/env python3
"""
Multi-Modal Inference Worker Model Download Script

Downloads and validates AI models for the multi-modal inference worker
while respecting storage constraints and ensuring model integrity.

Usage:
    python download_models.py --target-size=40GB --cache-dir=/runpod-volume/models
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
import torch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Manages model downloads with size constraints and validation."""

    def __init__(
        self,
        cache_dir: str = "/runpod-volume/models",
        target_size_gb: int = 40,
        validation_mode: str = "basic"
    ):
        self.cache_dir = Path(cache_dir)
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        self.validation_mode = validation_mode
        self.downloaded_size = 0
        self.model_manifest = {}

        # Create cache directory structure
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ModelDownloader:")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Target size limit: {target_size_gb}GB")
        logger.info(f"  Validation mode: {validation_mode}")

    def get_model_config(self) -> Dict[str, Dict]:
        """Define model configuration for MVP deployment."""
        return {
            "flux": {
                "repo_id": "black-forest-labs/FLUX.1-schnell",
                "variant": "fp8",
                "estimated_size_gb": 15,
                "local_dir": "flux",
                "priority": 1,
                "files": [
                    "flux1-schnell.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder_2/config.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer_2/tokenizer_config.json",
                    "vae/config.json"
                ]
            },
            "controlnet_canny": {
                "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
                "estimated_size_gb": 2,
                "local_dir": "controlnet/canny",
                "priority": 2
            },
            "controlnet_depth": {
                "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
                "estimated_size_gb": 2,
                "local_dir": "controlnet/depth",
                "priority": 3
            },
            "animatediff": {
                "repo_id": "guoyww/animatediff-motion-adapter-v1-5-2",
                "estimated_size_gb": 2,
                "local_dir": "animatediff",
                "priority": 4
            },
            "ltx_video": {
                "repo_id": "Lightricks/LTX-Video",
                "revision": "2b-distilled",
                "estimated_size_gb": 8,
                "local_dir": "video_backbones/ltx-2b",
                "priority": 5
            },
            "sdxl_inpaint": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "subfolder": "inpainting",
                "estimated_size_gb": 6,
                "local_dir": "inpaint/sdxl",
                "priority": 6
            },
            "camera_ctrl": {
                "repo_id": "hehao13/CameraCtrl",
                "estimated_size_gb": 1,
                "local_dir": "camera/camctrllib",
                "priority": 7
            }
        }

    def calculate_current_size(self) -> int:
        """Calculate current size of downloaded models."""
        total_size = 0
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size

    def format_size(self, size_bytes: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def check_model_exists(self, model_name: str, config: Dict) -> bool:
        """Check if model already exists locally."""
        local_path = self.cache_dir / config["local_dir"]

        if not local_path.exists():
            return False

        # Check if essential files exist
        if "files" in config:
            for file_name in config["files"]:
                if not (local_path / file_name).exists():
                    logger.info(f"Missing file {file_name} for {model_name}")
                    return False

        # Basic size check
        model_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        expected_size = config["estimated_size_gb"] * 1024 * 1024 * 1024

        # Allow 20% variance in size
        if model_size < expected_size * 0.8:
            logger.info(f"Model {model_name} appears incomplete (size: {self.format_size(model_size)})")
            return False

        logger.info(f"Model {model_name} already exists ({self.format_size(model_size)})")
        return True

    def download_model(self, model_name: str, config: Dict, progress_bar: bool = True) -> bool:
        """Download a single model with progress tracking."""
        logger.info(f"Downloading {model_name}...")

        # Check size constraints
        estimated_size = config["estimated_size_gb"] * 1024 * 1024 * 1024
        if self.downloaded_size + estimated_size > self.target_size_bytes:
            logger.warning(f"Skipping {model_name} - would exceed size limit")
            return False

        try:
            local_dir = self.cache_dir / config["local_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)

            download_kwargs = {
                "repo_id": config["repo_id"],
                "local_dir": str(local_dir),
                "resume_download": True,
                "local_dir_use_symlinks": False
            }

            # Add optional parameters
            if "revision" in config:
                download_kwargs["revision"] = config["revision"]
            if "subfolder" in config:
                download_kwargs["subfolder"] = config["subfolder"]

            # Download with progress bar
            if progress_bar:
                logger.info(f"Downloading {model_name} to {local_dir}")

            downloaded_path = snapshot_download(**download_kwargs)

            # Calculate actual downloaded size
            actual_size = sum(f.stat().st_size for f in Path(downloaded_path).rglob("*") if f.is_file())
            self.downloaded_size += actual_size

            # Update manifest
            self.model_manifest[model_name] = {
                "repo_id": config["repo_id"],
                "local_path": str(local_dir),
                "size_bytes": actual_size,
                "size_formatted": self.format_size(actual_size),
                "status": "downloaded"
            }

            logger.info(f"Downloaded {model_name}: {self.format_size(actual_size)}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            self.model_manifest[model_name] = {
                "repo_id": config["repo_id"],
                "status": "failed",
                "error": str(e)
            }
            return False

    def validate_model(self, model_name: str, config: Dict) -> bool:
        """Validate downloaded model integrity."""
        local_dir = self.cache_dir / config["local_dir"]

        if not local_dir.exists():
            return False

        try:
            # Check if PyTorch models can be loaded (basic validation)
            safetensors_files = list(local_dir.rglob("*.safetensors"))

            if safetensors_files and self.validation_mode == "strict":
                logger.info(f"Validating {model_name} tensors...")
                for tensor_file in safetensors_files[:1]:  # Validate first file only for speed
                    try:
                        # Basic tensor load test
                        import safetensors
                        with safetensors.safe_open(str(tensor_file), framework="pt") as f:
                            keys = f.keys()
                            if len(keys) == 0:
                                return False
                    except Exception as e:
                        logger.warning(f"Tensor validation failed for {tensor_file}: {e}")
                        return False

            logger.info(f"Model {model_name} validated successfully")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {str(e)}")
            return False

    def download_all_models(self, progress_bar: bool = True) -> None:
        """Download all models respecting size constraints and priorities."""
        models = self.get_model_config()

        # Sort by priority
        sorted_models = sorted(models.items(), key=lambda x: x[1].get("priority", 999))

        logger.info(f"Starting model downloads (target: {self.format_size(self.target_size_bytes)})")

        # Calculate existing size
        self.downloaded_size = self.calculate_current_size()
        logger.info(f"Current size: {self.format_size(self.downloaded_size)}")

        for model_name, config in sorted_models:
            # Check if model already exists
            if self.check_model_exists(model_name, config):
                model_size = sum(f.stat().st_size for f in (self.cache_dir / config["local_dir"]).rglob("*") if f.is_file())
                self.model_manifest[model_name] = {
                    "repo_id": config["repo_id"],
                    "local_path": str(self.cache_dir / config["local_dir"]),
                    "size_bytes": model_size,
                    "size_formatted": self.format_size(model_size),
                    "status": "existing"
                }
                continue

            # Download model
            if self.download_model(model_name, config, progress_bar):
                # Validate if requested
                if self.validation_mode in ["basic", "strict"]:
                    if not self.validate_model(model_name, config):
                        logger.warning(f"Model {model_name} failed validation")
                        self.model_manifest[model_name]["status"] = "validation_failed"

        # Save manifest
        self.save_manifest()

        # Report final status
        final_size = self.calculate_current_size()
        logger.info(f"Download complete. Total size: {self.format_size(final_size)}")

        if final_size > self.target_size_bytes:
            logger.warning(f"Total size exceeds target by {self.format_size(final_size - self.target_size_bytes)}")

    def save_manifest(self) -> None:
        """Save download manifest to JSON file."""
        manifest_path = self.cache_dir / "models_manifest.json"

        manifest_data = {
            "total_size_bytes": self.calculate_current_size(),
            "total_size_formatted": self.format_size(self.calculate_current_size()),
            "target_size_bytes": self.target_size_bytes,
            "download_date": None,  # Will be set by datetime
            "models": self.model_manifest
        }

        # Add timestamp
        from datetime import datetime
        manifest_data["download_date"] = datetime.now().isoformat()

        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Manifest saved to {manifest_path}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Download models for multi-modal inference worker")

    parser.add_argument(
        "--target-size",
        default="40GB",
        help="Target size limit (e.g., 40GB, 50GB)"
    )
    parser.add_argument(
        "--cache-dir",
        default="/runpod-volume/models",
        help="Directory to store downloaded models"
    )
    parser.add_argument(
        "--validation-mode",
        choices=["none", "basic", "strict"],
        default="basic",
        help="Model validation level"
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Show progress bars during download"
    )

    args = parser.parse_args()

    # Parse target size
    target_size_str = args.target_size.upper()
    if target_size_str.endswith("GB"):
        target_size_gb = int(target_size_str[:-2])
    else:
        target_size_gb = int(target_size_str)

    # Initialize downloader
    downloader = ModelDownloader(
        cache_dir=args.cache_dir,
        target_size_gb=target_size_gb,
        validation_mode=args.validation_mode
    )

    # Download all models
    try:
        downloader.download_all_models(progress_bar=args.progress_bar)
        logger.info("Model download completed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()