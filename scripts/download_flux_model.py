#!/usr/bin/env python3
"""
FLUX.1 Schnell Model Download Script

Downloads and validates the FLUX.1 Schnell fp8 model for text-to-image generation.
Provides options for local storage, validation, and progress tracking.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import hashlib
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download, snapshot_download, HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FluxModelDownloader:
    """Downloads and validates FLUX.1 Schnell fp8 model."""

    # Model configuration
    MODEL_ID = "black-forest-labs/FLUX.1-schnell"
    MODEL_VARIANT = "fp8"

    # Expected model files (key files for validation)
    EXPECTED_FILES = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json"
    ]

    # Large model files that need special handling
    LARGE_MODEL_FILES = [
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors"
    ]

    def __init__(self, download_dir: Optional[Path] = None, use_auth_token: bool = False):
        """
        Initialize FLUX.1 model downloader.

        Args:
            download_dir: Directory to download models to
            use_auth_token: Whether to use HuggingFace authentication
        """
        self.download_dir = download_dir or Path.home() / ".cache" / "flux_models"
        self.use_auth_token = use_auth_token
        self.api = HfApi()

        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FLUX.1 downloader, target directory: {self.download_dir}")

    def check_disk_space(self, required_gb: float = 20.0) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            required_gb: Required disk space in GB

        Returns:
            True if sufficient space available
        """
        try:
            statvfs = os.statvfs(self.download_dir)
            available_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)

            logger.info(f"Available disk space: {available_gb:.1f} GB (required: {required_gb} GB)")

            if available_gb < required_gb:
                logger.error(f"Insufficient disk space: {available_gb:.1f} GB available, {required_gb} GB required")
                return False

            return True

        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume sufficient space if check fails

    def check_model_exists(self) -> bool:
        """
        Check if FLUX.1 model repository exists and is accessible.

        Returns:
            True if model repository is accessible
        """
        try:
            # Try to get repository info
            repo_info = self.api.repo_info(
                repo_id=self.MODEL_ID,
                token=HfFolder.get_token() if self.use_auth_token else None
            )

            logger.info(f"Found FLUX.1 repository: {repo_info.id}")
            logger.info(f"Repository size: {repo_info.siblings_count} files")

            return True

        except RepositoryNotFoundError:
            logger.error(f"Repository not found: {self.MODEL_ID}")
            return False
        except GatedRepoError:
            logger.error(f"Repository is gated and requires authentication: {self.MODEL_ID}")
            logger.error("Please login with: huggingface-cli login")
            return False
        except Exception as e:
            logger.error(f"Error accessing repository: {e}")
            return False

    def download_model(self,
                      resume_download: bool = True,
                      force_download: bool = False,
                      local_files_only: bool = False) -> bool:
        """
        Download the complete FLUX.1 Schnell fp8 model.

        Args:
            resume_download: Resume interrupted downloads
            force_download: Force re-download even if files exist
            local_files_only: Only use local files (no downloading)

        Returns:
            True if download successful
        """
        try:
            logger.info(f"Starting FLUX.1 Schnell model download...")

            # Check repository accessibility (unless using local files only)
            if not local_files_only and not self.check_model_exists():
                return False

            # Check disk space
            if not self.check_disk_space():
                return False

            # Set up download parameters
            download_kwargs = {
                'repo_id': self.MODEL_ID,
                'cache_dir': str(self.download_dir),
                'resume_download': resume_download,
                'force_download': force_download,
                'local_files_only': local_files_only,
                'token': HfFolder.get_token() if self.use_auth_token else None
            }

            # Add variant filter for fp8
            if self.MODEL_VARIANT:
                download_kwargs['allow_patterns'] = [
                    f"*{self.MODEL_VARIANT}*",  # fp8 variant files
                    "*.json",  # Configuration files
                    "*.txt",   # Text files
                    "tokenizer/*",  # Tokenizer files
                    "scheduler/*",  # Scheduler files
                ]
                download_kwargs['ignore_patterns'] = [
                    "*bf16*",  # Ignore other precision variants
                    "*fp16*",
                    "*fp32*",
                    "*.bin",   # Ignore .bin files (prefer .safetensors)
                ]

            logger.info("Downloading FLUX.1 Schnell fp8 model (this may take a while)...")

            # Perform snapshot download
            model_path = snapshot_download(**download_kwargs)

            logger.info(f"FLUX.1 model downloaded to: {model_path}")

            # Validate downloaded model
            if self.validate_model(Path(model_path)):
                logger.info("FLUX.1 model validation passed")
                return True
            else:
                logger.error("FLUX.1 model validation failed")
                return False

        except Exception as e:
            logger.error(f"FLUX.1 model download failed: {e}")
            return False

    def validate_model(self, model_path: Path) -> bool:
        """
        Validate downloaded FLUX.1 model files.

        Args:
            model_path: Path to downloaded model directory

        Returns:
            True if validation passes
        """
        try:
            logger.info(f"Validating FLUX.1 model at: {model_path}")

            # Check if directory exists
            if not model_path.exists() or not model_path.is_dir():
                logger.error(f"Model directory not found: {model_path}")
                return False

            # Check for expected configuration files
            missing_files = []
            for expected_file in self.EXPECTED_FILES:
                file_path = model_path / expected_file
                if not file_path.exists():
                    missing_files.append(expected_file)

            if missing_files:
                logger.error(f"Missing required files: {missing_files}")
                return False

            # Check for at least some large model files
            found_model_files = []
            for model_file in self.LARGE_MODEL_FILES:
                file_path = model_path / model_file
                if file_path.exists():
                    found_model_files.append(model_file)
                    logger.debug(f"Found model file: {model_file} ({self._format_size(file_path.stat().st_size)})")

            if not found_model_files:
                logger.error("No large model files found (transformer, vae, text_encoder)")
                return False

            logger.info(f"Found {len(found_model_files)} model files")

            # Validate model index configuration
            model_index_path = model_path / "model_index.json"
            if model_index_path.exists():
                import json
                try:
                    with open(model_index_path, 'r') as f:
                        model_config = json.load(f)

                    logger.info(f"Model class: {model_config.get('_class_name', 'Unknown')}")
                    logger.info(f"Model library: {model_config.get('_diffusers_version', 'Unknown')}")

                except Exception as e:
                    logger.warning(f"Could not parse model index: {e}")

            # Calculate total model size
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            logger.info(f"Total model size: {self._format_size(total_size)}")

            # Validation passed
            logger.info("FLUX.1 model validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def test_model_loading(self, model_path: Path) -> bool:
        """
        Test loading the FLUX.1 model to verify it works.

        Args:
            model_path: Path to model directory

        Returns:
            True if model loads successfully
        """
        try:
            logger.info("Testing FLUX.1 model loading...")

            # Import required libraries
            from diffusers import FluxPipeline

            # Try to load the pipeline
            pipeline = FluxPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                variant=self.MODEL_VARIANT,
                device_map="auto",
                use_safetensors=True
            )

            logger.info("FLUX.1 pipeline loaded successfully")

            # Clean up to free memory
            del pipeline
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return True

        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False

    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive information about the downloaded model.

        Args:
            model_path: Path to model directory

        Returns:
            Dictionary with model information
        """
        info = {
            'model_id': self.MODEL_ID,
            'variant': self.MODEL_VARIANT,
            'path': str(model_path),
            'exists': model_path.exists(),
            'files': [],
            'total_size_bytes': 0,
            'total_size_gb': 0.0
        }

        if model_path.exists():
            # List all files and calculate sizes
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        info['files'].append({
                            'name': str(file_path.relative_to(model_path)),
                            'size_bytes': size,
                            'size_formatted': self._format_size(size)
                        })
                        info['total_size_bytes'] += size

            info['total_size_gb'] = info['total_size_bytes'] / (1024 ** 3)
            info['file_count'] = len(info['files'])

        return info

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def cleanup_incomplete_downloads(self) -> None:
        """Clean up any incomplete download files."""
        try:
            # Look for .tmp files and incomplete downloads
            tmp_files = list(self.download_dir.glob("**/*.tmp"))
            incomplete_files = list(self.download_dir.glob("**/*.incomplete"))

            cleanup_files = tmp_files + incomplete_files

            if cleanup_files:
                logger.info(f"Cleaning up {len(cleanup_files)} incomplete download files")
                for file_path in cleanup_files:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download and validate FLUX.1 Schnell fp8 model"
    )

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory to download model to (default: ~/.cache/flux_models)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume interrupted downloads"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing model, don't download"
    )

    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test loading the model after download/validation"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about downloaded model"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up incomplete download files"
    )

    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use HuggingFace authentication token"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize downloader
    downloader = FluxModelDownloader(
        download_dir=args.download_dir,
        use_auth_token=args.auth
    )

    # Handle cleanup
    if args.cleanup:
        logger.info("Cleaning up incomplete downloads...")
        downloader.cleanup_incomplete_downloads()
        return 0

    # Determine model path
    model_path = downloader.download_dir / "models--black-forest-labs--FLUX.1-schnell"

    # Handle info request
    if args.info:
        logger.info("Getting model information...")
        info = downloader.get_model_info(model_path)
        print(f"\nFLUX.1 Model Information:")
        print(f"Model ID: {info['model_id']}")
        print(f"Variant: {info['variant']}")
        print(f"Path: {info['path']}")
        print(f"Exists: {info['exists']}")
        if info['exists']:
            print(f"Files: {info['file_count']}")
            print(f"Total size: {info['total_size_gb']:.2f} GB")
        return 0

    # Handle validation only
    if args.validate_only:
        if model_path.exists():
            logger.info("Validating existing model...")
            success = downloader.validate_model(model_path)
            return 0 if success else 1
        else:
            logger.error("No model found to validate")
            return 1

    # Download model
    logger.info("Starting FLUX.1 Schnell model download...")

    success = downloader.download_model(
        resume_download=not args.no_resume,
        force_download=args.force,
        local_files_only=False
    )

    if not success:
        logger.error("Model download failed")
        return 1

    # Test loading if requested
    if args.test_loading:
        logger.info("Testing model loading...")
        if not downloader.test_model_loading(model_path):
            logger.error("Model loading test failed")
            return 1
        logger.info("Model loading test passed")

    logger.info("FLUX.1 Schnell model download completed successfully!")

    # Show model info
    info = downloader.get_model_info(model_path)
    print(f"\nDownload Summary:")
    print(f"Model Path: {info['path']}")
    print(f"Total Size: {info['total_size_gb']:.2f} GB")
    print(f"Files: {info['file_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())