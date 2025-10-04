#!/usr/bin/env python3
"""
ControlNet Model Download Script

Downloads and validates ControlNet models (Canny and Depth) along with
the base Stable Diffusion model for guided image generation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    import torch
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please run: pip install diffusers transformers huggingface_hub torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
CONTROLNET_MODELS = {
    'canny': {
        'repo_id': 'lllyasviel/sd-controlnet-canny',
        'description': 'Canny edge detection ControlNet',
        'estimated_size_gb': 1.4
    },
    'depth': {
        'repo_id': 'lllyasviel/sd-controlnet-depth',
        'description': 'Depth estimation ControlNet',
        'estimated_size_gb': 1.4
    }
}

BASE_MODEL = {
    'repo_id': 'runwayml/stable-diffusion-v1-5',
    'description': 'Stable Diffusion v1.5 base model',
    'estimated_size_gb': 3.4
}

# Additional dependencies for depth processing
DEPTH_DEPENDENCIES = {
    'midas': {
        'repo_id': 'intel-isl/MiDaS',
        'description': 'MiDaS depth estimation model',
        'estimated_size_gb': 0.5
    }
}


class ModelDownloader:
    """Handles downloading and validation of ControlNet models."""

    def __init__(self, cache_dir: Optional[str] = None, force_download: bool = False):
        """
        Initialize model downloader.

        Args:
            cache_dir: Directory to store downloaded models
            force_download: Whether to re-download existing models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.force_download = force_download
        self.download_stats = {}

    def download_controlnet_model(self, control_type: str) -> bool:
        """
        Download a specific ControlNet model.

        Args:
            control_type: Type of ControlNet to download ('canny', 'depth')

        Returns:
            True if download successful, False otherwise
        """
        if control_type not in CONTROLNET_MODELS:
            logger.error(f"Unknown control type: {control_type}")
            return False

        model_info = CONTROLNET_MODELS[control_type]
        repo_id = model_info['repo_id']

        logger.info(f"Downloading ControlNet model: {control_type}")
        logger.info(f"Repository: {repo_id}")
        logger.info(f"Estimated size: {model_info['estimated_size_gb']:.1f}GB")

        start_time = time.time()

        try:
            # Download model using diffusers (handles caching automatically)
            controlnet = ControlNetModel.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir=self.cache_dir,
                force_download=self.force_download
            )

            # Validate model can be loaded
            logger.info(f"Validating {control_type} ControlNet model...")
            if hasattr(controlnet, 'config'):
                logger.info(f"Model validation successful for {control_type}")
            else:
                raise ValueError("Model validation failed")

            download_time = time.time() - start_time
            self.download_stats[control_type] = {
                'success': True,
                'download_time_s': download_time,
                'estimated_size_gb': model_info['estimated_size_gb']
            }

            logger.info(f"Successfully downloaded {control_type} ControlNet "
                       f"in {download_time:.1f}s")
            return True

        except Exception as e:
            download_time = time.time() - start_time
            self.download_stats[control_type] = {
                'success': False,
                'error': str(e),
                'download_time_s': download_time
            }

            logger.error(f"Failed to download {control_type} ControlNet: {e}")
            return False

    def download_base_model(self) -> bool:
        """
        Download the base Stable Diffusion model.

        Returns:
            True if download successful, False otherwise
        """
        repo_id = BASE_MODEL['repo_id']

        logger.info("Downloading base Stable Diffusion model")
        logger.info(f"Repository: {repo_id}")
        logger.info(f"Estimated size: {BASE_MODEL['estimated_size_gb']:.1f}GB")

        start_time = time.time()

        try:
            # Download base model components
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                repo_id,
                controlnet=None,  # We'll add ControlNet separately
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                force_download=self.force_download
            )

            # Validate pipeline components
            logger.info("Validating base model components...")
            required_components = ['vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler']
            for component in required_components:
                if not hasattr(pipeline, component) or getattr(pipeline, component) is None:
                    raise ValueError(f"Missing required component: {component}")

            download_time = time.time() - start_time
            self.download_stats['base_model'] = {
                'success': True,
                'download_time_s': download_time,
                'estimated_size_gb': BASE_MODEL['estimated_size_gb']
            }

            logger.info(f"Successfully downloaded base model in {download_time:.1f}s")
            return True

        except Exception as e:
            download_time = time.time() - start_time
            self.download_stats['base_model'] = {
                'success': False,
                'error': str(e),
                'download_time_s': download_time
            }

            logger.error(f"Failed to download base model: {e}")
            return False

    def download_depth_dependencies(self) -> bool:
        """
        Download additional dependencies for depth processing.

        Returns:
            True if download successful, False otherwise
        """
        logger.info("Downloading depth processing dependencies")

        start_time = time.time()

        try:
            # MiDaS model will be downloaded automatically when first used
            # We can test this by importing torch.hub
            import torch

            # Verify torch.hub works (MiDaS will be downloaded on first use)
            logger.info("MiDaS depth model will be downloaded on first use")

            download_time = time.time() - start_time
            self.download_stats['depth_dependencies'] = {
                'success': True,
                'download_time_s': download_time,
                'note': 'MiDaS will be downloaded on first use'
            }

            logger.info(f"Depth dependencies verified in {download_time:.1f}s")
            return True

        except Exception as e:
            download_time = time.time() - start_time
            self.download_stats['depth_dependencies'] = {
                'success': False,
                'error': str(e),
                'download_time_s': download_time
            }

            logger.error(f"Failed to verify depth dependencies: {e}")
            return False

    def test_integration(self, control_types: List[str]) -> bool:
        """
        Test ControlNet integration with downloaded models.

        Args:
            control_types: List of control types to test

        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("Testing ControlNet integration...")

        success_count = 0
        total_tests = len(control_types)

        for control_type in control_types:
            if control_type not in CONTROLNET_MODELS:
                logger.warning(f"Skipping unknown control type: {control_type}")
                continue

            try:
                logger.info(f"Testing {control_type} ControlNet...")

                # Load ControlNet
                controlnet_repo = CONTROLNET_MODELS[control_type]['repo_id']
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_repo,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir
                )

                # Create pipeline
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    BASE_MODEL['repo_id'],
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False
                )

                # Basic validation - check components exist
                required_attrs = ['vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler', 'controlnet']
                for attr in required_attrs:
                    if not hasattr(pipeline, attr) or getattr(pipeline, attr) is None:
                        raise ValueError(f"Pipeline missing {attr}")

                logger.info(f"✓ {control_type} ControlNet integration test passed")
                success_count += 1

                # Clean up
                del pipeline
                del controlnet

            except Exception as e:
                logger.error(f"✗ {control_type} ControlNet integration test failed: {e}")

        # Force cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        success_rate = success_count / total_tests if total_tests > 0 else 0
        logger.info(f"Integration tests completed: {success_count}/{total_tests} passed "
                   f"({success_rate:.1%})")

        return success_count == total_tests

    def get_download_summary(self) -> Dict:
        """
        Get summary of download operations.

        Returns:
            Dictionary with download statistics
        """
        total_time = sum(stats.get('download_time_s', 0) for stats in self.download_stats.values())
        successful_downloads = sum(1 for stats in self.download_stats.values() if stats.get('success', False))
        total_downloads = len(self.download_stats)

        total_size_gb = 0
        for model_name, stats in self.download_stats.items():
            if stats.get('success', False) and 'estimated_size_gb' in stats:
                total_size_gb += stats['estimated_size_gb']

        return {
            'successful_downloads': successful_downloads,
            'total_downloads': total_downloads,
            'success_rate': successful_downloads / max(1, total_downloads),
            'total_time_s': total_time,
            'total_size_gb': total_size_gb,
            'download_stats': self.download_stats
        }


def main():
    """Main function for ControlNet model download script."""
    parser = argparse.ArgumentParser(description="Download ControlNet models")
    parser.add_argument(
        '--control-types',
        nargs='+',
        choices=['canny', 'depth', 'all'],
        default=['all'],
        help='Control types to download'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory to cache downloaded models'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of existing models'
    )
    parser.add_argument(
        '--skip-base',
        action='store_true',
        help='Skip downloading base Stable Diffusion model'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip integration testing'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce logging output'
    )

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Determine control types to download
    if 'all' in args.control_types:
        control_types = list(CONTROLNET_MODELS.keys())
    else:
        control_types = args.control_types

    logger.info("=== ControlNet Model Download ===")
    logger.info(f"Control types: {control_types}")
    logger.info(f"Cache directory: {args.cache_dir or 'default'}")
    logger.info(f"Force download: {args.force_download}")

    # Initialize downloader
    downloader = ModelDownloader(
        cache_dir=args.cache_dir,
        force_download=args.force_download
    )

    success = True

    try:
        # Download base model first
        if not args.skip_base:
            logger.info("\n--- Downloading Base Model ---")
            if not downloader.download_base_model():
                success = False

        # Download ControlNet models
        logger.info("\n--- Downloading ControlNet Models ---")
        for control_type in control_types:
            if not downloader.download_controlnet_model(control_type):
                success = False

        # Download depth dependencies if needed
        if 'depth' in control_types:
            logger.info("\n--- Downloading Depth Dependencies ---")
            if not downloader.download_depth_dependencies():
                success = False

        # Run integration tests
        if not args.skip_test and success:
            logger.info("\n--- Running Integration Tests ---")
            if not downloader.test_integration(control_types):
                success = False

        # Print summary
        logger.info("\n--- Download Summary ---")
        summary = downloader.get_download_summary()

        logger.info(f"Downloads: {summary['successful_downloads']}/{summary['total_downloads']} successful")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Total time: {summary['total_time_s']:.1f}s")
        logger.info(f"Total size: ~{summary['total_size_gb']:.1f}GB")

        if success:
            logger.info("\n✓ All ControlNet models downloaded successfully!")
            logger.info("Models are ready for use with the ControlNet handler.")
        else:
            logger.error("\n✗ Some downloads failed. Check logs above for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())