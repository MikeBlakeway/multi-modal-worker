#!/usr/bin/env python3
"""
AnimateDiff Model Download Script

Downloads and validates AnimateDiff motion adapter models and base diffusion models
for image-to-video generation. Supports both local development and production deployment.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from diffusers import AnimateDiffPipeline, MotionAdapter
    import torch
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Install with: pip install diffusers transformers torch huggingface_hub")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnimateDiffDownloader:
    """
    Handles downloading and validation of AnimateDiff models.
    """

    # Default model configurations
    DEFAULT_MODELS = {
        'motion_adapter': 'guoyww/animatediff-motion-adapter-v1-5-2',
        'base_model': 'runwayml/stable-diffusion-v1-5'
    }

    # Alternative model options
    ALTERNATIVE_MODELS = {
        'motion_adapter': [
            'guoyww/animatediff-motion-adapter-v1-5-2',
            'guoyww/animatediff-motion-adapter-v1-5'
        ],
        'base_model': [
            'runwayml/stable-diffusion-v1-5',
            'stabilityai/stable-diffusion-2-1-base'
        ]
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            cache_dir: Custom cache directory for models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.downloaded_models = []

    def download_motion_adapter(self, model_id: str = None) -> str:
        """
        Download AnimateDiff motion adapter.

        Args:
            model_id: HuggingFace model ID for motion adapter

        Returns:
            Path to downloaded model

        Raises:
            Exception: If download fails
        """
        model_id = model_id or self.DEFAULT_MODELS['motion_adapter']

        logger.info(f"Downloading motion adapter: {model_id}")

        try:
            start_time = time.time()

            # Download motion adapter using diffusers
            motion_adapter = MotionAdapter.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16
            )

            download_time = time.time() - start_time
            logger.info(f"Motion adapter downloaded successfully in {download_time:.1f}s")

            # Validate the motion adapter
            self._validate_motion_adapter(motion_adapter, model_id)

            self.downloaded_models.append({
                'type': 'motion_adapter',
                'model_id': model_id,
                'download_time': download_time,
                'status': 'success'
            })

            return model_id

        except Exception as e:
            logger.error(f"Failed to download motion adapter {model_id}: {e}")
            self.downloaded_models.append({
                'type': 'motion_adapter',
                'model_id': model_id,
                'error': str(e),
                'status': 'failed'
            })
            raise

    def download_base_model(self, model_id: str = None) -> str:
        """
        Download base diffusion model.

        Args:
            model_id: HuggingFace model ID for base model

        Returns:
            Path to downloaded model

        Raises:
            Exception: If download fails
        """
        model_id = model_id or self.DEFAULT_MODELS['base_model']

        logger.info(f"Downloading base model: {model_id}")

        try:
            start_time = time.time()

            # Download using snapshot_download for full model
            model_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                ignore_patterns=["*.bin"] if "float16" not in model_id else None
            )

            download_time = time.time() - start_time
            logger.info(f"Base model downloaded successfully in {download_time:.1f}s")
            logger.info(f"Model cached at: {model_path}")

            # Basic validation
            self._validate_base_model(model_path, model_id)

            self.downloaded_models.append({
                'type': 'base_model',
                'model_id': model_id,
                'path': model_path,
                'download_time': download_time,
                'status': 'success'
            })

            return model_id

        except Exception as e:
            logger.error(f"Failed to download base model {model_id}: {e}")
            self.downloaded_models.append({
                'type': 'base_model',
                'model_id': model_id,
                'error': str(e),
                'status': 'failed'
            })
            raise

    def download_all_models(self,
                           motion_adapter_id: str = None,
                           base_model_id: str = None) -> Dict[str, str]:
        """
        Download all required models for AnimateDiff.

        Args:
            motion_adapter_id: Custom motion adapter model ID
            base_model_id: Custom base model ID

        Returns:
            Dictionary with downloaded model IDs
        """
        logger.info("Starting AnimateDiff model download...")

        results = {}

        # Download motion adapter
        try:
            motion_adapter = self.download_motion_adapter(motion_adapter_id)
            results['motion_adapter'] = motion_adapter
            logger.info(f"✓ Motion adapter ready: {motion_adapter}")
        except Exception as e:
            logger.error(f"✗ Motion adapter download failed: {e}")
            results['motion_adapter'] = None

        # Download base model
        try:
            base_model = self.download_base_model(base_model_id)
            results['base_model'] = base_model
            logger.info(f"✓ Base model ready: {base_model}")
        except Exception as e:
            logger.error(f"✗ Base model download failed: {e}")
            results['base_model'] = None

        return results

    def validate_installation(self,
                            motion_adapter_id: str = None,
                            base_model_id: str = None) -> bool:
        """
        Validate that AnimateDiff can be loaded and used.

        Args:
            motion_adapter_id: Motion adapter model ID to validate
            base_model_id: Base model ID to validate

        Returns:
            True if validation passes, False otherwise
        """
        motion_adapter_id = motion_adapter_id or self.DEFAULT_MODELS['motion_adapter']
        base_model_id = base_model_id or self.DEFAULT_MODELS['base_model']

        logger.info("Validating AnimateDiff installation...")

        try:
            # Test loading motion adapter
            logger.info("Loading motion adapter...")
            motion_adapter = MotionAdapter.from_pretrained(
                motion_adapter_id,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir
            )

            # Test loading full pipeline
            logger.info("Loading AnimateDiff pipeline...")
            pipeline = AnimateDiffPipeline.from_pretrained(
                base_model_id,
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir
            )

            logger.info("✓ AnimateDiff pipeline loaded successfully")

            # Test basic functionality if CUDA is available
            if torch.cuda.is_available():
                logger.info("Testing basic generation (CUDA available)...")
                pipeline = pipeline.to("cuda")

                # Quick test with minimal parameters
                result = pipeline(
                    prompt="a flower blooming",
                    num_frames=8,
                    guidance_scale=5.0,
                    num_inference_steps=10,
                    height=256,
                    width=256
                )

                logger.info(f"✓ Test generation completed: {len(result.frames[0])} frames")
            else:
                logger.info("⚠ CUDA not available, skipping generation test")

            logger.info("✓ AnimateDiff validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"✗ AnimateDiff validation failed: {e}")
            return False

    def _validate_motion_adapter(self, motion_adapter, model_id: str) -> None:
        """Validate motion adapter structure."""
        if not hasattr(motion_adapter, 'config'):
            raise ValueError(f"Motion adapter {model_id} missing config")

        logger.info(f"Motion adapter validation passed: {model_id}")

    def _validate_base_model(self, model_path: str, model_id: str) -> None:
        """Validate base model structure."""
        model_path = Path(model_path)

        # Check for required files
        required_files = ['model_index.json']
        missing_files = []

        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)

        if missing_files:
            raise ValueError(f"Base model {model_id} missing files: {missing_files}")

        logger.info(f"Base model validation passed: {model_id}")

    def get_download_summary(self) -> Dict[str, Any]:
        """Get summary of download operations."""
        successful = [m for m in self.downloaded_models if m['status'] == 'success']
        failed = [m for m in self.downloaded_models if m['status'] == 'failed']

        total_time = sum(m.get('download_time', 0) for m in successful)

        return {
            'total_models': len(self.downloaded_models),
            'successful': len(successful),
            'failed': len(failed),
            'total_download_time': round(total_time, 1),
            'models': self.downloaded_models
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Download AnimateDiff models for image-to-video generation"
    )

    parser.add_argument(
        '--motion-adapter',
        default=AnimateDiffDownloader.DEFAULT_MODELS['motion_adapter'],
        help='Motion adapter model ID'
    )

    parser.add_argument(
        '--base-model',
        default=AnimateDiffDownloader.DEFAULT_MODELS['base_model'],
        help='Base diffusion model ID'
    )

    parser.add_argument(
        '--cache-dir',
        help='Custom cache directory for models'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate installation after download'
    )

    parser.add_argument(
        '--list-alternatives',
        action='store_true',
        help='List alternative model options'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List alternatives if requested
    if args.list_alternatives:
        print("Available AnimateDiff models:")
        print("\nMotion Adapters:")
        for model in AnimateDiffDownloader.ALTERNATIVE_MODELS['motion_adapter']:
            print(f"  - {model}")
        print("\nBase Models:")
        for model in AnimateDiffDownloader.ALTERNATIVE_MODELS['base_model']:
            print(f"  - {model}")
        return

    # Initialize downloader
    downloader = AnimateDiffDownloader(cache_dir=args.cache_dir)

    try:
        # Download models
        results = downloader.download_all_models(
            motion_adapter_id=args.motion_adapter,
            base_model_id=args.base_model
        )

        # Validate if requested
        if args.validate:
            validation_success = downloader.validate_installation(
                motion_adapter_id=args.motion_adapter,
                base_model_id=args.base_model
            )
            if not validation_success:
                logger.error("Validation failed")
                sys.exit(1)

        # Print summary
        summary = downloader.get_download_summary()

        print(f"\nDownload Summary:")
        print(f"  Total models: {summary['total_models']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Total time: {summary['total_download_time']}s")

        if summary['failed'] > 0:
            print(f"\nFailed downloads:")
            for model in summary['models']:
                if model['status'] == 'failed':
                    print(f"  - {model['type']}: {model['model_id']} - {model.get('error', 'Unknown error')}")
            sys.exit(1)
        else:
            print(f"\n✓ All AnimateDiff models downloaded successfully!")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()