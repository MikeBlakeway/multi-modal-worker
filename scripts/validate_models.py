#!/usr/bin/env python3
"""
Multi-Modal Inference Worker Model Validation Script

Validates the integrity and functionality of downloaded AI models
for the multi-modal inference worker deployment.

Usage:
    python validate_models.py --models-dir=/runpod-volume/models --mode=strict
"""

import os
import sys
import argparse
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive model validation for multi-modal inference worker."""

    def __init__(
        self,
        models_dir: str = "/runpod-volume/models",
        validation_mode: str = "basic",
        require_gpu: bool = False
    ):
        self.models_dir = Path(models_dir)
        self.validation_mode = validation_mode
        self.require_gpu = require_gpu
        self.validation_results = {}
        self.total_models = 0
        self.passed_models = 0

        logger.info(f"Initialized ModelValidator:")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Validation mode: {validation_mode}")
        logger.info(f"  GPU required: {require_gpu}")

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.require_gpu and not self.gpu_available:
            logger.warning("GPU required but not available")

    def get_expected_models(self) -> Dict[str, Dict]:
        """Define expected model structure and requirements."""
        return {
            "flux": {
                "local_dir": "flux",
                "required_files": [
                    "flux1-schnell.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder_2/config.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer_2/tokenizer_config.json",
                    "vae/config.json"
                ],
                "min_size_mb": 10000,  # 10GB minimum
                "model_type": "diffusion",
                "tensor_files": ["flux1-schnell.safetensors"]
            },
            "controlnet_canny": {
                "local_dir": "controlnet/canny",
                "required_files": ["config.json"],
                "min_size_mb": 1000,  # 1GB minimum
                "model_type": "controlnet",
                "tensor_files": ["diffusion_pytorch_model.safetensors"]
            },
            "controlnet_depth": {
                "local_dir": "controlnet/depth",
                "required_files": ["config.json"],
                "min_size_mb": 1000,
                "model_type": "controlnet",
                "tensor_files": ["diffusion_pytorch_model.safetensors"]
            },
            "animatediff": {
                "local_dir": "animatediff",
                "required_files": ["config.json"],
                "min_size_mb": 1000,
                "model_type": "motion_adapter",
                "tensor_files": ["diffusion_pytorch_model.safetensors"]
            },
            "ltx_video": {
                "local_dir": "video_backbones/ltx-2b",
                "required_files": ["config.json"],
                "min_size_mb": 5000,  # 5GB minimum
                "model_type": "video_diffusion",
                "tensor_files": ["model.safetensors"]
            },
            "sdxl_inpaint": {
                "local_dir": "inpaint/sdxl",
                "required_files": ["config.json"],
                "min_size_mb": 4000,  # 4GB minimum
                "model_type": "inpainting",
                "tensor_files": ["diffusion_pytorch_model.safetensors"]
            },
            "camera_ctrl": {
                "local_dir": "camera/camctrllib",
                "required_files": [],  # Flexible requirements
                "min_size_mb": 500,
                "model_type": "camera_control",
                "tensor_files": []
            }
        }

    def format_size(self, size_bytes: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory recursively."""
        total_size = 0
        if directory.exists() and directory.is_dir():
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size

    def check_file_existence(self, model_name: str, config: Dict) -> Tuple[bool, List[str]]:
        """Check if required files exist for a model."""
        model_path = self.models_dir / config["local_dir"]
        missing_files = []

        if not model_path.exists():
            return False, [f"Model directory {config['local_dir']} does not exist"]

        for required_file in config["required_files"]:
            file_path = model_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)

        return len(missing_files) == 0, missing_files

    def check_size_requirements(self, model_name: str, config: Dict) -> Tuple[bool, int]:
        """Check if model meets minimum size requirements."""
        model_path = self.models_dir / config["local_dir"]

        if not model_path.exists():
            return False, 0

        actual_size = self.calculate_directory_size(model_path)
        min_size_bytes = config["min_size_mb"] * 1024 * 1024

        return actual_size >= min_size_bytes, actual_size

    def validate_tensor_files(self, model_name: str, config: Dict) -> Tuple[bool, List[str]]:
        """Validate tensor files can be loaded."""
        if self.validation_mode == "basic":
            return True, []

        model_path = self.models_dir / config["local_dir"]
        errors = []

        # Check specified tensor files
        tensor_files = config.get("tensor_files", [])
        if not tensor_files:
            # Find any safetensors files
            tensor_files = [f.name for f in model_path.rglob("*.safetensors")]

        for tensor_file in tensor_files:
            tensor_path = model_path / tensor_file
            if not tensor_path.exists():
                # Try to find it recursively
                found_files = list(model_path.rglob(tensor_file))
                if not found_files:
                    errors.append(f"Tensor file {tensor_file} not found")
                    continue
                tensor_path = found_files[0]

            try:
                # Attempt to load tensor metadata
                if tensor_path.suffix == ".safetensors":
                    import safetensors
                    with safetensors.safe_open(str(tensor_path), framework="pt") as f:
                        keys = list(f.keys())
                        if len(keys) == 0:
                            errors.append(f"Empty tensor file: {tensor_file}")
                        elif self.validation_mode == "strict":
                            # Try to load first tensor to verify integrity
                            first_key = keys[0]
                            try:
                                tensor = f.get_tensor(first_key)
                                if tensor is None or tensor.numel() == 0:
                                    errors.append(f"Invalid tensor data in {tensor_file}")
                            except Exception as e:
                                errors.append(f"Cannot load tensor from {tensor_file}: {str(e)}")

                elif tensor_path.suffix in [".bin", ".pt", ".pth"]:
                    # PyTorch format validation
                    try:
                        checkpoint = torch.load(str(tensor_path), map_location="cpu")
                        if not isinstance(checkpoint, dict) or len(checkpoint) == 0:
                            errors.append(f"Invalid PyTorch file: {tensor_file}")
                    except Exception as e:
                        errors.append(f"Cannot load PyTorch file {tensor_file}: {str(e)}")

            except ImportError:
                logger.warning(f"Cannot validate {tensor_file} - missing dependencies")
            except Exception as e:
                errors.append(f"Tensor validation error for {tensor_file}: {str(e)}")

        return len(errors) == 0, errors

    def validate_config_files(self, model_name: str, config: Dict) -> Tuple[bool, List[str]]:
        """Validate configuration files can be loaded as JSON."""
        model_path = self.models_dir / config["local_dir"]
        errors = []

        # Check config.json files
        for config_file in model_path.rglob("config.json"):
            try:
                with open(config_file, 'r') as f:
                    json_data = json.load(f)
                    if not isinstance(json_data, dict):
                        errors.append(f"Invalid JSON structure in {config_file.relative_to(model_path)}")
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error in {config_file.relative_to(model_path)}: {str(e)}")
            except Exception as e:
                errors.append(f"Config file error {config_file.relative_to(model_path)}: {str(e)}")

        return len(errors) == 0, errors

    def run_model_specific_validation(self, model_name: str, config: Dict) -> Tuple[bool, List[str]]:
        """Run model-type specific validation."""
        if self.validation_mode == "basic":
            return True, []

        model_type = config.get("model_type", "unknown")
        errors = []

        try:
            if model_type == "diffusion":
                # Check for standard diffusion model components
                model_path = self.models_dir / config["local_dir"]

                # Look for UNet, VAE, text encoder components
                has_unet = any(model_path.rglob("*unet*"))
                has_vae = any(model_path.rglob("*vae*")) or any(model_path.rglob("vae/*"))
                has_text_encoder = any(model_path.rglob("*text_encoder*")) or any(model_path.rglob("text_encoder*/*"))

                if not (has_unet or has_vae or has_text_encoder):
                    errors.append("Missing standard diffusion model components")

            elif model_type == "controlnet":
                # Check ControlNet specific requirements
                model_path = self.models_dir / config["local_dir"]

                # Look for controlnet config
                config_files = list(model_path.rglob("config.json"))
                if config_files:
                    with open(config_files[0], 'r') as f:
                        config_data = json.load(f)
                        if "_class_name" not in config_data and "controlnet" not in str(config_data).lower():
                            logger.warning(f"Config may not be ControlNet format for {model_name}")

            elif model_type == "video_diffusion":
                # Check video model specific requirements
                model_path = self.models_dir / config["local_dir"]

                # Look for video-specific components
                has_temporal = any("temporal" in str(f).lower() for f in model_path.rglob("*"))
                has_motion = any("motion" in str(f).lower() for f in model_path.rglob("*"))

                if not (has_temporal or has_motion):
                    logger.warning(f"No temporal/motion components found for video model {model_name}")

        except Exception as e:
            errors.append(f"Model-specific validation error: {str(e)}")

        return len(errors) == 0, errors

    def validate_single_model(self, model_name: str, config: Dict) -> Dict[str, Any]:
        """Validate a single model comprehensively."""
        logger.info(f"Validating model: {model_name}")

        result = {
            "model_name": model_name,
            "model_type": config.get("model_type", "unknown"),
            "local_dir": config["local_dir"],
            "validation_timestamp": datetime.now().isoformat(),
            "passed": False,
            "checks": {}
        }

        # 1. File existence check
        files_exist, missing_files = self.check_file_existence(model_name, config)
        result["checks"]["files_exist"] = {
            "passed": files_exist,
            "missing_files": missing_files
        }

        # 2. Size requirements check
        size_ok, actual_size = self.check_size_requirements(model_name, config)
        result["checks"]["size_requirements"] = {
            "passed": size_ok,
            "actual_size_bytes": actual_size,
            "actual_size_formatted": self.format_size(actual_size),
            "min_size_mb": config["min_size_mb"]
        }

        # 3. Tensor file validation
        tensors_ok, tensor_errors = self.validate_tensor_files(model_name, config)
        result["checks"]["tensor_validation"] = {
            "passed": tensors_ok,
            "errors": tensor_errors
        }

        # 4. Configuration validation
        configs_ok, config_errors = self.validate_config_files(model_name, config)
        result["checks"]["config_validation"] = {
            "passed": configs_ok,
            "errors": config_errors
        }

        # 5. Model-specific validation
        specific_ok, specific_errors = self.run_model_specific_validation(model_name, config)
        result["checks"]["model_specific"] = {
            "passed": specific_ok,
            "errors": specific_errors
        }

        # Overall result
        all_checks_passed = all([
            files_exist,
            size_ok,
            tensors_ok,
            configs_ok,
            specific_ok
        ])

        result["passed"] = all_checks_passed

        if all_checks_passed:
            logger.info(f"✅ {model_name} validation PASSED")
            self.passed_models += 1
        else:
            logger.warning(f"❌ {model_name} validation FAILED")

        return result

    def validate_all_models(self) -> Dict[str, Any]:
        """Validate all expected models."""
        logger.info("Starting comprehensive model validation...")

        expected_models = self.get_expected_models()
        self.total_models = len(expected_models)

        # Validate each model
        for model_name, config in expected_models.items():
            result = self.validate_single_model(model_name, config)
            self.validation_results[model_name] = result

        # Generate summary
        summary = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_mode": self.validation_mode,
            "total_models": self.total_models,
            "passed_models": self.passed_models,
            "failed_models": self.total_models - self.passed_models,
            "success_rate": (self.passed_models / self.total_models) * 100 if self.total_models > 0 else 0,
            "gpu_available": self.gpu_available,
            "models": self.validation_results
        }

        return summary

    def save_validation_report(self, summary: Dict[str, Any]) -> None:
        """Save validation report to JSON file."""
        report_path = self.models_dir / "validation_report.json"

        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Validation report saved to {report_path}")

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print validation summary to console."""
        print("\n" + "="*60)
        print("MODEL VALIDATION SUMMARY")
        print("="*60)
        print(f"Validation Mode: {summary['validation_mode']}")
        print(f"Total Models: {summary['total_models']}")
        print(f"Passed: {summary['passed_models']}")
        print(f"Failed: {summary['failed_models']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"GPU Available: {summary['gpu_available']}")
        print()

        # Print per-model results
        for model_name, result in summary['models'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {model_name:20} ({result['model_type']})")

            if not result['passed']:
                for check_name, check_result in result['checks'].items():
                    if not check_result['passed']:
                        print(f"      ↳ {check_name}: {check_result}")

        print("="*60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Validate models for multi-modal inference worker")

    parser.add_argument(
        "--models-dir",
        default="/runpod-volume/models",
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "strict"],
        default="basic",
        help="Validation thoroughness level"
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require GPU for validation"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save validation report to JSON"
    )

    args = parser.parse_args()

    # Initialize validator
    validator = ModelValidator(
        models_dir=args.models_dir,
        validation_mode=args.mode,
        require_gpu=args.require_gpu
    )

    try:
        # Run validation
        summary = validator.validate_all_models()

        # Print results
        validator.print_summary(summary)

        # Save report if requested
        if args.save_report:
            validator.save_validation_report(summary)

        # Exit with appropriate code
        if summary['failed_models'] > 0:
            logger.warning(f"Validation completed with {summary['failed_models']} failures")
            sys.exit(1)
        else:
            logger.info("All models validated successfully")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()