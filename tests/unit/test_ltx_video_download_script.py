#!/usr/bin/env python3
"""
Test suite for LTX-Video model download script.
"""

import unittest
import tempfile
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the scripts directory to the path
import sys
scripts_path = str(Path(__file__).parent.parent.parent / "scripts")  # Go up three levels from tests/unit/
sys.path.insert(0, scripts_path)

try:
    from download_ltx_video_models import LTXVideoModelDownloader
except ImportError as e:
    print(f"Failed to import LTXVideoModelDownloader from {scripts_path}")
    print(f"Error: {e}")
    raise


class TestLTXVideoModelDownloader(unittest.TestCase):
    """Test cases for LTX-Video model downloader."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = LTXVideoModelDownloader(
            cache_dir=self.temp_dir,
            validation_mode="basic"  # Use basic validation to avoid loading tests
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test downloader initialization."""
        self.assertTrue(self.downloader.cache_dir.exists())
        self.assertTrue(self.downloader.models_dir.exists())
        self.assertEqual(str(self.downloader.validation_mode), "basic")

    def test_get_ltx_video_config(self):
        """Test LTX-Video configuration."""
        config = self.downloader.get_ltx_video_config()

        self.assertIn("ltx_video", config)
        ltx_config = config["ltx_video"]

        self.assertEqual(ltx_config["repo_id"], "Lightricks/LTX-Video")
        self.assertEqual(ltx_config["revision"], "2b-distilled")
        self.assertGreater(ltx_config["estimated_size_gb"], 0)
        self.assertIn("required_files", ltx_config)
        self.assertIsInstance(ltx_config["required_files"], list)
        self.assertGreater(len(ltx_config["required_files"]), 0)

    def test_check_storage_space(self):
        """Test storage space checking."""
        # Should return True for reasonable space requirements
        result = self.downloader.check_storage_space(1.0)  # 1GB
        self.assertTrue(result)

        # Should return False for unrealistic space requirements
        result = self.downloader.check_storage_space(1000000.0)  # 1PB
        self.assertFalse(result)

    def test_validate_model_files_missing_directory(self):
        """Test validation with missing model directory."""
        config = self.downloader.get_ltx_video_config()
        result = self.downloader.validate_model_files(config)
        self.assertFalse(result)

    def test_validate_model_files_empty_directory(self):
        """Test validation with empty model directory."""
        config = self.downloader.get_ltx_video_config()

        # Create empty directory
        self.downloader.ltx_dir.mkdir(parents=True, exist_ok=True)

        result = self.downloader.validate_model_files(config)
        self.assertFalse(result)

    def test_validate_model_files_complete(self):
        """Test validation with complete model files."""
        config = self.downloader.get_ltx_video_config()
        ltx_config = config["ltx_video"]

        # Create model directory structure with all required files
        base_dir = Path(ltx_config["local_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        for file_path in ltx_config["required_files"]:
            full_path = base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Create non-empty files
            if file_path.endswith('.json'):
                with open(full_path, 'w') as f:
                    json.dump({"test": "data"}, f)
            else:
                # Create dummy file with some content
                with open(full_path, 'wb') as f:
                    f.write(b'dummy model data')

        # Add special handling for transformer config
        transformer_config_path = base_dir / "transformer/config.json"
        with open(transformer_config_path, 'w') as f:
            json.dump({
                "_class_name": "LTXVideoTransformer2D",
                "test": "data"
            }, f)

        # Add special handling for model index
        model_index_path = base_dir / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump({
                "ltx_video": "model_info",
                "test": "data"
            }, f)

        result = self.downloader.validate_model_files(config)
        self.assertTrue(result)

    def test_get_model_info_not_downloaded(self):
        """Test model info when not downloaded."""
        info = self.downloader.get_model_info()

        self.assertEqual(info["model_name"], "LTX-Video")
        self.assertEqual(info["status"], "not_downloaded")
        self.assertEqual(info["size_gb"], 0)
        self.assertFalse(info["files_present"])

    def test_get_model_info_downloaded(self):
        """Test model info when downloaded."""
        config = self.downloader.get_ltx_video_config()
        ltx_config = config["ltx_video"]

        # Create a minimal valid model structure
        base_dir = Path(ltx_config["local_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create required files with some content
        for file_path in ltx_config["required_files"]:
            full_path = base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.endswith('.json'):
                with open(full_path, 'w') as f:
                    json.dump({"test": "data", "large_field": "x" * 1000}, f)
            else:
                with open(full_path, 'wb') as f:
                    f.write(b'dummy model data' * 1000)  # Make files larger

        # Add transformer config
        transformer_config_path = base_dir / "transformer/config.json"
        with open(transformer_config_path, 'w') as f:
            json.dump({"_class_name": "LTXVideoTransformer2D"}, f)

        # Add model index
        model_index_path = base_dir / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump({"ltx_video": "info"}, f)

        info = self.downloader.get_model_info()

        self.assertEqual(info["model_name"], "LTX-Video")
        self.assertEqual(info["status"], "downloaded")
        self.assertGreater(info["size_gb"], 0)
        self.assertTrue(info["files_present"])

    def test_create_manifest(self):
        """Test manifest creation."""
        config = self.downloader.get_ltx_video_config()
        ltx_config = config["ltx_video"]

        # Create model directory with some files
        base_dir = Path(ltx_config["local_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        test_file = base_dir / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("test content")

        # Create manifest
        self.downloader._create_manifest(config)

        # Check manifest exists and has correct content
        manifest_path = base_dir / "download_manifest.json"
        self.assertTrue(manifest_path.exists())

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        self.assertEqual(manifest["model_name"], "LTX-Video")
        self.assertEqual(manifest["repo_id"], "Lightricks/LTX-Video")
        self.assertEqual(manifest["revision"], "2b-distilled")
        self.assertGreater(manifest["total_size_gb"], 0)
        self.assertTrue(manifest["validation_passed"])

    def test_cleanup_cache(self):
        """Test cache cleanup."""
        # Create cache directory with some files
        cache_path = self.downloader.cache_dir / ".cache"
        cache_path.mkdir(parents=True, exist_ok=True)

        test_file = cache_path / "test_cache_file.txt"
        with open(test_file, 'w') as f:
            f.write("cache content")

        self.assertTrue(cache_path.exists())
        self.assertTrue(test_file.exists())

        # Clean up cache
        self.downloader.cleanup_cache()

        self.assertFalse(cache_path.exists())

    @patch('download_ltx_video_models.snapshot_download')
    def test_download_ltx_video_model_success(self, mock_snapshot_download):
        """Test successful model download."""
        config = self.downloader.get_ltx_video_config()
        ltx_config = config["ltx_video"]

        # Mock successful download
        mock_snapshot_download.return_value = str(ltx_config["local_dir"])

        # Create the expected directory structure after "download"
        base_dir = Path(ltx_config["local_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create required files to pass validation
        for file_path in ltx_config["required_files"]:
            full_path = base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.endswith('.json'):
                with open(full_path, 'w') as f:
                    json.dump({"test": "data"}, f)
            else:
                with open(full_path, 'wb') as f:
                    f.write(b'dummy model data')

        # Add transformer config
        transformer_config_path = base_dir / "transformer/config.json"
        with open(transformer_config_path, 'w') as f:
            json.dump({"_class_name": "LTXVideoTransformer2D"}, f)

        # Add model index
        model_index_path = base_dir / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump({"ltx_video": "info"}, f)

        # Test download with force flag to ensure download is called
        result = self.downloader.download_ltx_video_model(config, force=True)

        self.assertTrue(result)
        mock_snapshot_download.assert_called_once()

        # Check manifest was created
        manifest_path = base_dir / "download_manifest.json"
        self.assertTrue(manifest_path.exists())

    @patch('download_ltx_video_models.snapshot_download')
    def test_download_ltx_video_model_failure(self, mock_snapshot_download):
        """Test failed model download."""
        config = self.downloader.get_ltx_video_config()

        # Mock download failure
        mock_snapshot_download.side_effect = Exception("Download failed")

        result = self.downloader.download_ltx_video_model(config)

        self.assertFalse(result)
        mock_snapshot_download.assert_called_once()

    def test_download_already_exists(self):
        """Test download when model already exists and is valid."""
        config = self.downloader.get_ltx_video_config()
        ltx_config = config["ltx_video"]

        # Create a complete valid model structure
        base_dir = Path(ltx_config["local_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)

        for file_path in ltx_config["required_files"]:
            full_path = base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.endswith('.json'):
                with open(full_path, 'w') as f:
                    json.dump({"test": "data"}, f)
            else:
                with open(full_path, 'wb') as f:
                    f.write(b'dummy model data')

        # Add transformer config
        transformer_config_path = base_dir / "transformer/config.json"
        with open(transformer_config_path, 'w') as f:
            json.dump({"_class_name": "LTXVideoTransformer2D"}, f)

        # Add model index
        model_index_path = base_dir / "model_index.json"
        with open(model_index_path, 'w') as f:
            json.dump({"ltx_video": "info"}, f)

        # Should skip download
        result = self.downloader.download_ltx_video_model(config)
        self.assertTrue(result)


class TestLTXVideoDownloadScriptIntegration(unittest.TestCase):
    """Integration tests for the download script."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_script_info_command(self):
        """Test the script's info command."""
        import subprocess

        script_path = Path(__file__).parent.parent.parent / "scripts" / "download_ltx_video_models.py"

        # Run the script with --info
        result = subprocess.run([
            "python", str(script_path),
            "--info",
            "--cache-dir", self.temp_dir
        ], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)

        # Parse the JSON output
        info = json.loads(result.stdout)
        self.assertEqual(info["model_name"], "LTX-Video")
        self.assertEqual(info["status"], "not_downloaded")

    def test_script_validate_only_empty(self):
        """Test the script's validate-only command with no models."""
        import subprocess

        script_path = Path(__file__).parent.parent.parent / "scripts" / "download_ltx_video_models.py"

        # Run the script with --validate-only
        result = subprocess.run([
            "python", str(script_path),
            "--validate-only",
            "--cache-dir", self.temp_dir,
            "--validation-mode", "basic"
        ], capture_output=True, text=True)

        # Should fail because no models are present
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()