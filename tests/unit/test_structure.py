"""
Test directory structure verification.

Validates that the multi-modal inference worker directory structure
is properly established and accessible.
"""

import os
import unittest
from pathlib import Path


class TestDirectoryStructure(unittest.TestCase):
    """Test cases for validating directory structure setup."""

    def setUp(self):
        """Set up test fixtures with expected directory paths."""
        self.worker_root = Path(__file__).parent.parent.parent
        self.src_dir = self.worker_root / "src"
        self.docker_dir = self.worker_root / "docker"
        self.tests_dir = self.worker_root / "tests"
        self.docs_dir = self.worker_root / "docs"

    def test_main_directories_exist(self):
        """Verify that main directories exist."""
        directories = [
            self.src_dir,
            self.docker_dir,
            self.tests_dir,
            self.docs_dir
        ]

        for directory in directories:
            with self.subTest(directory=str(directory)):
                self.assertTrue(
                    directory.exists(),
                    f"Directory {directory} should exist"
                )
                self.assertTrue(
                    directory.is_dir(),
                    f"{directory} should be a directory"
                )

    def test_src_subdirectories_exist(self):
        """Verify that src subdirectories exist."""
        src_subdirs = [
            self.src_dir / "handlers",
            self.src_dir / "models",
            self.src_dir / "utils"
        ]

        for subdir in src_subdirs:
            with self.subTest(subdir=str(subdir)):
                self.assertTrue(
                    subdir.exists(),
                    f"Source subdirectory {subdir} should exist"
                )
                self.assertTrue(
                    subdir.is_dir(),
                    f"{subdir} should be a directory"
                )

    def test_test_subdirectories_exist(self):
        """Verify that test subdirectories exist."""
        test_subdirs = [
            self.tests_dir / "unit",
            self.tests_dir / "integration"
        ]

        for subdir in test_subdirs:
            with self.subTest(subdir=str(subdir)):
                self.assertTrue(
                    subdir.exists(),
                    f"Test subdirectory {subdir} should exist"
                )
                self.assertTrue(
                    subdir.is_dir(),
                    f"{subdir} should be a directory"
                )

    def test_python_init_files_exist(self):
        """Verify that __init__.py files exist for Python packages."""
        init_files = [
            self.src_dir / "__init__.py",
            self.src_dir / "handlers" / "__init__.py",
            self.src_dir / "models" / "__init__.py",
            self.src_dir / "utils" / "__init__.py",
            self.tests_dir / "__init__.py",
            self.tests_dir / "unit" / "__init__.py",
            self.tests_dir / "integration" / "__init__.py"
        ]

        for init_file in init_files:
            with self.subTest(init_file=str(init_file)):
                self.assertTrue(
                    init_file.exists(),
                    f"__init__.py file {init_file} should exist"
                )
                self.assertTrue(
                    init_file.is_file(),
                    f"{init_file} should be a file"
                )

    def test_main_entry_point_exists(self):
        """Verify that main.py entry point exists."""
        main_file = self.src_dir / "main.py"

        self.assertTrue(
            main_file.exists(),
            "Main entry point src/main.py should exist"
        )
        self.assertTrue(
            main_file.is_file(),
            "src/main.py should be a file"
        )

    def test_docker_files_exist(self):
        """Verify that Docker configuration files exist."""
        docker_files = [
            self.docker_dir / "Dockerfile",
            self.docker_dir / "requirements.txt"
        ]

        for docker_file in docker_files:
            with self.subTest(docker_file=str(docker_file)):
                self.assertTrue(
                    docker_file.exists(),
                    f"Docker file {docker_file} should exist"
                )
                self.assertTrue(
                    docker_file.is_file(),
                    f"{docker_file} should be a file"
                )

    def test_documentation_files_exist(self):
        """Verify that documentation files exist."""
        doc_files = [
            self.docs_dir / "api.md",
            self.docs_dir / "deployment.md"
        ]

        for doc_file in doc_files:
            with self.subTest(doc_file=str(doc_file)):
                self.assertTrue(
                    doc_file.exists(),
                    f"Documentation file {doc_file} should exist"
                )
                self.assertTrue(
                    doc_file.is_file(),
                    f"{doc_file} should be a file"
                )

    def test_directories_are_accessible(self):
        """Verify that directories have proper permissions."""
        directories = [
            self.src_dir,
            self.src_dir / "handlers",
            self.src_dir / "models",
            self.src_dir / "utils",
            self.docker_dir,
            self.tests_dir,
            self.tests_dir / "unit",
            self.tests_dir / "integration",
            self.docs_dir
        ]

        for directory in directories:
            with self.subTest(directory=str(directory)):
                # Test read access
                self.assertTrue(
                    os.access(directory, os.R_OK),
                    f"Directory {directory} should be readable"
                )

                # Test execute access (needed for directory traversal)
                self.assertTrue(
                    os.access(directory, os.X_OK),
                    f"Directory {directory} should be executable"
                )


if __name__ == "__main__":
    unittest.main()