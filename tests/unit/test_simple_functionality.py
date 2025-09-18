#!/usr/bin/env python3
"""
CRITICAL: Simple Functionality Tests

This module contains very basic tests that should work in any environment
without requiring heavy dependencies.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestSimpleFunctionality(unittest.TestCase):
    """Test cases for simple functionality that should always work"""
    
    def test_python_basics(self):
        """Test basic Python functionality"""
        # Test basic operations
        self.assertEqual(2 + 2, 4)
        self.assertTrue(True)
        self.assertFalse(False)
        
        # Test string operations
        test_string = "Hello, World!"
        self.assertEqual(len(test_string), 13)
        self.assertIn("World", test_string)
    
    def test_file_system_operations(self):
        """Test basic file system operations"""
        # Test that we can create and delete files
        test_file = Path("test_temp_file.txt")
        
        try:
            # Create file
            test_file.write_text("test content")
            self.assertTrue(test_file.exists())
            
            # Read file
            content = test_file.read_text()
            self.assertEqual(content, "test content")
            
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
    
    def test_path_operations(self):
        """Test path operations"""
        # Test path joining
        path1 = Path("tests")
        path2 = Path("unit")
        combined = path1 / path2
        self.assertEqual(str(combined), "tests/unit")
        
        # Test path existence
        self.assertTrue(Path("tests").exists())
        self.assertTrue(Path("har").exists())
    
    def test_import_basics(self):
        """Test basic imports"""
        # Test standard library imports
        import json
        import os
        import sys
        import pathlib
        
        # Test that we can import our modules
        try:
            import har
            self.assertTrue(True, "HAR module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import HAR module: {e}")
    
    def test_directory_structure(self):
        """Test that the project has the expected directory structure"""
        # Check main directories
        expected_dirs = ["har", "scripts", "configs", "tests"]
        
        for dir_name in expected_dirs:
            with self.subTest(directory=dir_name):
                dir_path = REPO_ROOT / dir_name
                self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
                self.assertTrue(dir_path.is_dir(), f"{dir_name} should be a directory")
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        config_dir = REPO_ROOT / "configs"
        
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            self.assertGreater(len(yaml_files), 0, "Should have at least one YAML config file")
            
            # Check specific files
            expected_files = [
                "har_cnn_tcn.yaml",
                "resnet_classifier.yaml",
                "vgg_classifier.yaml",
                "resnet_custom_head.yaml"
            ]
            
            for file_name in expected_files:
                file_path = config_dir / file_name
                if file_path.exists():
                    self.assertTrue(file_path.is_file(), f"{file_name} should be a file")
    
    def test_script_files_exist(self):
        """Test that important script files exist"""
        scripts_dir = REPO_ROOT / "scripts"
        
        if scripts_dir.exists():
            python_files = list(scripts_dir.glob("*.py"))
            self.assertGreater(len(python_files), 0, "Should have at least one Python script")
            
            # Check specific files
            expected_files = [
                "simple_model_builder_gui.py",
                "sweep_gui.py",
                "experiment_gui.py",
                "training_gui.py",
                "build_model_from_yaml.py"
            ]
            
            for file_name in expected_files:
                file_path = scripts_dir / file_name
                if file_path.exists():
                    self.assertTrue(file_path.is_file(), f"{file_name} should be a file")
    
    def test_har_module_structure(self):
        """Test that the HAR module has expected structure"""
        har_dir = REPO_ROOT / "har"
        
        if har_dir.exists():
            expected_subdirs = ["datasets", "models", "modules", "transforms"]
            
            for subdir in expected_subdirs:
                with self.subTest(subdirectory=subdir):
                    subdir_path = har_dir / subdir
                    if subdir_path.exists():
                        self.assertTrue(subdir_path.is_dir(), f"HAR subdirectory {subdir} should be a directory")
    
    def test_test_structure(self):
        """Test that the test structure is correct"""
        tests_dir = REPO_ROOT / "tests"
        
        if tests_dir.exists():
            # Check test subdirectories
            expected_test_dirs = ["unit", "integration", "yaml", "gui"]
            
            for test_dir in expected_test_dirs:
                with self.subTest(test_directory=test_dir):
                    test_path = tests_dir / test_dir
                    if test_path.exists():
                        self.assertTrue(test_path.is_dir(), f"Test directory {test_dir} should be a directory")
            
            # Check that we have test files
            test_files = list(tests_dir.rglob("test_*.py"))
            self.assertGreater(len(test_files), 0, "Should have at least one test file")
    
    def test_environment_variables(self):
        """Test that we can access environment variables"""
        # Test that we can read environment variables
        path_var = os.environ.get('PATH')
        self.assertIsNotNone(path_var, "PATH environment variable should exist")
        
        # Test that we can set and read custom environment variables
        test_var_name = "TEST_VAR_12345"
        test_var_value = "test_value_12345"
        
        try:
            os.environ[test_var_name] = test_var_value
            self.assertEqual(os.environ.get(test_var_name), test_var_value)
        finally:
            # Clean up
            if test_var_name in os.environ:
                del os.environ[test_var_name]


if __name__ == '__main__':
    unittest.main()
