#!/usr/bin/env python3
"""
CRITICAL: Basic Functionality Tests

This module contains basic tests that don't require heavy dependencies,
suitable for running in Docker environments.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestBasicFunctionality(unittest.TestCase):
    """Test cases for basic functionality"""
    
    def test_project_structure(self):
        """Test that the project has the expected structure"""
        # Check that main directories exist
        expected_dirs = [
            "har",
            "scripts", 
            "configs",
            "tests"
        ]
        
        for dir_name in expected_dirs:
            with self.subTest(directory=dir_name):
                dir_path = REPO_ROOT / dir_name
                self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
                self.assertTrue(dir_path.is_dir(), f"{dir_name} should be a directory")
    
    def test_har_module_structure(self):
        """Test that the HAR module has expected structure"""
        har_dir = REPO_ROOT / "har"
        
        expected_subdirs = [
            "datasets",
            "models", 
            "modules",
            "transforms"
        ]
        
        for subdir in expected_subdirs:
            with self.subTest(subdirectory=subdir):
                subdir_path = har_dir / subdir
                self.assertTrue(subdir_path.exists(), f"HAR subdirectory {subdir} should exist")
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        config_dir = REPO_ROOT / "configs"
        
        expected_configs = [
            "har_cnn_tcn.yaml",
            "resnet_classifier.yaml",
            "vgg_classifier.yaml", 
            "resnet_custom_head.yaml"
        ]
        
        for config_file in expected_configs:
            with self.subTest(config=config_file):
                config_path = config_dir / config_file
                self.assertTrue(config_path.exists(), f"Config file {config_file} should exist")
    
    def test_script_files_exist(self):
        """Test that important script files exist"""
        scripts_dir = REPO_ROOT / "scripts"
        
        expected_scripts = [
            "simple_model_builder_gui.py",
            "sweep_gui.py",
            "experiment_gui.py",
            "training_gui.py",
            "build_model_from_yaml.py"
        ]
        
        for script_file in expected_scripts:
            with self.subTest(script=script_file):
                script_path = scripts_dir / script_file
                self.assertTrue(script_path.exists(), f"Script {script_file} should exist")
    
    def test_yaml_parsing(self):
        """Test that YAML files can be parsed"""
        try:
            import yaml
            # Use the working approach from debug_yaml.py
            if not hasattr(yaml, 'safe_load'):
                # Try importing specific functions
                try:
                    from yaml import safe_load, YAMLError
                    yaml.safe_load = safe_load
                    yaml.YAMLError = YAMLError
                except ImportError:
                    # Try yaml.load with FullLoader as fallback
                    if hasattr(yaml, 'load') and hasattr(yaml, 'FullLoader'):
                        def safe_load_wrapper(stream):
                            return yaml.load(stream, Loader=yaml.FullLoader)
                        yaml.safe_load = safe_load_wrapper
                        yaml.YAMLError = getattr(yaml, 'YAMLError', Exception)
                    else:
                        self.skipTest("PyYAML not available or wrong yaml module")
        except ImportError:
            self.skipTest("PyYAML not available")
        
        config_dir = REPO_ROOT / "configs"
        
        # Only test model configuration files (exclude experiment configs)
        model_config_files = [
            "har_cnn_tcn.yaml",
            "resnet_classifier.yaml", 
            "vgg_classifier.yaml",
            "resnet_custom_head.yaml"
        ]
        
        for config_name in model_config_files:
            config_file = config_dir / config_name
            if not config_file.exists():
                self.skipTest(f"Configuration file not found: {config_file}")
                continue
                
            with self.subTest(config=config_name):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check basic structure
                    self.assertIn('model', config, f"{config_name} should have 'model' section")
                    self.assertIn('layers', config, f"{config_name} should have 'layers' section")
                    
                    # Check model section
                    model_section = config['model']
                    self.assertIn('name', model_section, "Model section should have 'name'")
                    self.assertIn('input_shape', model_section, "Model section should have 'input_shape'")
                    self.assertIn('num_classes', model_section, "Model section should have 'num_classes'")
                    
                except Exception as e:
                    self.fail(f"Failed to parse {config_name}: {e}")
    
    def test_python_imports(self):
        """Test that basic Python imports work"""
        # Test standard library imports
        try:
            import json
            import os
            import sys
            import pathlib
        except ImportError as e:
            self.fail(f"Standard library import failed: {e}")
        
        # Test optional imports
        optional_imports = [
            ('yaml', 'PyYAML'),
            ('numpy', 'NumPy'),
            ('torch', 'PyTorch'),
            ('pandas', 'Pandas'),
            ('tkinter', 'Tkinter')
        ]
        
        for module_name, package_name in optional_imports:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError:
                    # This is expected for some modules
                    pass
    
    def test_file_permissions(self):
        """Test that important files are readable"""
        important_files = [
            "tests/run_tests.py",
            "tests/run_tests.sh",
            "configs/har_cnn_tcn.yaml",
            "scripts/build_model_from_yaml.py"
        ]
        
        for file_path in important_files:
            with self.subTest(file=file_path):
                full_path = REPO_ROOT / file_path
                self.assertTrue(full_path.exists(), f"File {file_path} should exist")
                self.assertTrue(full_path.is_file(), f"{file_path} should be a file")
                # Check if file is readable (this might fail in some environments)
                try:
                    with open(full_path, 'r') as f:
                        f.read(1)  # Try to read at least one character
                except PermissionError:
                    self.fail(f"File {file_path} is not readable")


class TestYAMLConfigurations(unittest.TestCase):
    """Test cases for YAML configuration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_dir = REPO_ROOT / "configs"
    
    def test_yaml_syntax(self):
        """Test that all YAML files have valid syntax"""
        try:
            import yaml
            # Use the working approach from debug_yaml.py
            if not hasattr(yaml, 'safe_load'):
                # Try importing specific functions
                try:
                    from yaml import safe_load, YAMLError
                    yaml.safe_load = safe_load
                    yaml.YAMLError = YAMLError
                except ImportError:
                    # Try yaml.load with FullLoader as fallback
                    if hasattr(yaml, 'load') and hasattr(yaml, 'FullLoader'):
                        def safe_load_wrapper(stream):
                            return yaml.load(stream, Loader=yaml.FullLoader)
                        yaml.safe_load = safe_load_wrapper
                        yaml.YAMLError = getattr(yaml, 'YAMLError', Exception)
                    else:
                        self.skipTest("PyYAML not available or wrong yaml module")
        except ImportError:
            self.skipTest("PyYAML not available")
        
        for yaml_file in self.config_dir.glob("*.yaml"):
            with self.subTest(file=yaml_file.name):
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    self.fail(f"YAML syntax error in {yaml_file.name}: {e}")
    
    def test_model_configuration_structure(self):
        """Test that model configurations have the correct structure"""
        try:
            import yaml
            # Use the working approach from debug_yaml.py
            if not hasattr(yaml, 'safe_load'):
                # Try importing specific functions
                try:
                    from yaml import safe_load, YAMLError
                    yaml.safe_load = safe_load
                    yaml.YAMLError = YAMLError
                except ImportError:
                    # Try yaml.load with FullLoader as fallback
                    if hasattr(yaml, 'load') and hasattr(yaml, 'FullLoader'):
                        def safe_load_wrapper(stream):
                            return yaml.load(stream, Loader=yaml.FullLoader)
                        yaml.safe_load = safe_load_wrapper
                        yaml.YAMLError = getattr(yaml, 'YAMLError', Exception)
                    else:
                        self.skipTest("PyYAML not available or wrong yaml module")
        except ImportError:
            self.skipTest("PyYAML not available")
        
        # Only test model configuration files (exclude experiment configs)
        model_config_files = [
            "har_cnn_tcn.yaml",
            "resnet_classifier.yaml", 
            "vgg_classifier.yaml",
            "resnet_custom_head.yaml"
        ]
        
        for config_name in model_config_files:
            config_file = self.config_dir / config_name
            if not config_file.exists():
                self.skipTest(f"Configuration file not found: {config_file}")
                continue
                
            with self.subTest(file=config_name):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check top-level structure
                self.assertIn('model', config, "Config should have 'model' section")
                self.assertIn('layers', config, "Config should have 'layers' section")
                
                # Check model section
                model = config['model']
                required_model_fields = ['name', 'input_shape', 'num_classes']
                for field in required_model_fields:
                    self.assertIn(field, model, f"Model section should have '{field}'")
                
                # Check layers section
                layers = config['layers']
                self.assertIsInstance(layers, list, "Layers should be a list")
                
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIn('type', layer, f"Layer {i} should have 'type'")
                        self.assertIn('params', layer, f"Layer {i} should have 'params'")
                        self.assertIsInstance(layer['params'], dict, f"Layer {i} params should be a dict")


if __name__ == '__main__':
    unittest.main()
