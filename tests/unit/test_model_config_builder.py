#!/usr/bin/env python3
"""
CRITICAL: Unit Tests for ModelConfigBuilder

This module contains comprehensive unit tests for the ModelConfigBuilder class,
testing all core functionality including layer creation, model building, and validation.
"""

import unittest
import sys
import torch
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from har.models.model_config_builder import ModelConfigBuilder, load_yaml_model_config, build_model_from_yaml


class TestModelConfigBuilder(unittest.TestCase):
    """Test cases for ModelConfigBuilder class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.builder = ModelConfigBuilder()
    
    def test_builder_initialization(self):
        """Test that ModelConfigBuilder initializes correctly"""
        self.assertIsNotNone(self.builder)
        self.assertIsInstance(self.builder.available_layers, dict)
        self.assertGreater(len(self.builder.available_layers), 0)
    
    def test_available_layers(self):
        """Test that all expected layer types are available"""
        expected_layers = [
            'Linear', 'Conv1d', 'Conv2d', 'ReLU', 'Dropout', 
            'BatchNorm1d', 'BatchNorm2d', 'Flatten'
        ]
        
        for layer_name in expected_layers:
            with self.subTest(layer=layer_name):
                self.assertIn(layer_name, self.builder.available_layers)
                self.assertIn('class', self.builder.available_layers[layer_name])
                self.assertIn('type', self.builder.available_layers[layer_name])
    
    def test_torchvision_layers_available(self):
        """Test that torchvision models are available if torchvision is installed"""
        try:
            import torchvision
            expected_tv_layers = [
                'torchvision.models.resnet18',
                'torchvision.models.resnet50',
                'torchvision.models.vgg16'
            ]
            
            for layer_name in expected_tv_layers:
                with self.subTest(layer=layer_name):
                    self.assertIn(layer_name, self.builder.available_layers)
        except ImportError:
            self.skipTest("torchvision not available")
    
    def test_simple_model_creation(self):
        """Test creating a simple model with basic layers"""
        config = {
            "name": "test_model",
            "input_shape": [1, 3, 32, 32],
            "num_classes": 10,
            "layers": [
                {
                    "id": 0,
                    "layer_type": "Conv2d",
                    "config": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": 1}
                },
                {
                    "id": 1,
                    "layer_type": "ReLU",
                    "config": {}
                },
                {
                    "id": 2,
                    "layer_type": "Flatten",
                    "config": {}
                },
                {
                    "id": 3,
                    "layer_type": "Linear",
                    "config": {"in_features": 32 * 32 * 32, "out_features": 10}
                }
            ]
        }
        
        model = self.builder.build_model_from_config(config)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, "test_model")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 10))
    
    def test_model_validation(self):
        """Test model configuration validation"""
        # Valid config
        valid_config = {
            "name": "valid_model",
            "input_shape": [1, 3, 32, 32],
            "num_classes": 10,
            "layers": [
                {
                    "id": 0,
                    "layer_type": "Linear",
                    "config": {"in_features": 100, "out_features": 10}
                }
            ]
        }
        
        errors = self.builder.validate_config(valid_config)
        self.assertEqual(len(errors), 0)
        
        # Invalid config - missing required fields
        invalid_config = {
            "name": "invalid_model",
            # Missing input_shape and num_classes
            "layers": []
        }
        
        errors = self.builder.validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
        self.assertIn("input_shape", str(errors))
        self.assertIn("num_classes", str(errors))
    
    def test_layer_parameter_validation(self):
        """Test validation of layer parameters"""
        # Test with invalid layer type
        config = {
            "name": "test_model",
            "input_shape": [1, 3, 32, 32],
            "num_classes": 10,
            "layers": [
                {
                    "id": 0,
                    "layer_type": "NonExistentLayer",
                    "config": {}
                }
            ]
        }
        
        errors = self.builder.validate_config(config)
        self.assertGreater(len(errors), 0)
    
    def test_model_parameter_counting(self):
        """Test that model parameter counting works correctly"""
        config = {
            "name": "test_model",
            "input_shape": [1, 10],
            "num_classes": 5,
            "layers": [
                {
                    "id": 0,
                    "layer_type": "Linear",
                    "config": {"in_features": 10, "out_features": 20}
                },
                {
                    "id": 1,
                    "layer_type": "ReLU",
                    "config": {}
                },
                {
                    "id": 2,
                    "layer_type": "Linear",
                    "config": {"in_features": 20, "out_features": 5}
                }
            ]
        }
        
        model = self.builder.build_model_from_config(config)
        
        # Calculate expected parameters
        # Linear(10, 20): 10*20 + 20 = 220
        # ReLU: 0 parameters
        # Linear(20, 5): 20*5 + 5 = 105
        # Total: 325 parameters
        expected_params = 10*20 + 20 + 20*5 + 5  # 325
        
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, expected_params)


class TestYAMLLoading(unittest.TestCase):
    """Test cases for YAML configuration loading"""
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration files"""
        # Skip this test if YAML is not available
        try:
            import yaml
            if not hasattr(yaml, 'safe_load'):
                # Try to fix the yaml module
                try:
                    from yaml import safe_load, YAMLError
                    yaml.safe_load = safe_load
                    yaml.YAMLError = YAMLError
                except ImportError:
                    self.skipTest("YAML module not properly configured")
        except ImportError:
            self.skipTest("YAML module not available")
        
        # Create a test YAML config
        test_config_path = Path("tests/test_config.yaml")
        test_config_content = """
model:
  name: "test_yaml_model"
  input_shape: [1, 3, 32, 32]
  num_classes: 5

layers:
  - type: "Conv2d"
    params:
      in_channels: 3
      out_channels: 32
      kernel_size: 3
      padding: 1
  - type: "ReLU"
    params: {}
  - type: "Flatten"
    params: {}
  - type: "Linear"
    params:
      in_features: 32*32*32
      out_features: 5
"""
        
        try:
            with open(test_config_path, 'w') as f:
                f.write(test_config_content)
            
            config = load_yaml_model_config(test_config_path)
            
            self.assertEqual(config['name'], 'test_yaml_model')
            self.assertEqual(config['input_shape'], [1, 3, 32, 32])
            self.assertEqual(config['num_classes'], 5)
            self.assertEqual(len(config['layers']), 4)
            
        finally:
            # Clean up
            if test_config_path.exists():
                test_config_path.unlink()
    
    def test_build_model_from_yaml(self):
        """Test building model directly from YAML file"""
        # Skip this test if YAML is not available
        try:
            import yaml
            if not hasattr(yaml, 'safe_load'):
                # Try to fix the yaml module
                try:
                    from yaml import safe_load, YAMLError
                    yaml.safe_load = safe_load
                    yaml.YAMLError = YAMLError
                except ImportError:
                    self.skipTest("YAML module not properly configured")
        except ImportError:
            self.skipTest("YAML module not available")
        
        # Create a test YAML config
        test_config_path = Path("tests/test_model.yaml")
        test_config_content = """
model:
  name: "test_yaml_model"
  input_shape: [1, 10]
  num_classes: 3

layers:
  - type: "Linear"
    params:
      in_features: 10
      out_features: 5
  - type: "ReLU"
    params: {}
  - type: "Linear"
    params:
      in_features: 5
      out_features: 3
"""
        
        try:
            with open(test_config_path, 'w') as f:
                f.write(test_config_content)
            
            model = build_model_from_yaml(test_config_path)
            
            self.assertIsNotNone(model)
            self.assertEqual(model.__class__.__name__, 'test_yaml_model')
            
            # Test forward pass
            dummy_input = torch.randn(1, 10)
            with torch.no_grad():
                output = model(dummy_input)
            
            self.assertEqual(output.shape, (1, 3))
            
        finally:
            # Clean up
            if test_config_path.exists():
                test_config_path.unlink()


if __name__ == '__main__':
    unittest.main()
