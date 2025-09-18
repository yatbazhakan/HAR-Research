#!/usr/bin/env python3
"""
CRITICAL: YAML Model Loading Tests

This module contains comprehensive tests for YAML-based model loading functionality,
including torchvision models, custom architectures, and error handling.
"""

import unittest
import sys
import torch
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from har.models.model_config_builder import build_model_from_yaml, load_yaml_model_config


class TestYAMLModelLoading(unittest.TestCase):
    """Test cases for YAML model loading functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Get the project root directory (3 levels up from this test file)
        self.config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        self.test_configs = [
            "har_cnn_tcn.yaml",
            "resnet_classifier.yaml", 
            "vgg_classifier.yaml",
            "resnet_custom_head.yaml"
        ]
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration files"""
        for config_file in self.test_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    config = load_yaml_model_config(config_path)
                    
                    # Check required fields
                    self.assertIn('name', config)
                    self.assertIn('input_shape', config)
                    self.assertIn('num_classes', config)
                    self.assertIn('layers', config)
                    
                    # Check data types
                    self.assertIsInstance(config['name'], str)
                    self.assertIsInstance(config['input_shape'], list)
                    self.assertIsInstance(config['num_classes'], int)
                    self.assertIsInstance(config['layers'], list)
                    
                    # Check layer structure
                    for i, layer in enumerate(config['layers']):
                        self.assertIn('id', layer)
                        self.assertIn('layer_type', layer)
                        self.assertIn('config', layer)
                        self.assertEqual(layer['id'], i)
                    
                except Exception as e:
                    self.fail(f"Failed to load {config_file}: {e}")
    
    def test_model_building_from_yaml(self):
        """Test building models from YAML configuration files"""
        for config_file in self.test_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    model = build_model_from_yaml(config_path)
                    
                    # Check model creation
                    self.assertIsNotNone(model)
                    self.assertIsInstance(model, torch.nn.Module)
                    
                    # Check model has parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    self.assertGreater(total_params, 0)
                    
                except Exception as e:
                    self.fail(f"Failed to build model from {config_file}: {e}")
    
    def test_model_forward_pass(self):
        """Test forward pass through models built from YAML"""
        for config_file in self.test_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    # Load config to get input shape and num_classes
                    config = load_yaml_model_config(config_path)
                    model = build_model_from_yaml(config_path)
                    
                    # Create dummy input
                    input_shape = config['input_shape']
                    dummy_input = torch.randn(input_shape)
                    
                    # Test forward pass
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # Check output shape
                    expected_shape = (input_shape[0], config['num_classes'])
                    self.assertEqual(output.shape, expected_shape)
                    
                    # Check output is not NaN or Inf
                    self.assertFalse(torch.isnan(output).any())
                    self.assertFalse(torch.isinf(output).any())
                    
                except Exception as e:
                    self.fail(f"Forward pass failed for {config_file}: {e}")
    
    def test_torchvision_models(self):
        """Test torchvision model loading and functionality"""
        torchvision_configs = [
            "resnet_classifier.yaml",
            "vgg_classifier.yaml",
            "resnet_custom_head.yaml"
        ]
        
        for config_file in torchvision_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    model = build_model_from_yaml(config_path)
                    
                    # Check that model was created successfully
                    self.assertIsNotNone(model)
                    
                    # Test with different batch sizes
                    config = load_yaml_model_config(config_path)
                    input_shape = config['input_shape']
                    
                    for batch_size in [1, 2, 4]:
                        test_input = torch.randn(batch_size, *input_shape[1:])
                        with torch.no_grad():
                            output = model(test_input)
                        
                        expected_shape = (batch_size, config['num_classes'])
                        self.assertEqual(output.shape, expected_shape)
                    
                except Exception as e:
                    self.fail(f"Torchvision model test failed for {config_file}: {e}")
    
    def test_har_cnn_tcn_model(self):
        """Test HAR-specific CNN-TCN model"""
        config_file = "har_cnn_tcn.yaml"
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            self.skipTest(f"Configuration file not found: {config_path}")
        
        try:
            model = build_model_from_yaml(config_path)
            config = load_yaml_model_config(config_path)
            
            # Test with HAR input shape [batch, channels, time_steps]
            input_shape = config['input_shape']  # [1, 6, 128]
            dummy_input = torch.randn(input_shape)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            expected_shape = (input_shape[0], config['num_classes'])
            self.assertEqual(output.shape, expected_shape)
            
            # Check that model has reasonable number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            self.assertGreater(total_params, 1000)  # Should have substantial parameters
            self.assertLess(total_params, 10000000)  # But not too many
            
        except Exception as e:
            self.fail(f"HAR CNN-TCN model test failed: {e}")
    
    def test_model_parameter_counting(self):
        """Test that model parameter counting works correctly"""
        for config_file in self.test_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    model = build_model_from_yaml(config_path)
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    # Check that we have parameters
                    self.assertGreater(total_params, 0)
                    self.assertGreater(trainable_params, 0)
                    
                    # Check that trainable params <= total params
                    self.assertLessEqual(trainable_params, total_params)
                    
                except Exception as e:
                    self.fail(f"Parameter counting failed for {config_file}: {e}")
    
    def test_model_device_handling(self):
        """Test that models work on different devices"""
        for config_file in self.test_configs:
            with self.subTest(config=config_file):
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    self.skipTest(f"Configuration file not found: {config_path}")
                
                try:
                    model = build_model_from_yaml(config_path)
                    config = load_yaml_model_config(config_path)
                    
                    # Test on CPU
                    model_cpu = model.cpu()
                    input_shape = config['input_shape']
                    dummy_input = torch.randn(input_shape)
                    
                    with torch.no_grad():
                        output_cpu = model_cpu(dummy_input)
                    
                    self.assertEqual(output_cpu.device.type, 'cpu')
                    
                    # Test on CUDA if available
                    if torch.cuda.is_available():
                        model_cuda = model.cuda()
                        dummy_input_cuda = dummy_input.cuda()
                        
                        with torch.no_grad():
                            output_cuda = model_cuda(dummy_input_cuda)
                        
                        self.assertEqual(output_cuda.device.type, 'cuda')
                        
                        # Check that outputs are similar (within numerical precision)
                        output_cpu_cuda = output_cpu.cuda()
                        self.assertTrue(torch.allclose(output_cpu_cuda, output_cuda, atol=1e-6))
                    
                except Exception as e:
                    self.fail(f"Device handling test failed for {config_file}: {e}")


if __name__ == '__main__':
    unittest.main()