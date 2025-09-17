#!/usr/bin/env python3
"""
Test script for the Model Builder GUI
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_model_config_builder():
    """Test the model configuration builder"""
    print("Testing Model Configuration Builder...")
    
    try:
        from har.models.model_config_builder import ModelConfigBuilder
        
        # Create builder
        builder = ModelConfigBuilder()
        
        # Test configuration - using a simpler architecture
        test_config = {
            "name": "TestModel",
            "input_shape": [1, 9, 128],  # [batch, features, sequence_length]
            "num_classes": 6,
            "layers": [
                {
                    "layer_type": "Flatten",
                    "config": {}
                },
                {
                    "layer_type": "Linear",
                    "config": {
                        "in_features": 9 * 128,  # 9 features * 128 time steps
                        "out_features": 64
                    }
                },
                {
                    "layer_type": "ReLU",
                    "config": {}
                },
                {
                    "layer_type": "Dropout",
                    "config": {
                        "p": 0.5
                    }
                },
                {
                    "layer_type": "Linear",
                    "config": {
                        "in_features": 64,
                        "out_features": 6
                    }
                }
            ]
        }
        
        # Validate configuration
        errors = builder.validate_config(test_config)
        if errors:
            print(f"Configuration errors: {errors}")
            return False
        
        # Build model
        model = builder.build_model_from_config(test_config)
        print(f"✓ Model created successfully: {model.__class__.__name__}")
        
        # Test model
        import torch
        dummy_input = torch.randn(2, 9, 128)  # [batch, features, sequence_length]
        output = model(dummy_input)
        print(f"✓ Model test passed - Input: {dummy_input.shape}, Output: {output.shape}")
        
        # Get model summary
        summary = builder.get_model_summary(model, test_config['input_shape'])
        print("Model Summary:")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"✗ Model builder test failed: {e}")
        return False

def test_custom_har_modules():
    """Test custom HAR modules in model builder"""
    print("\nTesting Custom HAR Modules...")
    
    try:
        from har.models.model_config_builder import ModelConfigBuilder
        
        builder = ModelConfigBuilder()
        
        # Test configuration with custom HAR modules
        # Note: Conv1D expects [batch, channels, sequence_length] format
        test_config = {
            "name": "HARTestModel",
            "input_shape": [1, 9, 128],  # [batch, features, sequence_length]
            "num_classes": 6,
            "layers": [
                {
                    "layer_type": "ConvBlock1D",
                    "config": {
                        "in_channels": 9,  # 9 features
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "dropout": 0.1
                    }
                },
                {
                    "layer_type": "BiDirectionalLSTM",
                    "config": {
                        "input_size": 32,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "dropout": 0.1
                    }
                },
                {
                    "layer_type": "Linear",
                    "config": {
                        "in_features": 128,  # 64 * 2 for bidirectional
                        "out_features": 6
                    }
                }
            ]
        }
        
        # Validate configuration
        errors = builder.validate_config(test_config)
        if errors:
            print(f"Configuration errors: {errors}")
            return False
        
        # Build model
        model = builder.build_model_from_config(test_config)
        print(f"✓ HAR model created successfully: {model.__class__.__name__}")
        
        # Test model
        import torch
        dummy_input = torch.randn(2, 9, 128)  # [batch, features, sequence_length]
        output = model(dummy_input)
        print(f"✓ HAR model test passed - Input: {dummy_input.shape}, Output: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ HAR modules test failed: {e}")
        return False

def test_available_layers():
    """Test available layer loading"""
    print("\nTesting Available Layers...")
    
    try:
        from har.models.model_config_builder import ModelConfigBuilder
        
        builder = ModelConfigBuilder()
        layers = builder.available_layers
        
        print(f"✓ Loaded {len(layers)} available layers")
        
        # Show some examples
        categories = {}
        for name, info in layers.items():
            category = info.get('type', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        for category, layer_names in categories.items():
            print(f"  {category}: {len(layer_names)} layers")
            if len(layer_names) <= 5:
                print(f"    {', '.join(layer_names)}")
            else:
                print(f"    {', '.join(layer_names[:5])}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Layer loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Model Builder Test Suite")
    print("=" * 50)
    
    tests = [
        test_available_layers,
        test_model_config_builder,
        test_custom_har_modules,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("✓ All tests passed! Model Builder is ready to use.")
        print("\nTo launch the GUI:")
        print("  python scripts/model_builder_gui.py")
        print("  or")
        print("  python scripts/experiment_launcher.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
