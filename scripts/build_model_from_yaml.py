#!/usr/bin/env python3
"""
CRITICAL: Command-line Tool for Building Models from YAML Configurations

This script provides a command-line interface for building PyTorch models
from YAML configuration files. It supports model creation, testing, and export.

Usage:
    python scripts/build_model_from_yaml.py --config configs/my_model.yaml
    python scripts/build_model_from_yaml.py --config configs/my_model.yaml --test
    python scripts/build_model_from_yaml.py --config configs/my_model.yaml --export model.pth
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from har.models.model_config_builder import build_model_from_yaml, load_yaml_model_config


def main():
    """Main function for the YAML model builder CLI"""
    parser = argparse.ArgumentParser(
        description="Build PyTorch models from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build and test a model
  python scripts/build_model_from_yaml.py --config configs/resnet_classifier.yaml --test
  
  # Export model to PyTorch format
  python scripts/build_model_from_yaml.py --config configs/har_cnn_tcn.yaml --export model.pth
  
  # Show model summary
  python scripts/build_model_from_yaml.py --config configs/vgg_classifier.yaml --summary
        """
    )
    
    # CRITICAL: Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    
    # CRITICAL: Optional arguments
    parser.add_argument("--test", action="store_true",
                       help="Test the model with dummy input")
    parser.add_argument("--export", type=str, metavar="PATH",
                       help="Export model to PyTorch file")
    parser.add_argument("--summary", action="store_true",
                       help="Show model summary and statistics")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to run model on (default: cpu)")
    
    args = parser.parse_args()
    
    # CRITICAL: Validate configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        return 1
    
    try:
        # CRITICAL: Load and build model
        print(f"🔧 Loading configuration from: {config_path}")
        config = load_yaml_model_config(config_path)
        
        print(f"📋 Model Configuration:")
        print(f"   Name: {config['name']}")
        print(f"   Input shape: {config['input_shape']}")
        print(f"   Number of classes: {config['num_classes']}")
        print(f"   Number of layers: {len(config['layers'])}")
        
        print(f"\n🏗️  Building model...")
        model = build_model_from_yaml(config_path)
        print(f"   ✓ Model type: {type(model).__name__}")
        
        # CRITICAL: Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        
        # CRITICAL: Move model to specified device
        device = torch.device(args.device)
        model = model.to(device)
        print(f"   ✓ Model moved to: {device}")
        
        # CRITICAL: Test model if requested
        if args.test:
            print(f"\n🧪 Testing model...")
            input_shape = config['input_shape']
            dummy_input = torch.randn(input_shape).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   ✓ Input shape: {dummy_input.shape}")
            print(f"   ✓ Output shape: {output.shape}")
            
            # Verify output shape
            expected_shape = (input_shape[0], config['num_classes'])
            if output.shape == expected_shape:
                print(f"   ✓ Output shape matches expected: {expected_shape}")
            else:
                print(f"   ❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
                return 1
            
            # Test with different batch sizes
            batch_sizes = [1, 2, 4]
            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, *input_shape[1:]).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                print(f"   ✓ Batch size {batch_size}: {test_input.shape} -> {test_output.shape}")
        
        # CRITICAL: Show model summary if requested
        if args.summary:
            print(f"\n📊 Model Summary:")
            print(f"   Architecture: {model}")
            print(f"   Layer details:")
            for i, layer in enumerate(model.layers):
                layer_name = layer.__class__.__name__
                if hasattr(layer, 'weight') and layer.weight is not None:
                    param_count = layer.weight.numel()
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        param_count += layer.bias.numel()
                    print(f"     {i+1:2d}. {layer_name:20s} - {param_count:8,} parameters")
                else:
                    print(f"     {i+1:2d}. {layer_name:20s} - No parameters")
        
        # CRITICAL: Export model if requested
        if args.export:
            export_path = Path(args.export)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n💾 Exporting model to: {export_path}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config,
                'model_class': type(model).__name__,
                'input_shape': config['input_shape'],
                'num_classes': config['num_classes']
            }, export_path)
            print(f"   ✓ Model exported successfully!")
        
        print(f"\n✅ Model building completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
