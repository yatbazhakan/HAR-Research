#!/usr/bin/env python3
"""
Test script for the config-based experiment system.
This script tests the basic functionality without running full experiments.
"""

import sys
import yaml
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_config_loading():
    """Test loading of configuration files."""
    print("Testing configuration loading...")
    
    # Test HAR experiment config
    har_config_path = REPO_ROOT / "configs" / "har_experiment_example.yaml"
    if har_config_path.exists():
        with open(har_config_path, 'r') as f:
            har_config = yaml.safe_load(f)
        print(f"✓ HAR experiment config loaded successfully")
        print(f"  - Experiment name: {har_config.get('experiment_name', 'N/A')}")
        print(f"  - Device: {har_config.get('device', 'N/A')}")
        print(f"  - Method: {har_config.get('method', 'N/A')}")
    else:
        print(f"✗ HAR experiment config not found: {har_config_path}")
        return False
    
    # Test sweep config
    sweep_config_path = REPO_ROOT / "configs" / "har_sweep_example.yaml"
    if sweep_config_path.exists():
        with open(sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
        print(f"✓ Sweep config loaded successfully")
        print(f"  - Method: {sweep_config.get('method', 'N/A')}")
        print(f"  - Metric: {sweep_config.get('metric', {}).get('name', 'N/A')}")
        print(f"  - Parameters: {len(sweep_config.get('parameters', {}))}")
    else:
        print(f"✗ Sweep config not found: {sweep_config_path}")
        return False
    
    return True

def test_script_imports():
    """Test that all scripts can be imported."""
    print("\nTesting script imports...")
    
    scripts_to_test = [
        "scripts.run_har_experiment",
        "scripts.run_har_sweep", 
        "scripts.run_experiment_from_config",
        "scripts.config_experiment_gui"
    ]
    
    all_imports_ok = True
    
    for script in scripts_to_test:
        try:
            __import__(script)
            print(f"✓ {script} imported successfully")
        except ImportError as e:
            print(f"✗ {script} import failed: {e}")
            all_imports_ok = False
        except Exception as e:
            print(f"✗ {script} import error: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_config_validation():
    """Test basic config validation."""
    print("\nTesting configuration validation...")
    
    # Load HAR config
    har_config_path = REPO_ROOT / "configs" / "har_experiment_example.yaml"
    with open(har_config_path, 'r') as f:
        har_config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = [
        'experiment_name',
        'device', 
        'method',
        'har_experiment'
    ]
    
    for field in required_fields:
        if field in har_config:
            print(f"✓ Required field '{field}' present")
        else:
            print(f"✗ Required field '{field}' missing")
            return False
    
    # Check HAR experiment structure
    har_exp = har_config['har_experiment']
    har_required_fields = [
        'experiment_name',
        'method',
        'dataset',
        'dataloader'
    ]
    
    for field in har_required_fields:
        if field in har_exp:
            print(f"✓ HAR experiment field '{field}' present")
        else:
            print(f"✗ HAR experiment field '{field}' missing")
            return False
    
    return True

def test_command_generation():
    """Test command generation."""
    print("\nTesting command generation...")
    
    try:
        from scripts.config_experiment_gui import ConfigExperimentGUI
        import tkinter as tk
        
        # Create a minimal tkinter root (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create GUI instance
        gui = ConfigExperimentGUI(root)
        
        # Test command generation
        gui.config_var.set(str(REPO_ROOT / "configs" / "har_experiment_example.yaml"))
        gui.generate_command()
        
        command = gui.command_text.get(1.0, tk.END).strip()
        if command and "python scripts/run_experiment_from_config.py" in command:
            print("✓ Command generation works")
            print(f"  Generated command: {command}")
        else:
            print("✗ Command generation failed")
            return False
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ Command generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Config-based Experiment System Test")
    print("=" * 40)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Script Imports", test_script_imports),
        ("Configuration Validation", test_config_validation),
        ("Command Generation", test_command_generation)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Config-based experiment system is ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
