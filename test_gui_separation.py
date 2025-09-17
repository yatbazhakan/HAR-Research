#!/usr/bin/env python3
"""
Test script to verify the separated GUI tools work correctly
"""

import sys
import subprocess
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_imports():
    """Test that all GUI modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test training GUI
        import scripts.training_gui
        print("✅ training_gui.py imports successfully")
    except Exception as e:
        print(f"❌ training_gui.py import failed: {e}")
        
    try:
        # Test sweep GUI
        import scripts.sweep_gui
        print("✅ sweep_gui.py imports successfully")
    except Exception as e:
        print(f"❌ sweep_gui.py import failed: {e}")
        
    try:
        # Test launcher GUI
        import scripts.experiment_launcher
        print("✅ experiment_launcher.py imports successfully")
    except Exception as e:
        print(f"❌ experiment_launcher.py import failed: {e}")
        
    try:
        # Test new sweep script
        import scripts.run_sweep_new
        print("✅ run_sweep_new.py imports successfully")
    except Exception as e:
        print(f"❌ run_sweep_new.py import failed: {e}")


def test_script_execution():
    """Test that scripts can be executed"""
    print("\nTesting script execution...")
    
    scripts_to_test = [
        "scripts/training_gui.py",
        "scripts/sweep_gui.py", 
        "scripts/experiment_launcher.py",
        "scripts/run_sweep_new.py"
    ]
    
    for script in scripts_to_test:
        script_path = REPO_ROOT / script
        if script_path.exists():
            try:
                # Test with --help flag
                result = subprocess.run([sys.executable, str(script_path), "--help"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 or "usage:" in result.stdout.lower():
                    print(f"✅ {script} executes successfully")
                else:
                    print(f"⚠️  {script} executes but may have issues")
            except subprocess.TimeoutExpired:
                print(f"⚠️  {script} timed out (GUI may have opened)")
            except Exception as e:
                print(f"❌ {script} execution failed: {e}")
        else:
            print(f"❌ {script} not found")


def test_file_structure():
    """Test that all expected files exist"""
    print("\nTesting file structure...")
    
    expected_files = [
        "scripts/training_gui.py",
        "scripts/sweep_gui.py",
        "scripts/experiment_launcher.py", 
        "scripts/run_sweep_new.py",
        "docs/GUI_TOOLS.md"
    ]
    
    for file_path in expected_files:
        full_path = REPO_ROOT / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")


def test_config_directories():
    """Test that configuration directories exist"""
    print("\nTesting configuration directories...")
    
    config_dirs = [
        "artifacts/sweep_configs",
        "artifacts/plots",
        "logs"
    ]
    
    for dir_path in config_dirs:
        full_path = REPO_ROOT / dir_path
        if full_path.exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️  {dir_path} missing (will be created when needed)")


def main():
    print("HAR GUI Separation Test")
    print("=" * 50)
    
    test_imports()
    test_script_execution()
    test_file_structure()
    test_config_directories()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo use the new GUI tools:")
    print("1. python scripts/experiment_launcher.py  # Main launcher")
    print("2. python scripts/training_gui.py         # Single experiments")
    print("3. python scripts/sweep_gui.py            # Hyperparameter sweeps")


if __name__ == "__main__":
    main()
