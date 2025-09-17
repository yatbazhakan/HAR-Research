#!/usr/bin/env python3
"""
Launcher script for the Neural Network Model Builder GUI
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the model builder GUI"""
    try:
        # Get the script path
        script_path = Path(__file__).parent / "model_builder_gui.py"
        
        # Run the model builder
        subprocess.run([sys.executable, str(script_path)])
        
    except Exception as e:
        print(f"Error launching model builder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
