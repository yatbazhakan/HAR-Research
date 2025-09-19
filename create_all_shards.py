#!/usr/bin/env python3
"""
Create all dataset shards for color coding experiments.

This script runs the preprocessing pipeline to create both regular and raw shards
for all available HAR datasets (UCI-HAR, PAMAP2, MHEALTH).
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run preprocessing for all datasets."""
    print("Creating all dataset shards...")
    print("=" * 50)
    
    # Check if data directories exist
    data_dirs = {
        "UCI-HAR": "data/UCI-HAR",
        "PAMAP2": "data/PAMAP2/PAMAP2_Dataset", 
        "MHEALTH": "data/MHEALTH"
    }
    
    missing_dirs = []
    for name, path in data_dirs.items():
        if not Path(path).exists():
            missing_dirs.append(f"{name}: {path}")
    
    if missing_dirs:
        print("Missing data directories:")
        for missing in missing_dirs:
            print(f"  - {missing}")
        print("\nPlease ensure all datasets are downloaded and extracted.")
        return
    
    # Run preprocessing
    cmd = [
        "python", "scripts/preprocess.py",
        "--uci_har_root", "data/UCI-HAR",
        "--pamap2_root", "data/PAMAP2/PAMAP2_Dataset", 
        "--mhealth_root", "data/MHEALTH",
        "--outdir", "artifacts/preprocessed",
        "--fs", "50",
        "--win_sec", "3.0",
        "--overlap", "0.5",
        "--shard_size", "5000",
        "--datasets", "uci_har", "pamap2", "mhealth"
    ]
    
    print("Running preprocessing command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Preprocessing completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed with error code {e.returncode}")
        print("\nError output:")
        print(e.stderr)
        return
    
    # Check output directories
    output_dirs = [
        "artifacts/preprocessed/uci_har",
        "artifacts/preprocessed/uci_har_raw",
        "artifacts/preprocessed/pamap2", 
        "artifacts/preprocessed/pamap2_raw",
        "artifacts/preprocessed/mhealth",
        "artifacts/preprocessed/mhealth_raw"
    ]
    
    print("\nChecking output directories:")
    for dir_path in output_dirs:
        path = Path(dir_path)
        if path.exists():
            npz_files = list(path.glob("*.npz"))
            print(f"  ✓ {dir_path}: {len(npz_files)} shard files")
        else:
            print(f"  ✗ {dir_path}: Not found")
    
    print("\nShard creation complete!")
    print("You can now run color coding examples:")
    print("  python examples/color_coding_all_datasets.py")

if __name__ == "__main__":
    main()
