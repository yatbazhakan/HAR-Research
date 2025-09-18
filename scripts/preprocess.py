"""
CRITICAL: Data Preprocessing Script for HAR Datasets

This script is responsible for preprocessing raw HAR datasets into a standardized
format suitable for training neural networks. It handles multiple datasets and
converts them into NPZ shard files for efficient loading during training.

Key Functions:
1. Load raw datasets (UCI-HAR, PAMAP2, MHEALTH)
2. Apply windowing and segmentation
3. Normalize and standardize data
4. Save as compressed NPZ shards for efficient access
5. Generate dataset manifests and statistics

CRITICAL: This script must be run before training any models.
The output NPZ shards are required by the training pipeline.

Usage:
    python scripts/preprocess.py --uci_har_root /path/to/uci_har --outdir /path/to/output
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# CRITICAL: Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import dataset loading functions
from har.datasets import load_dataset

def save_npz_shards(df: pd.DataFrame, out_dir: Path, shard_size: int = 5000, prefix: str = "all"):
    """
    CRITICAL: Save DataFrame as compressed NPZ shard files
    
    This function splits a large DataFrame into smaller, compressed NPZ files
    for efficient loading during training. Each shard contains a subset of
    the data along with metadata.
    
    Args:
        df: DataFrame containing preprocessed HAR data
        out_dir: Output directory for shard files
        shard_size: Number of samples per shard (default: 5000)
        prefix: Prefix for shard filenames (default: "all")
    
    Returns:
        DataFrame with shard information (file paths and sample counts)
    
    CRITICAL: Each NPZ shard contains:
    - X: Sensor data array (N, C, T) where N=samples, C=channels, T=time_steps
    - y: Activity labels (N,) 
    - dataset: Dataset name
    - split: Train/test split
    - subject_id: Subject identifiers for LOSO CV
    - start_idx: Starting index in original sequence
    - fs: Sampling frequency
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    shards = []
    
    # CRITICAL: Split data into shards of specified size
    for s in range(0, n, shard_size):
        part = df.iloc[s:s+shard_size]
        
        # CRITICAL: Stack sensor data into 3D array (N, C, T)
        X = np.stack(part["x"].tolist(), axis=0)    # (N,C,T)
        y = part["y"].to_numpy(int)                  # Activity labels
        
        # CRITICAL: Extract metadata for each sample
        meta = part[["dataset","split","subject_id","start_idx","fs"]].to_dict(orient="list")
        
        # CRITICAL: Save as compressed NPZ file
        fname = out_dir / f"{prefix}_{s:08d}.npz"
        np.savez_compressed(fname, X=X, y=y, **meta)
        
        # Track shard information
        shards.append({"file": str(fname), "n": len(part)})
    
    return pd.DataFrame(shards)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci_har_root", type=str, default="/workspace/data/UCI-HAR")
    ap.add_argument("--pamap2_root", type=str, default="/workspace/data/PAMAP2")
    ap.add_argument("--mhealth_root", type=str, default="/workspace/data/MHEALTH")
    ap.add_argument("--outdir", type=str, default="/workspace/artifacts/preprocessed")
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--win_sec", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--shard_size", type=int, default=5000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- UCI-HAR (already windowed) ----
    if Path(args.uci_har_root).exists():
        print("[UCI-HAR] loading …")
        df_uci = load_dataset("uci_har", args.uci_har_root)
        print(f"[UCI-HAR] windows: {len(df_uci)}  shape example: {df_uci.iloc[0]['x'].shape}")
        shards = save_npz_shards(df_uci, outdir / "uci_har", shard_size=args.shard_size, prefix="uci")
        shards.to_csv(outdir / "uci_har_manifest.csv", index=False)
    else:
        print("[UCI-HAR] root not found, skipping.")

    # ---- PAMAP2 (stream → resample → window) ----
    if Path(args.pamap2_root).exists():
        print("[PAMAP2] loading …")
        df_pam = load_dataset("pamap2", args.pamap2_root, win_sec=args.win_sec, overlap=args.overlap, fs=args.fs)
        print(f"[PAMAP2] windows: {len(df_pam)}  shape example: {df_pam.iloc[0]['x'].shape}")
        shards = save_npz_shards(df_pam, outdir / "pamap2", shard_size=args.shard_size, prefix="pam")
        shards.to_csv(outdir / "pamap2_manifest.csv", index=False)
    else:
        print("[PAMAP2] root not found, skipping.")

    # ---- MHEALTH (logs → window) ----
    if Path(args.mhealth_root).exists():
        print("[MHEALTH] loading …")
        df_mh = load_dataset("mhealth", args.mhealth_root, fs=args.fs, win_sec=args.win_sec, overlap=args.overlap)
        print(f"[MHEALTH] windows: {len(df_mh)}  shape example: {df_mh.iloc[0]['x'].shape}")
        shards = save_npz_shards(df_mh, outdir / "mhealth", shard_size=args.shard_size, prefix="mh")
        shards.to_csv(outdir / "mhealth_manifest.csv", index=False)
    else:
        print("[MHEALTH] root not found, skipping.")

    print(f"Done. Shards under: {outdir}")

if __name__ == "__main__":
    main()
