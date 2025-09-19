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

def compute_global_normalization(df: pd.DataFrame, norm_type: str = "zscore") -> dict:
    """Compute global normalization statistics across all data.
    
    Each sensor channel is normalized independently using its own statistics
    across all samples and time steps.
    
    Args:
        df: DataFrame with sensor data
        norm_type: "zscore" for z-score normalization or "minmax" for min-max normalization
    """
    # Stack all sensor data
    all_data = np.stack(df["x"].tolist(), axis=0)  # (N, C, T)
    
    if norm_type == "zscore":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(all_data, axis=(0, 2))  # (C,) - mean across samples and time
        std = np.std(all_data, axis=(0, 2))    # (C,) - std across samples and time
        return {"mean": mean, "std": std, "norm_type": "zscore"}
    
    elif norm_type == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_vals = np.min(all_data, axis=(0, 2))  # (C,) - min across samples and time
        max_vals = np.max(all_data, axis=(0, 2))  # (C,) - max across samples and time
        return {"min": min_vals, "max": max_vals, "norm_type": "minmax"}
    
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}. Use 'zscore' or 'minmax'")

def compute_per_subject_normalization(df: pd.DataFrame, norm_type: str = "zscore") -> dict:
    """Compute per-subject normalization statistics.
    
    Each sensor channel is normalized independently for each subject
    using that subject's own statistics.
    
    Args:
        df: DataFrame with sensor data
        norm_type: "zscore" for z-score normalization or "minmax" for min-max normalization
    """
    subject_stats = {}
    
    for subject_id in df["subject_id"].unique():
        subject_data = df[df["subject_id"] == subject_id]
        
        # Stack subject's sensor data
        subject_array = np.stack(subject_data["x"].tolist(), axis=0)  # (N_subj, C, T)
        
        if norm_type == "zscore":
            # Z-score normalization: (x - mean) / std
            mean = np.mean(subject_array, axis=(0, 2))  # (C,) - mean across subject's samples and time
            std = np.std(subject_array, axis=(0, 2))    # (C,) - std across subject's samples and time
            subject_stats[subject_id] = {"mean": mean, "std": std, "norm_type": "zscore"}
        
        elif norm_type == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            min_vals = np.min(subject_array, axis=(0, 2))  # (C,) - min across subject's samples and time
            max_vals = np.max(subject_array, axis=(0, 2))  # (C,) - max across subject's samples and time
            subject_stats[subject_id] = {"min": min_vals, "max": max_vals, "norm_type": "minmax"}
        
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}. Use 'zscore' or 'minmax'")
    
    return subject_stats

def apply_global_normalization(df: pd.DataFrame, norm_stats: dict) -> pd.DataFrame:
    """Apply global normalization to the dataframe.
    
    Each channel is normalized using global statistics for that channel.
    """
    df_normalized = df.copy()
    norm_type = norm_stats.get("norm_type", "zscore")
    
    # Apply normalization to each sample using global stats
    normalized_data = []
    for _, row in df.iterrows():
        x = row["x"].copy()  # (C, T)
        
        if norm_type == "zscore":
            # Z-score normalization: (x - mean) / std
            x_norm = (x - norm_stats["mean"][:, np.newaxis]) / (norm_stats["std"][:, np.newaxis] + 1e-8)
        elif norm_type == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            x_norm = (x - norm_stats["min"][:, np.newaxis]) / (norm_stats["max"][:, np.newaxis] - norm_stats["min"][:, np.newaxis] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        normalized_data.append(x_norm)
    
    df_normalized["x"] = normalized_data
    return df_normalized

def apply_per_subject_normalization(df: pd.DataFrame, subject_stats: dict) -> pd.DataFrame:
    """Apply per-subject normalization to the dataframe.
    
    Each channel is normalized using the subject's own statistics for that channel.
    """
    df_normalized = df.copy()
    
    # Apply normalization to each sample using subject-specific stats
    normalized_data = []
    for _, row in df.iterrows():
        x = row["x"].copy()  # (C, T)
        subject_id = row["subject_id"]
        
        if subject_id in subject_stats:
            stats = subject_stats[subject_id]
            norm_type = stats.get("norm_type", "zscore")
            
            if norm_type == "zscore":
                # Z-score normalization: (x - mean) / std
                mean = stats["mean"]  # (C,)
                std = stats["std"]    # (C,)
                x_norm = (x - mean[:, np.newaxis]) / (std[:, np.newaxis] + 1e-8)
            elif norm_type == "minmax":
                # Min-max normalization: (x - min) / (max - min)
                min_vals = stats["min"]  # (C,)
                max_vals = stats["max"]  # (C,)
                x_norm = (x - min_vals[:, np.newaxis]) / (max_vals[:, np.newaxis] - min_vals[:, np.newaxis] + 1e-8)
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")
        else:
            # Fallback to no normalization if subject not found
            x_norm = x
            
        normalized_data.append(x_norm)
    
    df_normalized["x"] = normalized_data
    return df_normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci_har_root", type=str, default="/workspace/data/UCI-HAR")
    ap.add_argument("--pamap2_root", type=str, default="/workspace/data/PAMAP2/PAMAP2_Dataset")
    ap.add_argument("--mhealth_root", type=str, default="/workspace/data/MHEALTH")
    ap.add_argument("--outdir", type=str, default="/workspace/artifacts/preprocessed")
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--win_sec", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--datasets", type=str, nargs="+", default=["uci_har", "pamap2", "mhealth"], 
                   help="Datasets to process: uci_har, pamap2, mhealth")
    ap.add_argument("--generate_both", action="store_true", default=True,
                   help="Generate both raw (unnormalized) and processed (normalized) shards")
    ap.add_argument("--normalization", type=str, choices=["global", "per_subject"], default="global",
                   help="Normalization type: global (across all data) or per_subject (per subject)")
    ap.add_argument("--norm_method", type=str, choices=["zscore", "minmax"], default="zscore",
                   help="Normalization method: zscore (z-score) or minmax (min-max scaling)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- UCI-HAR (already windowed) ----
    if "uci_har" in args.datasets and Path(args.uci_har_root).exists():
        print("[UCI-HAR] loading …")
        df_uci = load_dataset("uci_har", args.uci_har_root)
        print(f"[UCI-HAR] windows: {len(df_uci)}  shape example: {df_uci.iloc[0]['x'].shape}")
        
        # Save raw (unnormalized) shards
        print("[UCI-HAR] saving raw shards...")
        shards_raw = save_npz_shards(df_uci, outdir / "uci_har_raw", shard_size=args.shard_size, prefix="uci_raw")
        shards_raw.to_csv(outdir / "uci_har_raw_manifest.csv", index=False)
        print(f"[UCI-HAR] raw shards: {len(shards_raw)} files")
        
        if args.generate_both:
            if args.normalization == "global":
                # Compute global normalization statistics
                print(f"[UCI-HAR] computing global {args.norm_method} normalization statistics...")
                norm_stats = compute_global_normalization(df_uci, args.norm_method)
                
                # Apply global normalization
                print(f"[UCI-HAR] applying global {args.norm_method} normalization...")
                df_uci_norm = apply_global_normalization(df_uci, norm_stats)
                
                # Save normalization statistics
                np.savez(outdir / "uci_har_norm_stats.npz", **norm_stats)
                print(f"[UCI-HAR] global {args.norm_method} normalization stats saved")
                
            else:  # per_subject
                # Compute per-subject normalization statistics
                print(f"[UCI-HAR] computing per-subject {args.norm_method} normalization statistics...")
                subject_stats = compute_per_subject_normalization(df_uci, args.norm_method)
                
                # Apply per-subject normalization
                print(f"[UCI-HAR] applying per-subject {args.norm_method} normalization...")
                df_uci_norm = apply_per_subject_normalization(df_uci, subject_stats)
                
                # Save normalization statistics
                np.savez(outdir / "uci_har_norm_stats.npz", subject_stats=subject_stats)
                print(f"[UCI-HAR] per-subject {args.norm_method} normalization stats saved")
            
            # Save normalized shards
            print("[UCI-HAR] saving normalized shards...")
            shards_norm = save_npz_shards(df_uci_norm, outdir / "uci_har", shard_size=args.shard_size, prefix="uci")
            shards_norm.to_csv(outdir / "uci_har_manifest.csv", index=False)
            print(f"[UCI-HAR] normalized shards: {len(shards_norm)} files")
    elif "uci_har" in args.datasets:
        print("[UCI-HAR] root not found, skipping.")
    else:
        print("[UCI-HAR] skipped (not in datasets list)")

    # ---- PAMAP2 (stream → resample → window) ----
    if "pamap2" in args.datasets and Path(args.pamap2_root).exists():
        print("[PAMAP2] loading …")
        df_pam = load_dataset("pamap2", args.pamap2_root, win_sec=args.win_sec, overlap=args.overlap, fs=args.fs)
        if len(df_pam) > 0:
            print(f"[PAMAP2] windows: {len(df_pam)}  shape example: {df_pam.iloc[0]['x'].shape}")
            
            # Save raw (unnormalized) shards
            print("[PAMAP2] saving raw shards...")
            shards_raw = save_npz_shards(df_pam, outdir / "pamap2_raw", shard_size=args.shard_size, prefix="pam_raw")
            shards_raw.to_csv(outdir / "pamap2_raw_manifest.csv", index=False)
            print(f"[PAMAP2] raw shards: {len(shards_raw)} files")
            
            if args.generate_both:
                if args.normalization == "global":
                    # Compute global normalization statistics
                    print(f"[PAMAP2] computing global {args.norm_method} normalization statistics...")
                    norm_stats = compute_global_normalization(df_pam, args.norm_method)
                    
                    # Apply global normalization
                    print(f"[PAMAP2] applying global {args.norm_method} normalization...")
                    df_pam_norm = apply_global_normalization(df_pam, norm_stats)
                    
                    # Save normalization statistics
                    np.savez(outdir / "pamap2_norm_stats.npz", **norm_stats)
                    print(f"[PAMAP2] global {args.norm_method} normalization stats saved")
                    
                else:  # per_subject
                    # Compute per-subject normalization statistics
                    print(f"[PAMAP2] computing per-subject {args.norm_method} normalization statistics...")
                    subject_stats = compute_per_subject_normalization(df_pam, args.norm_method)
                    
                    # Apply per-subject normalization
                    print(f"[PAMAP2] applying per-subject {args.norm_method} normalization...")
                    df_pam_norm = apply_per_subject_normalization(df_pam, subject_stats)
                    
                    # Save normalization statistics
                    np.savez(outdir / "pamap2_norm_stats.npz", subject_stats=subject_stats)
                    print(f"[PAMAP2] per-subject {args.norm_method} normalization stats saved")
                
                # Save normalized shards
                print("[PAMAP2] saving normalized shards...")
                shards_norm = save_npz_shards(df_pam_norm, outdir / "pamap2", shard_size=args.shard_size, prefix="pam")
                shards_norm.to_csv(outdir / "pamap2_manifest.csv", index=False)
                print(f"[PAMAP2] normalized shards: {len(shards_norm)} files")
        else:
            print("[PAMAP2] no data found, skipping.")
    elif "pamap2" in args.datasets:
        print("[PAMAP2] root not found, skipping.")
    else:
        print("[PAMAP2] skipped (not in datasets list)")

    # ---- MHEALTH (logs → window) ----
    if "mhealth" in args.datasets and Path(args.mhealth_root).exists():
        print("[MHEALTH] loading …")
        df_mh = load_dataset("mhealth", args.mhealth_root, fs=args.fs, win_sec=args.win_sec, overlap=args.overlap)
        if len(df_mh) > 0:
            print(f"[MHEALTH] windows: {len(df_mh)}  shape example: {df_mh.iloc[0]['x'].shape}")
            
            # Save raw (unnormalized) shards
            print("[MHEALTH] saving raw shards...")
            shards_raw = save_npz_shards(df_mh, outdir / "mhealth_raw", shard_size=args.shard_size, prefix="mh_raw")
            shards_raw.to_csv(outdir / "mhealth_raw_manifest.csv", index=False)
            print(f"[MHEALTH] raw shards: {len(shards_raw)} files")
            
            if args.generate_both:
                if args.normalization == "global":
                    # Compute global normalization statistics
                    print(f"[MHEALTH] computing global {args.norm_method} normalization statistics...")
                    norm_stats = compute_global_normalization(df_mh, args.norm_method)
                    
                    # Apply global normalization
                    print(f"[MHEALTH] applying global {args.norm_method} normalization...")
                    df_mh_norm = apply_global_normalization(df_mh, norm_stats)
                    
                    # Save normalization statistics
                    np.savez(outdir / "mhealth_norm_stats.npz", **norm_stats)
                    print(f"[MHEALTH] global {args.norm_method} normalization stats saved")
                    
                else:  # per_subject
                    # Compute per-subject normalization statistics
                    print(f"[MHEALTH] computing per-subject {args.norm_method} normalization statistics...")
                    subject_stats = compute_per_subject_normalization(df_mh, args.norm_method)
                    
                    # Apply per-subject normalization
                    print(f"[MHEALTH] applying per-subject {args.norm_method} normalization...")
                    df_mh_norm = apply_per_subject_normalization(df_mh, subject_stats)
                    
                    # Save normalization statistics
                    np.savez(outdir / "mhealth_norm_stats.npz", subject_stats=subject_stats)
                    print(f"[MHEALTH] per-subject {args.norm_method} normalization stats saved")
                
                # Save normalized shards
                print("[MHEALTH] saving normalized shards...")
                shards_norm = save_npz_shards(df_mh_norm, outdir / "mhealth", shard_size=args.shard_size, prefix="mh")
                shards_norm.to_csv(outdir / "mhealth_manifest.csv", index=False)
                print(f"[MHEALTH] normalized shards: {len(shards_norm)} files")
        else:
            print("[MHEALTH] no data found, skipping.")
    elif "mhealth" in args.datasets:
        print("[MHEALTH] root not found, skipping.")
    else:
        print("[MHEALTH] skipped (not in datasets list)")

    print(f"Done. Shards under: {outdir}")

if __name__ == "__main__":
    main()
