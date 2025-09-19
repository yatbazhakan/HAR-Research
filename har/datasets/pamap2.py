from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import torch
from torch.utils.data import Dataset
from .shards import NPZShardsDataset

# PAMAP2 sampling: IMU ~100Hz, heart rate ~9Hz; timestamps in seconds.
# We'll resample to 50Hz. Activity id 0 = "other/unknown".

# Column indices per PAMAP2 doc (0-based after reading with header=None):
# [0] timestamp, [1] activity_id, [2] heart_rate,
# IMU1 (hand):   3..19, IMU2 (chest): 20..36, IMU3 (ankle): 37..53
# Within each IMU block: acc_16g(3), acc_6g(3), gyro(3), mag(3), orientation(4)
# We'll use IMU2 (chest) acc_6g (23..25), gyro (26..28), mag (29..31)

COLS = {
    "timestamp": 0,
    "activity_id": 1,
    "heart_rate": 2,
}
COLS_CHEST = {
    "acc":  [23,24,25],   # acc_6g x,y,z
    "gyro": [26,27,28],
    "mag":  [29,30,31],
}

VALID_ACTS = {  # Keep as-is; you can later remap to your class set
    0,1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19
}

def _read_subject_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep='\s+', header=None, na_values=['NaN','nan'])
    df = df.rename(columns={COLS["timestamp"]:"t", COLS["activity_id"]:"y"})
    # Keep needed columns
    keep = ["t","y"] + COLS_CHEST["acc"] + COLS_CHEST["gyro"] + COLS_CHEST["mag"]
    df = df[keep]
    # Rename for clarity
    rename = {}
    for i,k in enumerate(["acc_x","acc_y","acc_z"]): rename[COLS_CHEST["acc"][i]] = k
    for i,k in enumerate(["gyro_x","gyro_y","gyro_z"]): rename[COLS_CHEST["gyro"][i]] = k
    for i,k in enumerate(["mag_x","mag_y","mag_z"]): rename[COLS_CHEST["mag"][i]] = k
    df = df.rename(columns=rename)

    # Drop rows with all-NaN sensors
    sens_cols = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]
    df = df.dropna(subset=sens_cols, how="all")
    # Forward-fill activity id
    df["y"] = df["y"].ffill().bfill().astype(int)
    # Filter to known ids
    df = df[df["y"].isin(VALID_ACTS)]
    return df

def _resample_to(df: pd.DataFrame, fs_target: int = 50) -> pd.DataFrame:
    # Convert numeric seconds to datetime index for time interpolation
    df = df.copy()
    t_dt = pd.to_datetime(df["t"], unit="s")
    df = df.drop(columns=["t"]).set_index(t_dt)
    # Create uniform timeline as DatetimeIndex
    freq = pd.to_timedelta(1.0 / fs_target, unit="s")
    t_new = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, inclusive="left")
    # Interpolate along time and reindex to uniform grid
    df_new = df.reindex(df.index.union(t_new)).interpolate(method="time").reindex(t_new)
    df_new.index.name = "t"
    df_new = df_new.reset_index()
    # Also keep numeric seconds if needed by downstream
    df_new["t"] = df_new["t"].astype("int64") / 1e9
    return df_new

def _windowize(df: pd.DataFrame, win_sec: float, overlap: float, fs: int) -> Iterable[dict]:
    sens_cols = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]
    L = int(round(win_sec*fs)); H = int(round(L*(1-overlap)))
    xarr = df[sens_cols].to_numpy(np.float32)   # (N, 9)
    yarr = df["y"].to_numpy(int)
    N = len(df)
    start = 0
    while start + L <= N:
        seg = xarr[start:start+L]  # (L, 9)
        # Majority label in window (simple)
        y = int(pd.Series(yarr[start:start+L]).mode().iloc[0])
        x = seg.T  # (C=9, T=L)
        yield {"x": x, "y": y, "start_idx": start}
        start += H

def load_pamap2_raw_stream(root_dir: str, win_sec: float = 3.0, overlap: float = 0.5, fs: int = 50) -> pd.DataFrame:
    """
    Load PAMAP2 raw data with all available sensors for color coding experiments.
    
    This function loads all available sensor data from PAMAP2 dataset and creates
    windows suitable for color coding visualization and analysis.
    
    Args:
        root_dir: Path to PAMAP2 dataset directory
        win_sec: Window size in seconds (default: 3.0)
        overlap: Overlap between windows (default: 0.5)
        fs: Target sampling frequency (default: 50)
    
    Returns:
        DataFrame with raw sensor data windows for color coding
    """
    root = Path(root_dir)
    files = sorted(list(root.glob("*.dat"))) + sorted(list(root.glob("Optional/*.dat")))
    rows = []
    
    for p in files:
        subj_id = p.stem.split("_")[-1]
        df = pd.read_csv(p, sep='\s+', header=None, na_values=['NaN','nan'])
        
        # Use all available sensor columns (0-52) plus activity label (53)
        # Keep timestamp, activity_id, heart_rate, and all IMU data
        df = df.rename(columns={0:"t", 1:"y"})
        
        # Keep all sensor data columns (2-52) - all IMU sensors
        sensor_cols = list(range(2, 53))  # Columns 2-52
        df_sensors = df[["t", "y"] + sensor_cols].copy()
        
        # Drop rows with all-NaN sensors
        df_sensors = df_sensors.dropna(subset=sensor_cols, how="all")
        if df_sensors.empty:
            continue
            
        # Forward-fill activity id
        df_sensors["y"] = df_sensors["y"].ffill().bfill().astype(int)
        
        # Filter to known activities
        df_sensors = df_sensors[df_sensors["y"].isin(VALID_ACTS)]
        if df_sensors.empty:
            continue
        
        # Resample to target frequency
        df_sensors = _resample_to(df_sensors, fs_target=fs)
        
        # Create windows manually (don't use _windowize as it expects processed columns)
        sens_cols = sensor_cols  # Use numeric column indices
        L = int(round(win_sec*fs))
        H = int(round(L*(1-overlap)))
        xarr = df_sensors[sens_cols].to_numpy(np.float32)   # (N, 51)
        yarr = df_sensors["y"].to_numpy(int)
        N = len(df_sensors)
        start = 0
        
        while start + L <= N:
            seg = xarr[start:start+L]  # (L, 51)
            # Majority label in window (simple)
            y = int(pd.Series(yarr[start:start+L]).mode().iloc[0])
            x = seg.T  # (C=51, T=L)
            
            rows.append({
                "x": x.astype(np.float32),  # (51, win_samples) - all sensor channels
                "y": y,
                "subject_id": subj_id,
                "dataset": "pamap2_raw",
                "split": "all",
                "fs": fs,
                "channels": [f"sensor_{i}" for i in range(51)],  # 51 sensor channels
                "start_idx": start
            })
            start += H
    
    return pd.DataFrame(rows)


def load_pamap2_stream(root_dir: str, win_sec: float = 3.0, overlap: float = 0.5, fs: int = 100) -> pd.DataFrame:
    """
    Reads all PAMAP2 subject files (Protocol and Optional) under root_dir,
    resamples to fs, windows, returns a DataFrame with rows:
      x (np.ndarray CxT), y (int), subject_id (str), dataset="pamap2", split="all"
    """
    root = Path(root_dir)
    files = sorted(list(root.glob("*.dat"))) + sorted(list(root.glob("Optional/*.dat")))
    rows = []
    for p in files:
        subj_id = p.stem.split("_")[-1]
        df = pd.read_csv(p, sep='\s+', header=None, na_values=['NaN','nan'])
        
        # Use all available sensor columns (0-52) plus activity label (53)
        df = df.rename(columns={0:"t", 1:"y"})
        
        # Keep all sensor data columns (2-52) - all IMU sensors
        sensor_cols = list(range(2, 53))  # Columns 2-52
        df_sensors = df[["t", "y"] + sensor_cols].copy()
        
        # Drop rows with all-NaN sensors
        df_sensors = df_sensors.dropna(subset=sensor_cols, how="all")
        if df_sensors.empty:
            continue
            
        # Forward-fill activity id
        df_sensors["y"] = df_sensors["y"].ffill().bfill().astype(int)
        
        # Filter to known activities
        df_sensors = df_sensors[df_sensors["y"].isin(VALID_ACTS)]
        if df_sensors.empty:
            continue
        
        # Resample to target frequency
        df_sensors = _resample_to(df_sensors, fs_target=fs)
        
        # Create windows manually
        sens_cols = sensor_cols  # Use numeric column indices
        L = int(round(win_sec*fs))
        H = int(round(L*(1-overlap)))
        xarr = df_sensors[sens_cols].to_numpy(np.float32)   # (N, 51)
        yarr = df_sensors["y"].to_numpy(int)
        N = len(df_sensors)
        start = 0
        
        while start + L <= N:
            seg = xarr[start:start+L]  # (L, 51)
            # Majority label in window (simple)
            y = int(pd.Series(yarr[start:start+L]).mode().iloc[0])
            x = seg.T  # (C=51, T=L)
            
            rows.append({
                "x": x.astype(np.float32),  # (51, win_samples) - all sensor channels
                "y": y,
                "subject_id": subj_id,
                "dataset": "pamap2",
                "split": "all",
                "fs": fs,
                "channels": [f"sensor_{i}" for i in range(51)],  # 51 sensor channels
                "start_idx": start
            })
            start += H
    
    return pd.DataFrame(rows)


class PAMAP2Dataset(Dataset):
    """
    PyTorch Dataset class for PAMAP2 data using preprocessed NPZ shards.
    
    This class wraps the NPZShardsDataset to provide a clean interface for
    loading PAMAP2 data from preprocessed shard files.
    """
    
    def __init__(self, shards_glob: str, transform=None, split: str = "all"):
        """
        Initialize PAMAP2 dataset.
        
        Args:
            shards_glob: Glob pattern for NPZ shard files (e.g., "data/pamap2/*.npz")
            transform: Optional NormStats object for normalization
            split: Data split to use ("train", "test", "val", or "all")
        """
        self.shards_dataset = NPZShardsDataset(shards_glob, split, stats=transform)
        self.transform = transform
        
    def __len__(self):
        return len(self.shards_dataset)
    
    def __getitem__(self, idx):
        # NPZShardsDataset already applies normalization internally
        x, y = self.shards_dataset[idx]
        return x, y
    
    def get_dataset_statistics(self) -> dict:
        """
        Extract comprehensive dataset statistics for PAMAP2 dataset.
        
        Returns:
            dict: Dictionary containing:
                - total_samples: Total number of samples
                - num_subjects: Number of unique subjects
                - num_activities: Number of unique activities
                - activities_per_subject: Dictionary mapping subject_id to list of activities
                - samples_per_subject: Dictionary mapping subject_id to sample count
                - samples_per_activity: Dictionary mapping activity_id to sample count
                - subject_activity_matrix: Dictionary showing activity distribution per subject
        """
        # Load all data to compute statistics
        all_data = []
        for i in range(len(self.shards_dataset)):
            x, y = self.shards_dataset[i]
            # Get subject_id from the shard if available
            file_idx, sample_idx = self.shards_dataset.index[i]
            file_path = self.shards_dataset._map[file_idx][0]
            z = np.load(file_path, allow_pickle=False)
            if "subject_id" in z.files:
                subject_id = z["subject_id"][sample_idx]
            else:
                subject_id = f"unknown_{file_idx}"
            all_data.append({"subject_id": subject_id, "activity": y})
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_data)
        
        # Calculate statistics
        stats = {
            "total_samples": len(df),
            "num_subjects": df["subject_id"].nunique(),
            "num_activities": df["activity"].nunique(),
            "unique_subjects": sorted(df["subject_id"].unique().tolist()),
            "unique_activities": sorted(df["activity"].unique().tolist()),
            "activities_per_subject": {},
            "samples_per_subject": {},
            "samples_per_activity": {},
            "subject_activity_matrix": {}
        }
        
        # Activities per subject
        for subject in df["subject_id"].unique():
            subject_data = df[df["subject_id"] == subject]
            activities = sorted(subject_data["activity"].unique().tolist())
            stats["activities_per_subject"][subject] = activities
            stats["samples_per_subject"][subject] = len(subject_data)
        
        # Samples per activity
        for activity in df["activity"].unique():
            activity_data = df[df["activity"] == activity]
            stats["samples_per_activity"][activity] = len(activity_data)
        
        # Subject-activity matrix
        for subject in df["subject_id"].unique():
            subject_data = df[df["subject_id"] == subject]
            activity_counts = subject_data["activity"].value_counts().to_dict()
            stats["subject_activity_matrix"][subject] = activity_counts
        
        return stats