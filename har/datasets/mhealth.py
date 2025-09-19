from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from .shards import NPZShardsDataset

# MHEALTH: each file has columns (commonly 23): timestamp + multiple sensors + activity label.
# We'll try a robust parse and select: chest acc (3), wrist gyro (3), ankle mag (3), ECG (1).
# If a group is missing, we fill with zeros.

def _guess_schema(df: pd.DataFrame) -> dict:
    """
    Try common MHEALTH column layouts; return a mapping dict.
    Fallbacks fill missing groups with zeros.
    """
    ncol = df.shape[1]
    # Heuristic: last column often activity id
    act_col = ncol - 1
    # Try a common mapping (based on popular distribution):
    # 0:timestamp, 1..3:acc_chest, 4..5:ecg1,ecg2, 6..8:acc_wrist, 9..11:gyro_wrist, 12..14:mag_wrist,
    # 15..17:acc_ankle, 18..20:gyro_ankle, 21..23:mag_ankle, 24:activity  (if ncol==25)
    mapping = {
        "acc_chest": [1,2,3],
        "ecg":      [4,5],
        "gyro_wrist":[9,10,11],
        "mag_ankle":[21,22,23] if ncol>23 else [],
        "activity": act_col
    }
    # If columns don't exist, adapt conservatively
    for k, idxs in list(mapping.items()):
        if k == "activity":
            # keep as single integer index, clamp if out of range
            mapping[k] = min(max(0, idxs), ncol - 1)
        else:
            mapping[k] = [i for i in idxs if i < ncol]
    return mapping

def _read_subject_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep='\s+', header=None)
    mp = _guess_schema(df)
    # Build signals, fill missing with zeros
    def get_cols(idxs): 
        if not idxs: 
            return np.zeros((len(df), 0), dtype=np.float32)
        return df.iloc[:, idxs].to_numpy(np.float32)

    acc_chest = get_cols(mp["acc_chest"])
    gyro_wrist = get_cols(mp["gyro_wrist"])
    mag_ankle = get_cols(mp["mag_ankle"])
    ecg_cols = get_cols(mp["ecg"])
    if ecg_cols.shape[1] == 0:
        ecg = np.zeros((len(df),1), dtype=np.float32)
    elif ecg_cols.shape[1] == 1:
        ecg = ecg_cols
    else:
        ecg = ecg_cols.mean(axis=1, keepdims=True)  # average two leads

    # Activity
    y = df.iloc[:, mp["activity"]].astype(int).to_numpy()

    # Concatenate: acc_chest(3) + gyro_wrist(3?) + mag_ankle(3?) + ecg(1) → aim for C up to 10
    X = np.concatenate([
        acc_chest, 
        gyro_wrist if gyro_wrist.shape[1] else np.zeros((len(df),3), np.float32),
        mag_ankle if mag_ankle.shape[1] else np.zeros((len(df),3), np.float32),
        ecg
    ], axis=1)  # (N, C)

    cols = []
    cols += ["acc_x","acc_y","acc_z"]
    cols += ["gyro_x","gyro_y","gyro_z"]
    cols += ["mag_x","mag_y","mag_z"]
    cols += ["ecg"]
    # Trim to actual C
    cols = cols[:X.shape[1]]
    out = pd.DataFrame(X, columns=cols)
    out["y"] = y
    return out

def _windowize_df(df: pd.DataFrame, fs: int, win_sec: float, overlap: float, subj_id: str):
    cols = [c for c in df.columns if c != "y"]
    X = df[cols].to_numpy(np.float32)
    y = df["y"].to_numpy(int)
    L = int(round(fs*win_sec)); H = int(round(L*(1-overlap)))
    i = 0
    while i + L <= len(df):
        seg = X[i:i+L]
        lab = int(pd.Series(y[i:i+L]).mode().iloc[0])
        yield {
            "x": seg.T,  # (C,T)
            "y": lab,
            "subject_id": subj_id,
            "start_idx": i,
            "fs": fs,
            "channels": cols
        }
        i += H

def load_mhealth_raw_stream(root_dir: str, fs: int = 50, win_sec: float = 3.0, overlap: float = 0.5) -> pd.DataFrame:
    """
    Reads all mHealth_subject*.log files with ALL 23 channels for color coding.
    """
    root = Path(root_dir)
    files = sorted(list(root.glob("mHealth_subject*.log")))
    rows = []
    for p in files:
        subj = p.stem.split("subject")[-1]
        df = pd.read_csv(p, sep='\s+', header=None)
        
        # Use all 23 channels (columns 0-22, skip column 23 which is label)
        X = df.iloc[:, :23].to_numpy(np.float32)  # (N, 23)
        y = df.iloc[:, 23].astype(int).to_numpy()  # (N,) - column 24 (1-indexed) = column 23 (0-indexed)
        
        # Resample if needed
        approx_src = 50
        stride = max(1, int(round(approx_src / fs)))
        if stride > 1:
            X = X[::stride]
            y = y[::stride]
        
        # Create windows
        win_samples = int(win_sec * fs)
        step_samples = int(win_samples * (1 - overlap))
        
        for i in range(0, len(X) - win_samples + 1, step_samples):
            window_x = X[i:i + win_samples]  # (win_samples, 23)
            window_y = y[i:i + win_samples]
            
            # Use majority vote for label
            unique, counts = np.unique(window_y, return_counts=True)
            label = unique[np.argmax(counts)]
            
            rows.append({
                "x": window_x,
                "y": label,
                "subject_id": subj,
                "start_idx": i,
                "fs": fs,
                "channels": [f"ch_{j}" for j in range(23)],
                "dataset": "mhealth",
                "split": "all"
            })
    
    return pd.DataFrame(rows)




def load_mhealth_stream(root_dir: str, fs: int = 50, win_sec: float = 3.0, overlap: float = 0.5) -> pd.DataFrame:
    """
    Reads all mHealth_subject*.log files, builds windows at fs with ALL 23 channels.
    """
    root = Path(root_dir)
    files = sorted(list(root.glob("mHealth_subject*.log")))
    rows = []
    for p in files:
        subj = p.stem.split("subject")[-1]
        df = pd.read_csv(p, sep='\s+', header=None)
        
        # Use all 23 channels (columns 0-22, skip column 23 which is label)
        X = df.iloc[:, :23].to_numpy(np.float32)  # (N, 23)
        y = df.iloc[:, 23].astype(int).to_numpy()  # (N,) - column 24 (1-indexed) = column 23 (0-indexed)
        
        # Resample if needed
        approx_src = 50
        stride = max(1, int(round(approx_src / fs)))
        if stride > 1:
            X = X[::stride]
            y = y[::stride]
        
        # Create windows
        win_samples = int(win_sec * fs)
        step_samples = int(win_samples * (1 - overlap))
        
        for i in range(0, len(X) - win_samples + 1, step_samples):
            window_x = X[i:i + win_samples]  # (win_samples, 23)
            window_y = y[i:i + win_samples]
            
            # Use majority vote for label
            unique, counts = np.unique(window_y, return_counts=True)
            label = unique[np.argmax(counts)]
            
            rows.append({
                "x": window_x.T,  # (23, win_samples) - transpose to (C, T) format
                "y": label,
                "subject_id": subj,
                "start_idx": i,
                "fs": fs,
                "channels": [f"ch_{j}" for j in range(23)],
                "dataset": "mhealth",
                "split": "all"
            })
    
    return pd.DataFrame(rows)


class MHealthDataset(Dataset):
    """
    PyTorch Dataset class for MHEALTH data using preprocessed NPZ shards.
    
    This class wraps the NPZShardsDataset to provide a clean interface for
    loading MHEALTH data from preprocessed shard files.
    """
    
    def __init__(self, shards_glob: str, transform=None, split: str = "all"):
        """
        Initialize MHEALTH dataset.
        
        Args:
            shards_glob: Glob pattern for NPZ shard files (e.g., "data/mhealth/*.npz")
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
        Extract comprehensive dataset statistics for MHEALTH dataset.
        
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