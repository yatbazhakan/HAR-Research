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

def load_mhealth_stream(root_dir: str, fs: int = 50, win_sec: float = 3.0, overlap: float = 0.5) -> pd.DataFrame:
    """
    Reads all mHealth_subject*.log files, builds windows at fs (assumes ~50Hz logs; if not, acts as downsampler by stride)
    """
    root = Path(root_dir)
    files = sorted(list(root.glob("mHealth_subject*.log")))
    rows = []
    for p in files:
        subj = p.stem.split("subject")[-1]
        df = _read_subject_file(p)
        # If original fs != target, we do a simple stride-based resample assuming roughly uniform sampling
        # (True timestamp isn't always provided consistently)
        # Adjust stride to keep ~fs rate relative to ~50Hz common sampling
        approx_src = 50
        stride = max(1, int(round(approx_src / fs)))
        if stride > 1:
            df = df.iloc[::stride].reset_index(drop=True)
        for rec in _windowize_df(df, fs=fs, win_sec=win_sec, overlap=overlap, subj_id=subj):
            rec.update({"dataset":"mhealth","split":"all"})
            rows.append(rec)
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
