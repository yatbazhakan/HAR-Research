from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# We use the raw "Inertial Signals": body_acc_* and body_gyro_* (50 Hz, 2.56 s windows -> 128 samples)
# Output windows in canonical order: [acc_x,acc_y,acc_z, gyro_x,gyro_y,gyro_z]  (shape: C=6, T=128)

def _load_block(root: Path, split: str, name: str) -> np.ndarray:
    p = root / split / "Inertial Signals" / f"{name}_{split}.txt"
    arr = np.loadtxt(p)
    # shape: (num_windows, 128)
    return arr

def _load_labels(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int).ravel()
    subj = np.loadtxt(root / split / f"subject_{split}.txt").astype(int).ravel()
    return y, subj

def load_ucihar_windows(root_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame where each row is one window:
      - x: np.ndarray float32 of shape (C=6, T=128)
      - y: int (activity id 1..6)
      - subject_id: int
      - dataset: "uci_har"
      - split: "train"/"test"
      - start_idx: always 0 (pre-windowed)
    """
    root = Path(root_dir)
    rows = []
    for split in ["train", "test"]:
        body_ax = _load_block(root, split, "body_acc_x")
        body_ay = _load_block(root, split, "body_acc_y")
        body_az = _load_block(root, split, "body_acc_z")
        gyro_x  = _load_block(root, split, "body_gyro_x")
        gyro_y  = _load_block(root, split, "body_gyro_y")
        gyro_z  = _load_block(root, split, "body_gyro_z")
        y, subj = _load_labels(root, split)

        n = body_ax.shape[0]
        assert all(a.shape[0] == n for a in [body_ay,body_az,gyro_x,gyro_y,gyro_z]) and y.shape[0]==n and subj.shape[0]==n

        for i in range(n):
            # stack into (C, T)
            x = np.stack([
                body_ax[i], body_ay[i], body_az[i],
                gyro_x[i],  gyro_y[i],  gyro_z[i],
            ], axis=0).astype(np.float32)  # (6,128)
            rows.append({
                "x": x,
                "y": int(y[i]),
                "subject_id": int(subj[i]),
                "dataset": "uci_har",
                "split": split,
                "start_idx": 0,
                "fs": 50,
                "channels": ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"],
            })
    return pd.DataFrame(rows)
