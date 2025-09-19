"""
UCI-HAR Dataset Loader

This module handles loading and preprocessing of the UCI Human Activity Recognition (HAR) dataset.
The UCI-HAR dataset contains smartphone sensor data for 6 different human activities.

Key Features:
- Loads raw inertial signals (accelerometer and gyroscope data)
- Pre-processes data into fixed-length windows (128 samples = 2.56 seconds at 50Hz)
- Returns data in standardized format for training HAR models
- Supports both training and test splits

Data Format:
- Input: Raw sensor signals from smartphone accelerometer and gyroscope
- Output: 6-channel time series data (3 acc + 3 gyro) with 128 time steps
- Activities: 1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from .shards import NPZShardsDataset

# CRITICAL: Data preprocessing parameters
# We use the raw "Inertial Signals": body_acc_* and body_gyro_* (50 Hz, 2.56 s windows -> 128 samples)
# Output windows in canonical order: [acc_x,acc_y,acc_z, gyro_x,gyro_y,gyro_z]  (shape: C=6, T=128)
# This ordering is CRITICAL for model compatibility - changing it will break trained models

def _load_block(root: Path, split: str, name: str) -> np.ndarray:
    """
    CRITICAL: Load a single sensor signal block from UCI-HAR dataset
    
    Args:
        root: Dataset root directory
        split: "train" or "test" 
        name: Signal name (e.g., "body_acc_x", "body_gyro_y")
    
    Returns:
        Array of shape (num_windows, 128) - each row is one 2.56-second window
        
    Note: This function loads the raw inertial signals that are pre-windowed
    by the UCI-HAR dataset creators. Each window represents 2.56 seconds of data.
    """
    p = root / split / "Inertial Signals" / f"{name}_{split}.txt"
    arr = np.loadtxt(p)
    # shape: (num_windows, 128) - CRITICAL: 128 samples = 2.56 seconds at 50Hz
    return arr

def _load_labels(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    CRITICAL: Load activity labels and subject IDs for a dataset split
    
    Args:
        root: Dataset root directory
        split: "train" or "test"
    
    Returns:
        Tuple of (activity_labels, subject_ids)
        - activity_labels: 1-6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
        - subject_ids: 1-30 (identifies which person performed the activity)
    
    Note: Subject IDs are CRITICAL for Leave-One-Subject-Out (LOSO) cross-validation
    """
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int).ravel()
    subj = np.loadtxt(root / split / f"subject_{split}.txt").astype(int).ravel()
    return y, subj

def load_ucihar_raw_stream(root_dir: str, win_sec: float = 3.0, overlap: float = 0.5, fs: int = 50) -> pd.DataFrame:
    """
    Load UCI-HAR raw data and create windows for color coding experiments.
    
    This function loads the raw inertial signals and creates windows with all available
    sensor data for color coding visualization and analysis.
    
    Args:
        root_dir: Path to UCI-HAR dataset directory
        win_sec: Window size in seconds (default: 3.0)
        overlap: Overlap between windows (default: 0.5)
        fs: Target sampling frequency (default: 50)
    
    Returns:
        DataFrame with raw sensor data windows for color coding
    """
    root = Path(root_dir)
    rows = []
    
    # Calculate window parameters
    win_samples = int(win_sec * fs)
    step_samples = int(win_samples * (1 - overlap))
    
    for split in ["train", "test"]:
        # Load all sensor signals
        body_ax = _load_block(root, split, "body_acc_x")
        body_ay = _load_block(root, split, "body_acc_y") 
        body_az = _load_block(root, split, "body_acc_z")
        gyro_x = _load_block(root, split, "body_gyro_x")
        gyro_y = _load_block(root, split, "body_gyro_y")
        gyro_z = _load_block(root, split, "body_gyro_z")
        
        # Load total acceleration (for additional sensor data)
        total_ax = _load_block(root, split, "total_acc_x")
        total_ay = _load_block(root, split, "total_acc_y")
        total_az = _load_block(root, split, "total_acc_z")
        
        # Load labels and subjects
        y, subj = _load_labels(root, split)
        
        # Validate data consistency
        n = body_ax.shape[0]
        assert all(a.shape[0] == n for a in [body_ay, body_az, gyro_x, gyro_y, gyro_z, total_ax, total_ay, total_az])
        assert y.shape[0] == n and subj.shape[0] == n
        
        # Create windows from the pre-windowed data
        for i in range(n):
            # Stack all sensor data: body acc(3) + body gyro(3) + total acc(3) = 9 channels
            x = np.stack([
                body_ax[i], body_ay[i], body_az[i],  # Body accelerometer
                gyro_x[i],  gyro_y[i],  gyro_z[i],   # Body gyroscope  
                total_ax[i], total_ay[i], total_az[i] # Total accelerometer
            ], axis=0).astype(np.float32)  # (9, 128)
            
            # Resample to target window size if needed
            if win_samples != 128:
                # Simple linear interpolation to resize
                try:
                    from scipy import interpolate
                    x_resampled = np.zeros((9, win_samples), dtype=np.float32)
                    for ch in range(9):
                        f = interpolate.interp1d(np.linspace(0, 1, 128), x[ch], kind='linear')
                        x_resampled[ch] = f(np.linspace(0, 1, win_samples))
                    x = x_resampled
                except ImportError:
                    # Fallback: simple decimation/upsampling
                    if win_samples < 128:
                        # Decimate
                        step = 128 // win_samples
                        x = x[:, ::step]
                    else:
                        # Upsample by repetition
                        repeat_factor = win_samples // 128
                        x = np.repeat(x, repeat_factor, axis=1)
                        # Trim if needed
                        if x.shape[1] > win_samples:
                            x = x[:, :win_samples]
            
            rows.append({
                "x": x,                                    # Raw sensor data (9, win_samples)
                "y": int(y[i]),                           # Activity label (1-6)
                "subject_id": int(subj[i]),               # Subject ID (1-30)
                "dataset": "uci_har_raw",                 # Dataset identifier
                "split": split,                           # Train/test split
                "start_idx": 0,                           # Always 0 (pre-windowed)
                "fs": fs,                                 # Sampling frequency
                "channels": ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z"],
            })
    
    return pd.DataFrame(rows)


def load_ucihar_windows(root_dir: str) -> pd.DataFrame:
    """
    CRITICAL: Main function to load UCI-HAR dataset into standardized format
    
    This is the primary entry point for loading UCI-HAR data. It processes the raw
    sensor signals into a format suitable for training HAR models.
    
    Args:
        root_dir: Path to UCI-HAR dataset directory (should contain train/ and test/ folders)
    
    Returns:
        DataFrame where each row represents one 2.56-second activity window:
          - x: np.ndarray float32 of shape (C=9, T=128) - sensor data
            * Channels: [body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z]
            * Time steps: 128 samples (2.56 seconds at 50Hz)
          - y: int (activity id 1..6) - activity label
          - subject_id: int (1..30) - person who performed the activity
          - dataset: "uci_har" - dataset identifier
          - split: "train"/"test" - data split
          - start_idx: always 0 (pre-windowed by UCI-HAR creators)
    
    CRITICAL NOTES:
    - Data ordering includes all available sensor channels
    - Window size of 128 samples (2.56s) is FIXED - models expect this exact size
    - Subject IDs are essential for LOSO cross-validation
    - Activity labels: 1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING
    """
    root = Path(root_dir)
    rows = []
    
    # CRITICAL: Process both train and test splits
    for split in ["train", "test"]:
        # Load all 9 sensor signals (3 body acc + 3 body gyro + 3 total acc)
        body_ax = _load_block(root, split, "body_acc_x")    # Body Accelerometer X
        body_ay = _load_block(root, split, "body_acc_y")    # Body Accelerometer Y  
        body_az = _load_block(root, split, "body_acc_z")    # Body Accelerometer Z
        gyro_x  = _load_block(root, split, "body_gyro_x")   # Body Gyroscope X
        gyro_y  = _load_block(root, split, "body_gyro_y")   # Body Gyroscope Y
        gyro_z  = _load_block(root, split, "body_gyro_z")   # Body Gyroscope Z
        total_ax = _load_block(root, split, "total_acc_x")  # Total Accelerometer X
        total_ay = _load_block(root, split, "total_acc_y")  # Total Accelerometer Y
        total_az = _load_block(root, split, "total_acc_z")  # Total Accelerometer Z
        
        # Load activity labels and subject IDs
        y, subj = _load_labels(root, split)

        # CRITICAL: Validate data consistency - all arrays must have same number of windows
        n = body_ax.shape[0]
        assert all(a.shape[0] == n for a in [body_ay,body_az,gyro_x,gyro_y,gyro_z,total_ax,total_ay,total_az]) and y.shape[0]==n and subj.shape[0]==n

        # CRITICAL: Process each window individually
        for i in range(n):
            # CRITICAL: Stack signals into (C=9, T=128) format
            # Order: [body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z]
            x = np.stack([
                body_ax[i], body_ay[i], body_az[i],  # Body accelerometer channels
                gyro_x[i],  gyro_y[i],  gyro_z[i],   # Body gyroscope channels
                total_ax[i], total_ay[i], total_az[i] # Total accelerometer channels
            ], axis=0).astype(np.float32)  # (9,128) - CRITICAL: float32 for memory efficiency
            
            # Create standardized row for DataFrame
            rows.append({
                "x": x,                                    # Sensor data (9, 128)
                "y": int(y[i]),                           # Activity label (1-6)
                "subject_id": int(subj[i]),               # Subject ID (1-30) - CRITICAL for LOSO
                "dataset": "uci_har",                     # Dataset identifier
                "split": split,                           # Train/test split
                "start_idx": 0,                           # Always 0 (pre-windowed)
                "fs": 50,                                 # Sampling frequency (Hz)
                "channels": ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z"],  # Channel names
            })
    return pd.DataFrame(rows)


class UCIHARDataset(Dataset):
    """
    PyTorch Dataset class for UCI-HAR data using preprocessed NPZ shards.
    
    This class wraps the NPZShardsDataset to provide a clean interface for
    loading UCI-HAR data from preprocessed shard files.
    """
    
    def __init__(self, shards_glob: str, transform=None, split: str = "all"):
        """
        Initialize UCI-HAR dataset.
        
        Args:
            shards_glob: Glob pattern for NPZ shard files (e.g., "data/uci_har/*.npz")
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
        Extract comprehensive dataset statistics for UCI-HAR dataset.
        
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