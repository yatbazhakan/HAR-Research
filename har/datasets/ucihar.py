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

def load_ucihar_windows(root_dir: str) -> pd.DataFrame:
    """
    CRITICAL: Main function to load UCI-HAR dataset into standardized format
    
    This is the primary entry point for loading UCI-HAR data. It processes the raw
    sensor signals into a format suitable for training HAR models.
    
    Args:
        root_dir: Path to UCI-HAR dataset directory (should contain train/ and test/ folders)
    
    Returns:
        DataFrame where each row represents one 2.56-second activity window:
          - x: np.ndarray float32 of shape (C=6, T=128) - sensor data
            * Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            * Time steps: 128 samples (2.56 seconds at 50Hz)
          - y: int (activity id 1..6) - activity label
          - subject_id: int (1..30) - person who performed the activity
          - dataset: "uci_har" - dataset identifier
          - split: "train"/"test" - data split
          - start_idx: always 0 (pre-windowed by UCI-HAR creators)
    
    CRITICAL NOTES:
    - Data ordering [acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z] is FIXED - do not change
    - Window size of 128 samples (2.56s) is FIXED - models expect this exact size
    - Subject IDs are essential for LOSO cross-validation
    - Activity labels: 1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING
    """
    root = Path(root_dir)
    rows = []
    
    # CRITICAL: Process both train and test splits
    for split in ["train", "test"]:
        # Load all 6 sensor signals (3 accelerometer + 3 gyroscope)
        # CRITICAL: Order must match the expected channel ordering
        body_ax = _load_block(root, split, "body_acc_x")    # Accelerometer X
        body_ay = _load_block(root, split, "body_acc_y")    # Accelerometer Y  
        body_az = _load_block(root, split, "body_acc_z")    # Accelerometer Z
        gyro_x  = _load_block(root, split, "body_gyro_x")   # Gyroscope X
        gyro_y  = _load_block(root, split, "body_gyro_y")   # Gyroscope Y
        gyro_z  = _load_block(root, split, "body_gyro_z")   # Gyroscope Z
        
        # Load activity labels and subject IDs
        y, subj = _load_labels(root, split)

        # CRITICAL: Validate data consistency - all arrays must have same number of windows
        n = body_ax.shape[0]
        assert all(a.shape[0] == n for a in [body_ay,body_az,gyro_x,gyro_y,gyro_z]) and y.shape[0]==n and subj.shape[0]==n

        # CRITICAL: Process each window individually
        for i in range(n):
            # CRITICAL: Stack signals into (C=6, T=128) format
            # Order is FIXED: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            x = np.stack([
                body_ax[i], body_ay[i], body_az[i],  # Accelerometer channels
                gyro_x[i],  gyro_y[i],  gyro_z[i],   # Gyroscope channels
            ], axis=0).astype(np.float32)  # (6,128) - CRITICAL: float32 for memory efficiency
            
            # Create standardized row for DataFrame
            rows.append({
                "x": x,                                    # Sensor data (6, 128)
                "y": int(y[i]),                           # Activity label (1-6)
                "subject_id": int(subj[i]),               # Subject ID (1-30) - CRITICAL for LOSO
                "dataset": "uci_har",                     # Dataset identifier
                "split": split,                           # Train/test split
                "start_idx": 0,                           # Always 0 (pre-windowed)
                "fs": 50,                                 # Sampling frequency (Hz)
                "channels": ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"],  # Channel names
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
