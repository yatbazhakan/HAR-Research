#!/usr/bin/env python3
"""
Dataset Information Printer

This script loads and prints comprehensive information about all available datasets.
It works with both raw dataset files and preprocessed NPZ shards.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from har.datasets.mhealth import load_mhealth_stream, MHealthDataset
from har.datasets.pamap2 import load_pamap2_stream, PAMAP2Dataset
from har.datasets.ucihar import load_ucihar_windows, UCIHARDataset


def print_raw_dataset_info():
    """Print information from raw dataset files."""
    print("="*80)
    print("RAW DATASET INFORMATION")
    print("="*80)
    
    # MHEALTH Dataset
    print("\n" + "="*60)
    print("MHEALTH DATASET (Raw Data)")
    print("="*60)
    try:
        mhealth_data = load_mhealth_stream("data/MHEALTH", fs=50, win_sec=3.0, overlap=0.5)
        print(f"Total samples: {len(mhealth_data):,}")
        print(f"Number of subjects: {mhealth_data['subject_id'].nunique()}")
        print(f"Number of activities: {mhealth_data['y'].nunique()}")
        print(f"Unique subjects: {sorted(mhealth_data['subject_id'].unique().tolist())}")
        print(f"Unique activities: {sorted(mhealth_data['y'].unique().tolist())}")
        
        print(f"\nSamples per subject:")
        subject_counts = mhealth_data['subject_id'].value_counts().sort_index()
        for subject, count in subject_counts.items():
            print(f"  Subject {subject}: {count:,} samples")
        
        print(f"\nSamples per activity:")
        activity_counts = mhealth_data['y'].value_counts().sort_index()
        for activity, count in activity_counts.items():
            print(f"  Activity {activity}: {count:,} samples")
        
        print(f"\nActivities per subject:")
        for subject in sorted(mhealth_data['subject_id'].unique()):
            subject_data = mhealth_data[mhealth_data['subject_id'] == subject]
            activities = sorted(subject_data['y'].unique().tolist())
            print(f"  Subject {subject}: {activities}")
            
    except Exception as e:
        print(f"Error loading MHEALTH dataset: {e}")
    
    # PAMAP2 Dataset
    print("\n" + "="*60)
    print("PAMAP2 DATASET (Raw Data)")
    print("="*60)
    try:
        pamap2_data = load_pamap2_stream("data/PAMAP2/PAMAP2_Dataset", win_sec=3.0, overlap=0.5, fs=50)
        print(f"Total samples: {len(pamap2_data):,}")
        print(f"Number of subjects: {pamap2_data['subject_id'].nunique()}")
        print(f"Number of activities: {pamap2_data['y'].nunique()}")
        print(f"Unique subjects: {sorted(pamap2_data['subject_id'].unique().tolist())}")
        print(f"Unique activities: {sorted(pamap2_data['y'].unique().tolist())}")
        
        print(f"\nSamples per subject:")
        subject_counts = pamap2_data['subject_id'].value_counts().sort_index()
        for subject, count in subject_counts.items():
            print(f"  Subject {subject}: {count:,} samples")
        
        print(f"\nSamples per activity:")
        activity_counts = pamap2_data['y'].value_counts().sort_index()
        for activity, count in activity_counts.items():
            print(f"  Activity {activity}: {count:,} samples")
        
        print(f"\nActivities per subject:")
        for subject in sorted(pamap2_data['subject_id'].unique()):
            subject_data = pamap2_data[pamap2_data['subject_id'] == subject]
            activities = sorted(subject_data['y'].unique().tolist())
            print(f"  Subject {subject}: {activities}")
            
    except Exception as e:
        print(f"Error loading PAMAP2 dataset: {e}")
    
    # UCI-HAR Dataset
    print("\n" + "="*60)
    print("UCI-HAR DATASET (Raw Data)")
    print("="*60)
    try:
        ucihar_data = load_ucihar_windows("data/UCI-HAR")
        print(f"Total samples: {len(ucihar_data):,}")
        print(f"Number of subjects: {ucihar_data['subject_id'].nunique()}")
        print(f"Number of activities: {ucihar_data['y'].nunique()}")
        print(f"Unique subjects: {sorted(ucihar_data['subject_id'].unique().tolist())}")
        print(f"Unique activities: {sorted(ucihar_data['y'].unique().tolist())}")
        
        print(f"\nSamples per subject:")
        subject_counts = ucihar_data['subject_id'].value_counts().sort_index()
        for subject, count in subject_counts.items():
            print(f"  Subject {subject}: {count:,} samples")
        
        print(f"\nSamples per activity:")
        activity_counts = ucihar_data['y'].value_counts().sort_index()
        for activity, count in activity_counts.items():
            print(f"  Activity {activity}: {count:,} samples")
        
        print(f"\nActivities per subject:")
        for subject in sorted(ucihar_data['subject_id'].unique()):
            subject_data = ucihar_data[ucihar_data['subject_id'] == subject]
            activities = sorted(subject_data['y'].unique().tolist())
            print(f"  Subject {subject}: {activities}")
            
    except Exception as e:
        print(f"Error loading UCI-HAR dataset: {e}")


def print_shard_dataset_info():
    """Print information from preprocessed NPZ shard files."""
    print("\n" + "="*80)
    print("PREPROCESSED SHARD DATASET INFORMATION")
    print("="*80)
    
    # Check for shard files
    shard_patterns = [
        "data/MHEALTH/*.npz",
        "data/PAMAP2/*.npz", 
        "data/UCI-HAR/*.npz",
        "artifacts/preprocessed/*.npz"
    ]
    
    shard_found = False
    for pattern in shard_patterns:
        import glob
        files = glob.glob(pattern)
        if files:
            shard_found = True
            break
    
    if not shard_found:
        print("No preprocessed NPZ shard files found.")
        print("To create shards, run the preprocessing scripts first.")
        return
    
    # MHEALTH Shards
    try:
        mhealth_dataset = MHealthDataset("data/MHEALTH/*.npz", split="all")
        print(f"\nMHEALTH Shards: {len(mhealth_dataset)} samples")
        stats = mhealth_dataset.get_dataset_statistics()
        print(f"Subjects: {stats['num_subjects']}, Activities: {stats['num_activities']}")
    except Exception as e:
        print(f"MHEALTH shards not available: {e}")
    
    # PAMAP2 Shards
    try:
        pamap2_dataset = PAMAP2Dataset("data/PAMAP2/*.npz", split="all")
        print(f"\nPAMAP2 Shards: {len(pamap2_dataset)} samples")
        stats = pamap2_dataset.get_dataset_statistics()
        print(f"Subjects: {stats['num_subjects']}, Activities: {stats['num_activities']}")
    except Exception as e:
        print(f"PAMAP2 shards not available: {e}")
    
    # UCI-HAR Shards
    try:
        ucihar_dataset = UCIHARDataset("data/UCI-HAR/*.npz", split="all")
        print(f"\nUCI-HAR Shards: {len(ucihar_dataset)} samples")
        stats = ucihar_dataset.get_dataset_statistics()
        print(f"Subjects: {stats['num_subjects']}, Activities: {stats['num_activities']}")
    except Exception as e:
        print(f"UCI-HAR shards not available: {e}")


def print_dataset_summary():
    """Print a summary of all available datasets."""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    datasets = {
        "MHEALTH": "data/MHEALTH",
        "PAMAP2": "data/PAMAP2/PAMAP2_Dataset", 
        "UCI-HAR": "data/UCI-HAR"
    }
    
    for name, path in datasets.items():
        if Path(path).exists():
            print(f"✓ {name}: Available at {path}")
        else:
            print(f"✗ {name}: Not found at {path}")


def main():
    """Main function to print all dataset information."""
    print("HAR Dataset Information Printer")
    print("==============================")
    
    # Print dataset summary
    print_dataset_summary()
    
    # Print raw dataset information
    print_raw_dataset_info()
    
    # Print shard dataset information
    print_shard_dataset_info()
    
    print("\n" + "="*80)
    print("END OF DATASET INFORMATION")
    print("="*80)


if __name__ == "__main__":
    main()
