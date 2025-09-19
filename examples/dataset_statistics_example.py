#!/usr/bin/env python3
"""
Example script demonstrating how to use the new dataset statistics functions.

This script shows how to extract comprehensive statistics from each dataset class,
including information about subjects, activities, and their distributions.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from har.datasets.mhealth import MHealthDataset
from har.datasets.pamap2 import PAMAP2Dataset
from har.datasets.ucihar import UCIHARDataset
from har.datasets.shards import NPZShardsDataset


def print_dataset_statistics(dataset, dataset_name):
    """
    Print comprehensive statistics for a dataset.
    
    Args:
        dataset: Dataset instance with get_dataset_statistics method
        dataset_name: Name of the dataset for display
    """
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    try:
        stats = dataset.get_dataset_statistics()
        
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Number of Subjects: {stats['num_subjects']}")
        print(f"Number of Activities: {stats['num_activities']}")
        
        print(f"\nUnique Subjects: {stats['unique_subjects']}")
        print(f"Unique Activities: {stats['unique_activities']}")
        
        print(f"\nSamples per Subject:")
        for subject, count in stats['samples_per_subject'].items():
            print(f"  Subject {subject}: {count:,} samples")
        
        print(f"\nSamples per Activity:")
        for activity, count in stats['samples_per_activity'].items():
            print(f"  Activity {activity}: {count:,} samples")
        
        print(f"\nActivities per Subject:")
        for subject, activities in stats['activities_per_subject'].items():
            print(f"  Subject {subject}: {activities}")
        
        print(f"\nSubject-Activity Distribution Matrix:")
        for subject, activity_counts in stats['subject_activity_matrix'].items():
            print(f"  Subject {subject}:")
            for activity, count in activity_counts.items():
                print(f"    Activity {activity}: {count:,} samples")
                
    except Exception as e:
        print(f"Error computing statistics for {dataset_name}: {e}")


def main():
    """
    Main function demonstrating dataset statistics extraction.
    """
    print("Dataset Statistics Extraction Example")
    print("====================================")
    
    # Example usage with different dataset types
    # Note: These examples assume you have preprocessed shard files available
    
    # Example 1: MHEALTH Dataset
    try:
        mhealth_dataset = MHealthDataset(
            shards_glob="data/MHEALTH/*.npz",  # Adjust path as needed
            split="all"
        )
        print_dataset_statistics(mhealth_dataset, "MHEALTH")
    except Exception as e:
        print(f"MHEALTH dataset not available: {e}")
    
    # Example 2: PAMAP2 Dataset
    try:
        pamap2_dataset = PAMAP2Dataset(
            shards_glob="data/PAMAP2/*.npz",  # Adjust path as needed
            split="all"
        )
        print_dataset_statistics(pamap2_dataset, "PAMAP2")
    except Exception as e:
        print(f"PAMAP2 dataset not available: {e}")
    
    # Example 3: UCI-HAR Dataset
    try:
        ucihar_dataset = UCIHARDataset(
            shards_glob="data/UCI-HAR/*.npz",  # Adjust path as needed
            split="all"
        )
        print_dataset_statistics(ucihar_dataset, "UCI-HAR")
    except Exception as e:
        print(f"UCI-HAR dataset not available: {e}")
    
    # Example 4: Generic NPZ Shards Dataset
    try:
        shards_dataset = NPZShardsDataset(
            shards_glob="data/*/*.npz",  # Adjust path as needed
            split="all"
        )
        print_dataset_statistics(shards_dataset, "Generic NPZ Shards")
    except Exception as e:
        print(f"Generic shards dataset not available: {e}")


if __name__ == "__main__":
    main()
