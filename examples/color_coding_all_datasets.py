#!/usr/bin/env python3
"""
Color Coding Transform Example for All Datasets

This script demonstrates color coding across all available HAR datasets:
- UCI-HAR: 9 sensor channels (body acc + body gyro + total acc)
- PAMAP2: 51 sensor channels (all available sensors)
- MHEALTH: 23 sensor channels (all available sensors)

The script loads raw shards and creates color-coded visualizations for each dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from har.transforms.color_coding import compute_subject_minmax, ColorCodingTransform


def load_shards(shards_glob: str, max_samples: int = 1000, use_normalized: bool = False):
    """Load shards and return a sample of data.
    
    Args:
        shards_glob: Glob pattern for shard files
        max_samples: Maximum number of samples to load
        use_normalized: If True, expects normalized shards; if False, expects raw shards
    """
    files = sorted(glob.glob(shards_glob))
    if not files:
        return None, None
    
    print(f"   Found {len(files)} shard files")
    print(f"   Using {'normalized' if use_normalized else 'raw'} shards")
    
    all_data = []
    for file_path in files[:5]:  # Load first 5 shards
        data = np.load(file_path, allow_pickle=False)
        n_samples = data["X"].shape[0]
        
        for i in range(min(n_samples, max_samples // len(files))):
            all_data.append({
                "x": data["X"][i],
                "y": int(data["y"][i]),
                "subject_id": data["subject_id"][i] if "subject_id" in data.files else f"unknown_{i}",
                "dataset": data["dataset"][i] if "dataset" in data.files else "unknown",
                "is_normalized": use_normalized
            })
    
    return pd.DataFrame(all_data), files[0]


def compute_color_coding_normalization(df: pd.DataFrame, subject_id: str) -> dict:
    """Compute normalization for color coding.
    
    For raw data: compute min-max per sensor
    For normalized data: use fixed range [-3, 3] (typical for z-score normalized data)
    """
    subject_data = df[df["subject_id"] == subject_id]
    
    if subject_data.empty:
        return {}
    
    # Check if data is normalized
    is_normalized = subject_data.iloc[0].get("is_normalized", False)
    
    if is_normalized:
        # For normalized data, use fixed range assuming z-score normalization
        print(f"   Using fixed range [-3, 3] for normalized data")
        return {"normalized_range": (-3.0, 3.0)}
    else:
        # For raw data, compute actual min-max
        print(f"   Computing min-max for raw data")
        all_data = np.stack(subject_data["x"].tolist(), axis=0)  # (N, C, T)
        
        # Compute min-max for each channel
        min_vals = np.min(all_data, axis=(0, 2))  # (C,)
        max_vals = np.max(all_data, axis=(0, 2))  # (C,)
        
        return {"min_vals": min_vals, "max_vals": max_vals}

def create_sensor_mapping(dataset_name: str, data_shape: tuple):
    """Create sensor mapping for different datasets."""
    channels, time_steps = data_shape
    
    if dataset_name == "uci_har":
        # UCI-HAR: 9 channels (body acc + body gyro + total acc)
        sensor_names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z", 
            "total_acc_x", "total_acc_y", "total_acc_z"
        ]
        # Group into 3 sensors of 3 channels each
        sensors_order = ["body_acc", "body_gyro", "total_acc"]
        sensor_groups = {
            "body_acc": [0, 1, 2],
            "body_gyro": [3, 4, 5], 
            "total_acc": [6, 7, 8]
        }
        
    elif dataset_name == "pamap2":
        # PAMAP2: 51 channels - group into logical sensors
        sensor_names = [f"sensor_{i}" for i in range(channels)]
        # Group into 17 sensors of 3 channels each (51/3 = 17)
        sensors_order = [f"imu_{i//3}" for i in range(0, channels, 3)]
        sensor_groups = {}
        for i, sensor in enumerate(sensors_order):
            start_idx = i * 3
            sensor_groups[sensor] = list(range(start_idx, min(start_idx + 3, channels)))
            
    elif dataset_name == "mhealth":
        # MHEALTH: 23 channels - group into logical sensors
        sensor_names = [f"ch_{i}" for i in range(channels)]
        # Group into 7 sensors based on MHEALTH structure
        sensors_order = ["acc_chest", "ecg", "acc_ankle", "gyro_ankle", "mag_ankle", "acc_arm", "gyro_arm"]
        sensor_groups = {
            "acc_chest": [0, 1, 2],      # Columns 1-3
            "ecg": [3, 4],               # Columns 4-5
            "acc_ankle": [5, 6, 7],      # Columns 6-8
            "gyro_ankle": [8, 9, 10],    # Columns 9-11
            "mag_ankle": [11, 12, 13],   # Columns 12-14
            "acc_arm": [14, 15, 16],     # Columns 15-17
            "gyro_arm": [17, 18, 19]     # Columns 18-20
        }
        
    else:
        # Default: treat each channel as a separate sensor
        sensor_names = [f"sensor_{i}" for i in range(channels)]
        sensors_order = [f"sensor_{i}" for i in range(0, channels, 3)]  # Group by 3s
        sensor_groups = {}
        for i, sensor in enumerate(sensors_order):
            start_idx = i * 3
            sensor_groups[sensor] = list(range(start_idx, min(start_idx + 3, channels)))
    
    return sensors_order, sensor_groups, sensor_names


def extract_sensor_data_from_window(window_data, sensor_groups, sensors_order):
    """Extract sensor data from a window using the sensor grouping."""
    sensor_data = {}
    
    for sensor_name in sensors_order:
        if sensor_name in sensor_groups:
            channel_indices = sensor_groups[sensor_name]
            # Extract data for this sensor's channels
            sensor_channels = window_data[channel_indices, :]  # (n_channels, time_steps)
            sensor_data[sensor_name] = sensor_channels.T  # (time_steps, n_channels)
        else:
            # Create dummy data if sensor not found
            sensor_data[sensor_name] = np.zeros((window_data.shape[1], 3), dtype=np.float32)
    
    return sensor_data


def visualize_color_coded_images(images_by_activity, activity_labels, dataset_name, save_path=None):
    """Visualize the color-coded images for different activities."""
    num_activities = len(images_by_activity)
    if num_activities == 0:
        print("   No activities to visualize")
        return
        
    cols = 3
    rows = (num_activities + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (activity_id, image_data) in enumerate(images_by_activity.items()):
        row = idx // cols
        col = idx % cols
        
        ax = axes[row, col]
        
        # Display the image
        ax.imshow(image_data, aspect='auto')
        ax.set_title(f"Activity {activity_id}: {activity_labels.get(activity_id, 'Unknown')}", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Sensor Bands")
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_activities, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"Color-Coded Visualization: {dataset_name.upper()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    
    plt.show()


def process_dataset(dataset_name: str, shards_glob: str, max_samples: int = 1000, use_normalized: bool = False):
    """Process a single dataset for color coding.
    
    Args:
        dataset_name: Name of the dataset
        shards_glob: Glob pattern for shard files
        max_samples: Maximum number of samples to load
        use_normalized: Whether to use normalized shards
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {dataset_name.upper()} ({'normalized' if use_normalized else 'raw'})")
    print(f"{'='*60}")
    
    # Load shards
    print(f"Loading {'normalized' if use_normalized else 'raw'} shards from: {shards_glob}")
    df, sample_file = load_shards(shards_glob, max_samples, use_normalized)
    
    if df is None or len(df) == 0:
        print(f"   No data found for {dataset_name}")
        return
    
    print(f"   Loaded {len(df)} samples")
    print(f"   Data shape: {df.iloc[0]['x'].shape}")
    
    # Create sensor mapping
    data_shape = df.iloc[0]['x'].shape
    sensors_order, sensor_groups, sensor_names = create_sensor_mapping(dataset_name, data_shape)
    
    print(f"   Sensor groups: {list(sensor_groups.keys())}")
    print(f"   Sensors order: {sensors_order}")
    
    # Get unique activities
    unique_activities = sorted(df['y'].unique())
    print(f"   Activities: {unique_activities}")
    
    # Activity labels
    activity_labels = {i: f"Activity {i}" for i in unique_activities}
    
    # Collect one window per activity
    windows_by_activity = {}
    for activity_id in unique_activities:
        activity_data = df[df['y'] == activity_id]
        if len(activity_data) > 0:
            window = activity_data.iloc[0]
            windows_by_activity[activity_id] = {
                'data': window['x'],
                'activity_name': activity_labels[activity_id],
                'subject_id': window['subject_id']
            }
    
    print(f"   Found windows for {len(windows_by_activity)} activities")
    
    # Process each activity
    all_sensor_data = {}
    for activity_id, window_info in windows_by_activity.items():
        sensor_data = extract_sensor_data_from_window(window_info['data'], sensor_groups, sensors_order)
        
        # Store for min-max computation
        for sensor, data in sensor_data.items():
            if sensor not in all_sensor_data:
                all_sensor_data[sensor] = []
            all_sensor_data[sensor].append(data)
    
    # Compute subject min-max values based on data type
    print("Computing min-max values...")
    if use_normalized:
        # For normalized data, use fixed range assuming z-score normalization
        print("   Using fixed range [-3, 3] for normalized data")
        subject_minmax = {}
        for sensor in sensors_order:
            subject_minmax[sensor] = (
                np.array([-3.0, -3.0, -3.0]),
                np.array([3.0, 3.0, 3.0])
            )
    else:
        # For raw data, compute actual min-max
        print("   Computing actual min-max for raw data")
        flattened_data = {}
        for sensor in sensors_order:
            if sensor in all_sensor_data:
                sensor_windows = np.concatenate(all_sensor_data[sensor], axis=0)
                flattened_data[sensor] = sensor_windows
        
        subject_minmax = compute_subject_minmax(flattened_data)
    
    # Create color coding transform
    time_steps = df.iloc[0]['x'].shape[1]
    transform = ColorCodingTransform(
        sensors_order=sensors_order,
        time_steps_per_window=time_steps,
        sensor_band_height_px=10,
        output_format='HWC'
    )
    
    print(f"   Expected output shape: ({10 * len(sensors_order)}, {time_steps}, 3)")
    
    # Apply transform to each activity
    print("Applying color coding transform...")
    images_by_activity = {}
    
    for activity_id, window_info in windows_by_activity.items():
        print(f"   Processing Activity {activity_id}: {window_info['activity_name']}")
        
        sensor_data = extract_sensor_data_from_window(window_info['data'], sensor_groups, sensors_order)
        
        try:
            color_image = transform(sensor_data, subject_minmax)
            images_by_activity[activity_id] = color_image
            print(f"      Output shape: {color_image.shape}")
        except Exception as e:
            print(f"      Error: {e}")
    
    # Visualize results
    if images_by_activity:
        print("Creating visualization...")
        suffix = "_normalized" if use_normalized else "_raw"
        save_path = f"{dataset_name}_color_coding{suffix}.png"
        dataset_title = f"{dataset_name.upper()} ({'Normalized' if use_normalized else 'Raw'})"
        visualize_color_coded_images(images_by_activity, activity_labels, dataset_title, save_path)
    else:
        print("   No images to visualize")


def main():
    """Main function to process all datasets."""
    print("Color Coding Transform Example for All Datasets")
    print("=" * 60)
    
    # Define dataset configurations (try raw first, then normalized)
    datasets = {
        "uci_har": ["artifacts/preprocessed/uci_har_raw/*.npz", "artifacts/preprocessed/uci_har/*.npz"],
        "pamap2": ["artifacts/preprocessed/pamap2_raw/*.npz", "artifacts/preprocessed/pamap2/*.npz"], 
        "mhealth": ["artifacts/preprocessed/mhealth_raw/*.npz", "artifacts/preprocessed/mhealth/*.npz"]
    }
    
    for dataset_name, shard_paths in datasets.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING {dataset_name.upper()}")
        print("="*60)
        
        # Try raw shards first, then normalized
        processed = False
        for i, shards_glob in enumerate(shard_paths):
            if Path(shards_glob.replace("/*.npz", "")).exists():
                use_normalized = i == 1  # Second path is normalized
                data_type = "normalized" if use_normalized else "raw"
                print(f"Using {data_type} shards for {dataset_name}")
                process_dataset(dataset_name, shards_glob, max_samples=500, use_normalized=use_normalized)
                processed = True
                break
        
        if not processed:
            print(f"No shards found for {dataset_name}")
            print("Run preprocessing first:")
            print("python scripts/preprocess.py --uci_har_root data/UCI-HAR --pamap2_root data/PAMAP2/PAMAP2_Dataset --mhealth_root data/MHEALTH --outdir artifacts/preprocessed --fs 50 --win_sec 3.0 --overlap 0.5")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
