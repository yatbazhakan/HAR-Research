#!/usr/bin/env python3
"""
Color Coding Transform Example with MHEALTH Data

This script loads real MHEALTH data from subject 1, collects one window from each activity,
and demonstrates how to use the ColorCodingTransform to convert multi-sensor time series 
data into color-coded RGB images with visualization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from har.transforms.color_coding import compute_subject_minmax, ColorCodingTransform
from har.datasets.mhealth import load_mhealth_raw_stream
from har.datasets.ucihar import load_ucihar_raw_stream
from har.datasets.pamap2 import load_pamap2_raw_stream


def load_mhealth_subject_data(subject_id=1, data_dir="data/MHEALTH"):
    """Load MHEALTH data for a specific subject and extract one window per activity."""
    print(f"Loading MHEALTH data for subject {subject_id}...")
    
    # Load the raw data
    mhealth_data = load_mhealth_raw_stream(data_dir, fs=50, win_sec=3.0, overlap=0.5)
    
    # Filter for the specific subject
    subject_data = mhealth_data[mhealth_data['subject_id'] == str(subject_id)].copy()
    
    if len(subject_data) == 0:
        raise ValueError(f"No data found for subject {subject_id}")
    
    print(f"   Found {len(subject_data)} windows for subject {subject_id}")
    print(f"   Activities present: {sorted(subject_data['y'].unique())}")
    
    # Activity labels mapping (from MHEALTH README)
    activity_labels = {
        0: "Null class",
        1: "Standing still",
        2: "Sitting and relaxing", 
        3: "Lying down",
        4: "Walking",
        5: "Climbing stairs",
        6: "Waist bends forward",
        7: "Frontal elevation of arms",
        8: "Knees bending (crouching)",
        9: "Cycling",
        10: "Jogging",
        11: "Running",
        12: "Jump front & back"
    }
    
    # Collect one window per activity
    windows_by_activity = {}
    for activity_id in sorted(subject_data['y'].unique()):
        if activity_id == 0:  # Skip null class
            continue
            
        activity_windows = subject_data[subject_data['y'] == activity_id]
        if len(activity_windows) > 0:
            # Take the first window for this activity
            window = activity_windows.iloc[0]
            windows_by_activity[activity_id] = {
                'data': window['x'],
                'label': activity_id,
                'activity_name': activity_labels.get(activity_id, f"Activity {activity_id}"),
                'subject_id': window['subject_id']
            }
            print(f"   Activity {activity_id} ({activity_labels.get(activity_id, 'Unknown')}): {len(activity_windows)} windows")
    
    return windows_by_activity, activity_labels


def extract_sensor_data_from_window(window_data, sensors_order):
    """Extract sensor data from a window using the MHEALTH channel mapping."""
    # window_data shape: (C, T) where C=channels, T=time_steps
    # We need to convert to (T, 3) for each sensor
    
    # MHEALTH has 23 channels, we'll map them to 7 sensors of 3 channels each
    # For simplicity, we'll group consecutive channels into sensors
    
    sensor_data = {}
    channels, time_steps = window_data.shape
    
    # Group channels into sensors (3 channels per sensor)
    for i, sensor_name in enumerate(sensors_order):
        start_ch = i * 3
        end_ch = min(start_ch + 3, channels)
        
        if end_ch - start_ch == 3:
            # Extract 3 channels and transpose to (T, 3)
            sensor_channels = window_data[start_ch:end_ch, :]  # (3, T)
            sensor_data[sensor_name] = sensor_channels.T  # (T, 3)
        else:
            # If not enough channels, pad with zeros
            available_ch = end_ch - start_ch
            sensor_channels = np.zeros((3, time_steps))
            sensor_channels[:available_ch, :] = window_data[start_ch:end_ch, :]
            sensor_data[sensor_name] = sensor_channels.T  # (T, 3)
    
    return sensor_data


def visualize_color_coded_images(images_by_activity, activity_labels, save_path=None):
    """Visualize the color-coded images for different activities."""
    num_activities = len(images_by_activity)
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
        activity_name = activity_labels.get(activity_id, f'Activity {activity_id}')
        ax.set_title(f"Activity {activity_id}: {activity_name}", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Sensor Bands")
        
        # Add sensor labels on the y-axis
        sensor_names = ['acc_chest', 'gyro_chest', 'mag_chest', 'acc_left_ankle', 
                       'gyro_left_ankle', 'acc_right_arm', 'gyro_right_arm']
        ax.set_yticks(range(5, image_data.shape[0], 10))  # Every 10 pixels
        ax.set_yticklabels(sensor_names, fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_activities, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    
    plt.show()


def load_from_shards(shard_path: str, use_normalized: bool = False, max_samples: int = 10):
    """Load data from preprocessed shards (raw or normalized)."""
    import glob
    
    shard_files = sorted(glob.glob(shard_path))
    if not shard_files:
        return None, None
    
    print(f"Loading {'normalized' if use_normalized else 'raw'} shards from: {shard_files[0]}")
    
    # MHEALTH activity labels mapping
    activity_labels_map = {
        0: "Null class",
        1: "Standing still",
        2: "Sitting and relaxing", 
        3: "Lying down",
        4: "Walking",
        5: "Climbing stairs",
        6: "Waist bends forward",
        7: "Frontal elevation of arms",
        8: "Knees bending (crouching)",
        9: "Cycling",
        10: "Jogging",
        11: "Running",
        12: "Jump front & back"
    }
    
    windows_by_activity = {}
    activity_labels = {}
    
    # Try to load from multiple shards to get all activities
    for shard_file in shard_files[:3]:  # Try first 3 shards
        print(f"   Loading from: {shard_file}")
        data = np.load(shard_file, allow_pickle=False)
        n_samples = min(data["X"].shape[0], max_samples)
        
        print(f"     Shard contains {data['X'].shape[0]} samples, using {n_samples}")
        print(f"     Activities in this shard: {sorted(np.unique(data['y']))}")
        
        # Get one sample per activity from this shard
        for i in range(n_samples):
            activity = int(data["y"][i])
            # Skip null class (activity 0) and only take one sample per activity
            if activity != 0 and activity not in windows_by_activity:
                windows_by_activity[activity] = {
                    'data': data["X"][i],  # (C, T)
                    'activity_name': activity_labels_map.get(activity, f"Activity {activity}"),
                    'subject_id': data["subject_id"][i] if "subject_id" in data.files else f"unknown_{i}",
                    'is_normalized': use_normalized
                }
                activity_labels[activity] = activity_labels_map.get(activity, f"Activity {activity}")
        
        # If we have enough activities, break
        if len(windows_by_activity) >= 8:  # MHEALTH has 12 activities (1-12)
            break
    
    print(f"   Found {len(windows_by_activity)} activities: {sorted(windows_by_activity.keys())}")
    for activity_id in sorted(windows_by_activity.keys()):
        print(f"     Activity {activity_id}: {activity_labels[activity_id]}")
    
    return windows_by_activity, activity_labels


def main():
    """Demonstrate color coding transform with real MHEALTH data."""
    print("Color Coding Transform Example with MHEALTH Data")
    print("================================================")
    
    try:
        # Try to load from shards first
        shard_paths = [
            "artifacts/preprocessed/mhealth_raw/*.npz",
            "artifacts/preprocessed/mhealth/*.npz"
        ]
        
        windows_by_activity = None
        activity_labels = None
        use_normalized = False
        
        for shard_path in shard_paths:
            if Path(shard_path.replace("/*.npz", "")).exists():
                use_normalized = "mhealth_raw" not in shard_path
                windows_by_activity, activity_labels = load_from_shards(shard_path, use_normalized, max_samples=100)
                break
        
        if windows_by_activity is None:
            print("No shards found, using raw data loading...")
            # Load MHEALTH data for subject 1
            print("\n1. Loading MHEALTH data for subject 1...")
            windows_by_activity, activity_labels = load_mhealth_subject_data(subject_id=1)
        
        if not windows_by_activity:
            print("   No activity data found!")
            return
        
        # Define sensor order for color coding
        sensors_order = ['acc_chest', 'gyro_chest', 'mag_chest', 'acc_left_ankle', 
                        'gyro_left_ankle', 'acc_right_arm', 'gyro_right_arm']
        
        print(f"\n2. Processing {len(windows_by_activity)} activities...")
        
        # Process each activity
        images_by_activity = {}
        all_sensor_data = {}
        
        for activity_id, window_info in windows_by_activity.items():
            print(f"   Processing Activity {activity_id}: {window_info['activity_name']}")
            
            # Extract sensor data from the window
            sensor_data = extract_sensor_data_from_window(window_info['data'], sensors_order)
            
            # Store for min-max computation
            for sensor, data in sensor_data.items():
                if sensor not in all_sensor_data:
                    all_sensor_data[sensor] = []
                all_sensor_data[sensor].append(data)
        
        # Compute subject min-max values across all activities
        print("\n3. Computing subject min-max values...")
        
        # Check if data is normalized
        is_normalized = windows_by_activity[list(windows_by_activity.keys())[0]].get('is_normalized', False)
        
        if is_normalized:
            # For normalized data, use fixed range
            print("   Using fixed range [-3, 3] for normalized data")
            subject_minmax = {}
            for sensor in sensors_order:
                subject_minmax[sensor] = (
                    np.array([-3.0, -3.0, -3.0]),
                    np.array([3.0, 3.0, 3.0])
                )
        else:
            # For raw data, compute actual min-max
            print("   Computing min-max for raw data")
            flattened_data = {}
            for sensor in sensors_order:
                if sensor in all_sensor_data:
                    # Concatenate all windows for this sensor
                    sensor_windows = np.concatenate(all_sensor_data[sensor], axis=0)
                    flattened_data[sensor] = sensor_windows
            
            subject_minmax = compute_subject_minmax(flattened_data)
        
        # Create color coding transform
        print("\n4. Creating ColorCodingTransform...")
        # Get actual time steps from the data
        time_steps = list(windows_by_activity.values())[0]['data'].shape[1]
        print(f"   Time steps: {time_steps}")
        transform = ColorCodingTransform(
            sensors_order=sensors_order,
            time_steps_per_window=time_steps,
            sensor_band_height_px=10,
            output_format='HWC'
        )
        
        print(f"   Expected output shape: ({10 * len(sensors_order)}, {time_steps}, 3)")
        
        # Apply transform to each activity
        print("\n5. Applying color coding transform...")
        for activity_id, window_info in windows_by_activity.items():
            print(f"   Processing Activity {activity_id}: {window_info['activity_name']}")
            
            # Extract sensor data
            sensor_data = extract_sensor_data_from_window(window_info['data'], sensors_order)
            
            # Apply color coding transform
            try:
                color_image = transform(sensor_data, subject_minmax)
                images_by_activity[activity_id] = color_image
                print(f"      Output shape: {color_image.shape}")
                print(f"      Value range: [{color_image.min()}, {color_image.max()}]")
            except Exception as e:
                print(f"      Error: {e}")
        
        # Visualize results
        print("\n6. Creating visualization...")
        data_type = "normalized" if is_normalized else "raw"
        save_path = f"mhealth_color_coding_{data_type}.png"
        print(f"   Saving as: {save_path}")
        visualize_color_coded_images(images_by_activity, activity_labels, save_path=save_path)
        
        print("\n7. Example completed successfully!")
        print("   Color-coded images show different sensor patterns for each activity.")
        print("   Each horizontal band represents a different sensor (acc_chest, gyro_chest, etc.)")
        print("   Colors represent the 3D sensor values: Red=X, Green=Y, Blue=Z")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the MHEALTH dataset is available in the data/MHEALTH directory.")


if __name__ == "__main__":
    main()
