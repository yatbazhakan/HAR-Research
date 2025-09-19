#!/usr/bin/env python3
"""
Simple Color Coding Test Script

This script loads one window from subject 1 MHEALTH data and tests the color coding transform.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from har.transforms.color_coding import compute_subject_minmax, ColorCodingTransform
from har.datasets.mhealth import load_mhealth_raw_stream


def test_color_coding():
    """Test color coding transform on one MHEALTH window."""
    print("Simple Color Coding Test")
    print("=" * 40)
    
    # Load MHEALTH data for subject 1
    print("1. Loading MHEALTH data for subject 1...")
    mhealth_data = load_mhealth_raw_stream("data/MHEALTH", fs=50, win_sec=3.0, overlap=0.5)
    
    # Filter for subject 1
    subject_data = mhealth_data[mhealth_data['subject_id'] == '1']
    print(f"   Found {len(subject_data)} windows for subject 1")
    
    if len(subject_data) == 0:
        print("   Error: No data found for subject 1")
        return
    
    # Get a window with Activity 1 (Standing still)
    activity_1_data = subject_data[subject_data['y'] == 1]
    if len(activity_1_data) > 0:
        window = activity_1_data.iloc[0]
        print(f"   Using Activity 1 (Standing still)")
    else:
        # Fallback to first available non-null activity
        non_null_data = subject_data[subject_data['y'] != 0]
        if len(non_null_data) > 0:
            window = non_null_data.iloc[0]
            print(f"   Activity 1 not found, using Activity {window['y']}")
        else:
            window = subject_data.iloc[0]
            print(f"   Warning: Only null activity available, using Activity {window['y']}")
    
    window_data = window['x']  # (C, T) - 23 channels, 150 time steps
    activity = window['y']
    
    print(f"   Window data shape: {window_data.shape}")
    print(f"   Activity: {activity}")
    
    # Define sensor order (7 sensors, 3 channels each)
    sensors_order = ['acc_chest', 'gyro_chest', 'mag_chest', 'acc_left_ankle', 
                    'gyro_left_ankle', 'acc_right_arm', 'gyro_right_arm']
    
    # Extract sensor data from window
    print("\n2. Extracting sensor data...")
    sensor_data = {}
    channels, time_steps = window_data.shape
    
    for i, sensor_name in enumerate(sensors_order):
        start_ch = i * 3
        end_ch = min(start_ch + 3, channels)
        
        if end_ch - start_ch == 3:
            # Extract 3 channels and transpose to (T, 3)
            sensor_channels = window_data[start_ch:end_ch, :]  # (3, T)
            sensor_data[sensor_name] = sensor_channels.T  # (T, 3)
        else:
            # Pad with zeros if not enough channels
            available_ch = end_ch - start_ch
            sensor_channels = np.zeros((3, time_steps))
            sensor_channels[:available_ch, :] = window_data[start_ch:end_ch, :]
            sensor_data[sensor_name] = sensor_channels.T  # (T, 3)
        
        print(f"   {sensor_name}: {sensor_data[sensor_name].shape}")
    
    # Compute min-max normalization
    print("\n3. Computing min-max normalization...")
    subject_minmax = compute_subject_minmax(sensor_data)
    
    for sensor_name, (min_vals, max_vals) in subject_minmax.items():
        print(f"   {sensor_name}: min={min_vals}, max={max_vals}")
    
    # Create color coding transform
    print("\n4. Creating ColorCodingTransform...")
    transform = ColorCodingTransform(
        sensors_order=sensors_order,
        time_steps_per_window=time_steps,
        sensor_band_height_px=10,
        output_format='HWC'
    )
    
    print(f"   Expected output shape: ({10 * len(sensors_order)}, {time_steps}, 3)")
    
    # Apply transform
    print("\n5. Applying color coding transform...")
    try:
        color_image = transform(sensor_data, subject_minmax)
        print(f"   ✓ Success! Output shape: {color_image.shape}")
        print(f"   Value range: [{color_image.min()}, {color_image.max()}]")
        
        # Visualize result
        print("\n6. Creating visualization...")
        plt.figure(figsize=(15, 8))
        plt.imshow(color_image, aspect='auto')
        activity_name = "Standing still" if activity == 1 else f"Activity {activity}"
        plt.title(f"MHEALTH Subject 1 - Activity {activity} ({activity_name}) - Color Coded Sensor Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Sensor Bands")
        
        # Add sensor labels
        sensor_names = sensors_order
        plt.yticks(range(5, color_image.shape[0], 10), sensor_names)
        
        plt.colorbar(label='Normalized Sensor Value')
        plt.tight_layout()
        plt.savefig("test_color_coding_simple.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   ✓ Visualization saved as: test_color_coding_simple.png")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    print("\n7. Test completed successfully!")
    print("   The color-coded image shows:")
    print("   - Each horizontal band represents a different sensor")
    print("   - Colors represent 3D sensor values: Red=X, Green=Y, Blue=Z")
    print("   - Time progresses from left to right")


if __name__ == "__main__":
    test_color_coding()
