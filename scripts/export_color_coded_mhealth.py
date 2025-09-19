#!/usr/bin/env python3
"""
Export color-coded RGB images from normalized MHEALTH shards.

This script loads preprocessed MHEALTH NPZ shards and exports per-window
color-coded RGB images using the ColorCodingTransform with 0-255 operation only.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from har.transforms.color_coding import ColorCodingTransform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MHEALTH activity mapping
ACTIVITY_MAPPING = {
    1: "standing_still",
    2: "sitting_relaxing", 
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_arms",
    8: "knees_bending_crouching",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back"
}

# Target activities to export
TARGET_ACTIVITIES = {
    1: "standing_still",
    2: "sitting_relaxing", 
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    9: "cycling",
    10: "jogging",
    11: "running"
}

def parse_sensors_order(sensors_order_str: str) -> List[str]:
    """Parse sensors order from JSON string."""
    try:
        return json.loads(sensors_order_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse sensors_order: {e}")
        sys.exit(1)

def group_channels_to_sensors(window_data: np.ndarray, 
                            channel_names: Optional[List[str]], 
                            sensors_order: List[str]) -> Dict[str, np.ndarray]:
    """Group channels into sensors based on channel names or assume triple grouping."""
    C, T = window_data.shape
    
    if channel_names is not None:
        # Group by channel names (e.g., 'acc_chest_x', 'acc_chest_y', 'acc_chest_z')
        sensor_data = {}
        for sensor in sensors_order:
            sensor_channels = []
            for axis in ['x', 'y', 'z']:
                channel_name = f"{sensor}_{axis}"
                if channel_name in channel_names:
                    channel_idx = channel_names.index(channel_name)
                    sensor_channels.append(window_data[channel_idx, :])
                else:
                    logger.warning(f"Channel {channel_name} not found in channel_names")
                    sensor_channels.append(np.zeros(T))
            
            if len(sensor_channels) == 3:
                sensor_data[sensor] = np.stack(sensor_channels, axis=1)  # (T, 3)
            else:
                logger.warning(f"Could not find 3 channels for sensor {sensor}")
                sensor_data[sensor] = np.zeros((T, 3))
    else:
        # Assume channels are already grouped in triples per sensors_order
        sensor_data = {}
        channels_per_sensor = C // len(sensors_order)
        
        for i, sensor in enumerate(sensors_order):
            start_idx = i * channels_per_sensor
            end_idx = min(start_idx + 3, C)
            
            if end_idx - start_idx == 3:
                sensor_data[sensor] = window_data[start_idx:end_idx, :].T  # (T, 3)
            else:
                logger.warning(f"Not enough channels for sensor {sensor}")
                sensor_data[sensor] = np.zeros((T, 3))
    
    return sensor_data

def process_window(window_data: np.ndarray, 
                  activity: int, 
                  subject_id: int,
                  window_idx: int,
                  shard_file: str,
                  sample_idx: int,
                  start_idx: int,
                  fs: int,
                  channel_names: Optional[List[str]],
                  sensors_order: List[str],
                  input_range: str,
                  output_dir: Path,
                  overwrite: bool,
                  debug: bool = False) -> Optional[Dict]:
    """Process a single window and save color-coded image."""
    
    # Check if we should export this activity
    if activity not in TARGET_ACTIVITIES:
        return None
    
    activity_name = TARGET_ACTIVITIES[activity]
    print(activity_name)
    # Validate time steps
    C, T = window_data.shape
    # if T != 150:
    #     logger.warning(f"Skipping window with T={T}, expected 150")
    #     return None
    
    # Group channels into sensors
    sensor_data = group_channels_to_sensors(window_data, channel_names, sensors_order)
    print(sensor_data.keys())
    # Validate all required sensors are present
    missing_sensors = set(sensors_order) - set(sensor_data.keys())
    if missing_sensors:
        logger.warning(f"Missing sensors {missing_sensors} for window {window_idx}")
        return None
    
    # Debug: Print data ranges for first few windows
    if debug and window_idx < 5:
        for sensor in sensor_data:
            data = sensor_data[sensor]
            min_val, max_val = np.min(data), np.max(data)
            logger.info(f"Window {window_idx}, Sensor {sensor}: min={min_val:.3f}, max={max_val:.3f}")
    
    # Data is already scaled in shards - use as-is
    
    # Create ColorCodingTransform
    # Since shards are already normalized/scaled, use appropriate mode based on input_range
    if input_range == "0_1":
        print("Scaling from [0,1] to [0,255]")
        normalize_mode = 'unit_to_255'  # Scale from [0,1] to [0,255]
    else:  # 0_255
        normalize_mode = 'none'  # Assume already in [0,255] range
    
    transform = ColorCodingTransform(
        sensors_order=sensors_order,
        time_steps_per_window=150,
        sensor_band_height_px=10,
        output_format='HWC',
        normalize_mode=normalize_mode
    )
    
    # Apply transform (subject_minmax=None for no normalization)
    try:
        rgb_image = transform(sensor_data, subject_minmax=None)
    except Exception as e:
        logger.error(f"Error applying transform to window {window_idx}: {e}")
        return None
    
    # Create output path
    subject_dir = output_dir / f"subject{subject_id:02d}" / activity_name
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = subject_dir / f"win_{window_idx:06d}.png"
    
    # Skip if file exists and not overwriting
    if image_path.exists() and not overwrite:
        logger.debug(f"Skipping existing file: {image_path}")
        return None
    
    # Save image
    try:
        Image.fromarray(rgb_image).save(image_path)
    except Exception as e:
        logger.error(f"Error saving image {image_path}: {e}")
        return None
    
    # Return manifest entry
    return {
        'image_path': str(image_path.relative_to(output_dir)),
        'subject_id': subject_id,
        'activity': activity,
        'activity_name': activity_name,
        'window_idx': window_idx,
        'shard_file': shard_file,
        'sample_idx': sample_idx,
        'start_idx': start_idx,
        'fs': fs
    }

def process_shard(shard_file: Path, 
                 sensors_order: List[str],
                 input_range: str,
                 output_dir: Path,
                 overwrite: bool,
                 global_window_idx: int,
                 debug: bool = False) -> Tuple[List[Dict], int]:
    """Process a single shard file."""
    logger.info(f"Processing shard: {shard_file}")
    
    try:
        data = np.load(shard_file, allow_pickle=False)
    except Exception as e:
        logger.error(f"Error loading shard {shard_file}: {e}")
        return [], global_window_idx
    
    X = data["X"]  # (N, C, T)
    y = data["y"]  # (N,)
    subject_ids = data.get("subject_id", np.zeros(len(y), dtype=int))
    start_indices = data.get("start_idx", np.zeros(len(y), dtype=int))
    fs_values = data.get("fs", np.full(len(y), 50, dtype=int))
    channel_names = data.get("channel_names", None)
    
    if channel_names is not None:
        channel_names = channel_names.tolist()
    
    manifest_entries = []
    current_window_idx = global_window_idx
    
    for i in range(len(X)):
        window_data = X[i]  # (C, T)
        activity = int(y[i])
        subject_id = int(subject_ids[i])
        start_idx = int(start_indices[i])
        fs = int(fs_values[i])
        
        # Process window
        entry = process_window(
            window_data=window_data,
            activity=activity,
            subject_id=subject_id,
            window_idx=current_window_idx,
            shard_file=shard_file.name,
            sample_idx=i,
            start_idx=start_idx,
            fs=fs,
            channel_names=channel_names,
            sensors_order=sensors_order,
            input_range=input_range,
            output_dir=output_dir,
            overwrite=overwrite,
            debug=debug
        )
        
        if entry is not None:
            manifest_entries.append(entry)
            current_window_idx += 1
    
    logger.info(f"Processed {len(manifest_entries)} windows from {shard_file}")
    return manifest_entries, current_window_idx

def main():
    parser = argparse.ArgumentParser(description="Export color-coded RGB images from MHEALTH shards")
    parser.add_argument("--shards_dir", type=str, required=True,
                       help="Directory containing MHEALTH NPZ shards")
    parser.add_argument("--outdir", type=str, required=True,
                       help="Output directory for color-coded images")
    parser.add_argument("--input_range", type=str, choices=["0_255", "0_1"], default="0_255",
                       help="Input data range: 0_255 (no scaling) or 0_1 (scale to 0-255)")
    parser.add_argument("--sensors_order", type=str, 
                       default='["acc_chest","gyro_chest","mag_chest","acc_left_ankle","gyro_left_ankle","acc_right_arm","gyro_right_arm"]',
                       help="JSON string defining sensor order")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads for parallel processing")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing files")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode to print data ranges")
    
    args = parser.parse_args()
    
    # Parse arguments
    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.outdir)
    sensors_order = parse_sensors_order(args.sensors_order)
    
    # Validate inputs
    if not shards_dir.exists():
        logger.error(f"Shards directory does not exist: {shards_dir}")
        sys.exit(1)
    
    # Find shard files
    shard_files = sorted(list(shards_dir.glob("*.npz")))
    if not shard_files:
        logger.error(f"No NPZ files found in {shards_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(shard_files)} shard files")
    logger.info(f"Sensors order: {sensors_order}")
    logger.info(f"Input range: {args.input_range}")
    logger.info(f"Target activities: {list(TARGET_ACTIVITIES.values())}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process shards
    all_manifest_entries = []
    global_window_idx = 0
    
    if args.workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for shard_file in shard_files:
                future = executor.submit(
                    process_shard, 
                    shard_file, 
                    sensors_order, 
                    args.input_range, 
                    output_dir, 
                    args.overwrite, 
                    global_window_idx,
                    args.debug
                )
                futures.append(future)
                global_window_idx += 10000  # Estimate for indexing
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing shards"):
                try:
                    manifest_entries, _ = future.result()
                    all_manifest_entries.extend(manifest_entries)
                except Exception as e:
                    logger.error(f"Error processing shard: {e}")
    else:
        # Sequential processing
        for shard_file in tqdm(shard_files, desc="Processing shards"):
            manifest_entries, global_window_idx = process_shard(
                shard_file, 
                sensors_order, 
                args.input_range, 
                output_dir, 
                args.overwrite, 
                global_window_idx,
                args.debug
            )
            all_manifest_entries.extend(manifest_entries)
    
    # Save manifest
    if all_manifest_entries:
        manifest_df = pd.DataFrame(all_manifest_entries)
        manifest_path = output_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Saved manifest with {len(manifest_df)} entries to {manifest_path}")
        
        # Print summary
        summary = manifest_df.groupby(['subject_id', 'activity_name']).size().unstack(fill_value=0)
        logger.info("Summary by subject and activity:")
        logger.info(f"\n{summary}")
        
        logger.info(f"Total images exported: {len(manifest_df)}")
        logger.info(f"Output directory: {output_dir}")
    else:
        logger.warning("No images were exported")
        sys.exit(1)

if __name__ == "__main__":
    main()
