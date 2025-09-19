# Color Coding HAR System

This document describes the implementation of the color-coding approach for Human Activity Recognition (HAR) as described in your academic paper. The system converts sensor data into RGB images for CNN processing.

## Overview

The color-coding approach treats sensor data as visual information by:
- **Inertial Data**: Mapping x,y,z axes to R,G,B channels, stacked by sensor
- **ECG Data**: Combining two leads into RGB channels and resizing to square images
- **Per-subject Normalization**: Min-max scaling to 0-255 range for each subject
- **CNN Processing**: Using specialized CNN architectures for color-coded data

## Key Features

### 1. Color Coding Transforms

#### Inertial Data Color Coding
- Maps sensor axes (x,y,z) to RGB channels
- Stacks sensors vertically by sensor type
- Per-subject min-max normalization to 0-255 range
- Configurable window size and overlap

#### ECG Data Color Coding
- Two-stage median filtering for baseline wander removal
- Combines two ECG leads into RGB channels
- Resizes to square image format (e.g., 90x90x3)
- Per-subject normalization

### 2. CNN Models

#### InertialColorCNN
- Designed for (num_sensors, time_steps, 3) input
- Preserves sensor dimension with [1,2] pooling
- Configurable number of conv blocks and filters

#### ECGColorCNN
- Designed for (H, W, 3) input
- Standard 2D CNN with [2,2] pooling
- Configurable architecture parameters

#### MultiModalColorCNN
- Combines inertial and ECG data
- Multiple fusion methods: concat, add, attention
- End-to-end training

### 3. Dataset Support

- **MHEALTH**: Full multi-modal support (inertial + ECG)
- **UCI-HAR**: Inertial data only
- **PAMAP2**: Inertial data only
- Extensible for other datasets

## Usage

### 1. Basic Color-Coding Experiment

```bash
# Run a single color-coding experiment
python scripts/run_color_coding_experiment.py configs/color_coding_har_experiment.yaml
```

### 2. Using the Universal Runner

```bash
# Single experiment
python scripts/run_experiment_from_config.py configs/color_coding_har_experiment.yaml

# With sweep (when implemented)
python scripts/run_experiment_from_config.py configs/color_coding_har_experiment.yaml --sweep --sweep-config configs/color_coding_har_sweep.yaml
```

### 3. Using the GUI

```bash
# Launch the config experiment GUI
python scripts/config_experiment_gui.py
# Click "Color Coding HAR" to load the example
```

## Configuration Structure

### Color Coding Configuration

```yaml
color_coding_har_experiment:
  method:
    color_coding:
      data_type: both  # Options: inertial, ecg, both
      inertial:
        window_seconds: 1.0
        overlap: 0.5
        sensors_order: ["acc_chest", "gyro_chest", "mag_chest", ...]
        axes_per_sensor: 3
        color_mapping: {"R": "x", "G": "y", "B": "z"}
        per_subject_minmax_to_255: true
        time_steps_per_window: 50
      ecg:
        leads: 2
        baseline_wander_removal:
          method: "median_filter_two_stage"
          stage1_kernel: 200
          stage2_kernel: 600
        per_subject_minmax: true
        combine_leads_to_rgb: true
        resize_to: [90, 90, 3]
```

### Model Configuration

```yaml
model:
  type: multimodal  # Options: inertial, ecg, multimodal
  inertial_config:
    input_shape: [7, 50, 3]  # [num_sensors, time_steps, 3]
    conv_blocks: 4
    filters_per_block: 32
    kernel_size: [3, 3]
    pooling:
      type: "max"
      kernel: [1, 2]
      stride: [1, 2]
    fc_layers: 3
    dropout: 0.25
  ecg_config:
    input_shape: [90, 90, 3]  # [H, W, 3]
    conv_blocks: 4
    filters_per_block: 32
    kernel_size: [3, 3]
    pooling:
      type: "max"
      kernel: [2, 2]
      stride: [2, 2]
    fc_layers: 3
    dropout: 0.25
  fusion_method: "concat"  # Options: concat, add, attention
  fusion_dim: 256
```

## Implementation Details

### Color Coding Process

1. **Data Preprocessing**:
   - Load raw sensor data from NPZ shards
   - Apply per-subject min-max normalization
   - Handle windowing and overlap

2. **Inertial Color Coding**:
   - Reshape data to (num_sensors, time_steps, axes)
   - Map axes to RGB channels: x→R, y→G, z→B
   - Stack sensors vertically
   - Normalize to 0-255 range

3. **ECG Color Coding**:
   - Apply two-stage median filtering for baseline wander removal
   - Combine two leads into RGB channels
   - Resize to square image format
   - Normalize to 0-255 range

4. **CNN Processing**:
   - Feed color-coded data to specialized CNN
   - Extract features and classify activities
   - Support both single-modal and multi-modal processing

### Model Architectures

#### InertialColorCNN
- Input: (batch_size, num_sensors, time_steps, 3)
- Conv blocks with [1,2] pooling to preserve sensor dimension
- Global average pooling or flatten before FC layers
- Configurable architecture parameters

#### ECGColorCNN
- Input: (batch_size, H, W, 3)
- Standard 2D CNN with [2,2] pooling
- Global average pooling or flatten before FC layers
- Configurable architecture parameters

#### MultiModalColorCNN
- Processes both inertial and ECG data
- Individual CNNs for each modality
- Fusion layer for combining features
- Multiple fusion strategies available

## Dataset-Specific Implementations

### MHEALTH Dataset
- Full multi-modal support
- 7 inertial sensors + 2 ECG leads
- 12 activity classes
- Per-subject normalization

### UCI-HAR Dataset
- Inertial data only
- 6 sensor channels (3 acc + 3 gyro)
- 6 activity classes
- Standard color coding

### PAMAP2 Dataset
- Inertial data only
- Multiple sensor types
- 12 activity classes
- Configurable sensor selection

## Performance Considerations

### Memory Usage
- Color-coded data requires more memory than raw sensor data
- RGB images are 3x larger than single-channel data
- Consider batch size and model architecture

### Training Time
- CNN processing is typically faster than RNN-based approaches
- Multi-modal models require more computation
- Consider using mixed precision training

### Data Preprocessing
- Per-subject normalization is computationally expensive
- Baseline wander removal adds preprocessing time
- Consider caching preprocessed data

## Extending the System

### Adding New Datasets
1. Create dataset-specific color coding configuration
2. Implement data extraction logic in `ColorCodingDataset`
3. Update model input shapes and parameters
4. Add dataset-specific preprocessing if needed

### Adding New Modalities
1. Implement color coding transform for new modality
2. Create specialized CNN architecture
3. Update multi-modal fusion logic
4. Add configuration parameters

### Customizing Color Coding
1. Modify color mapping in `InertialColorCoding`
2. Adjust baseline wander removal in `ECGColorCoding`
3. Implement custom normalization strategies
4. Add new preprocessing steps

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce batch size
   - Use mixed precision training
   - Implement gradient checkpointing

2. **Data Shape Mismatches**:
   - Check input shape configuration
   - Verify color coding output shapes
   - Ensure model input shapes match

3. **Poor Performance**:
   - Adjust learning rate and optimizer settings
   - Try different model architectures
   - Check data preprocessing and normalization

4. **Convergence Issues**:
   - Use different initialization strategies
   - Adjust learning rate schedule
   - Try different loss functions

### Debugging Tips

1. **Visualize Color-Coded Data**:
   ```python
   import matplotlib.pyplot as plt
   plt.imshow(color_coded_data[0])  # First sample
   plt.show()
   ```

2. **Check Data Shapes**:
   ```python
   print(f"Inertial shape: {inertial_data.shape}")
   print(f"ECG shape: {ecg_data.shape}")
   ```

3. **Monitor Training**:
   - Use WandB logging
   - Plot loss and accuracy curves
   - Check gradient norms

## Future Enhancements

1. **Advanced Fusion Methods**:
   - Cross-modal attention
   - Transformer-based fusion
   - Graph neural networks

2. **Data Augmentation**:
   - Color space augmentations
   - Temporal augmentations
   - Sensor-specific augmentations

3. **Model Architectures**:
   - Vision transformers
   - EfficientNet-style architectures
   - Neural architecture search

4. **Multi-Task Learning**:
   - Activity recognition + sensor fusion
   - Transfer learning across datasets
   - Few-shot learning capabilities

## References

This implementation is based on the academic paper describing the color-coding approach for HAR. The system provides a complete framework for reproducing and extending the research results.

