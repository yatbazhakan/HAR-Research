# Config-based Experiments

This document describes how to run HAR experiments directly from YAML configuration files, similar to the introspection system but focused on HAR training and sweeps.

## Overview

The config-based experiment system allows you to:
- Run single HAR experiments from YAML configs
- Run Weights & Biases sweeps for hyperparameter optimization
- Separate different experiment types (HAR, introspection, detection, extraction) into individual configs
- Use both GUI and command-line interfaces

## Quick Start

### 1. Using the GUI

1. Launch the main experiment launcher:
   ```bash
   python scripts/experiment_launcher.py
   ```

2. Click "Run from Config" to open the config-based experiment runner

3. Select an experiment configuration file (e.g., `configs/har_experiment_example.yaml`)

4. Optionally select a sweep configuration for hyperparameter optimization

5. Click "Run Experiment"

### 2. Using the Command Line

#### Single Experiment
```bash
python scripts/run_experiment_from_config.py configs/har_experiment_example.yaml
```

#### Sweep Experiment
```bash
python scripts/run_experiment_from_config.py configs/har_experiment_example.yaml --sweep --sweep-config configs/har_sweep_example.yaml
```

## Configuration Structure

### HAR Experiment Config

The HAR experiment configuration follows this structure:

```yaml
experiment_name: uci_har_cnn_tcn_experiment
experiment_path: ./
device: cuda:0
method: har_training
verbose: True

har_experiment:
  experiment_name: uci_har_cnn_tcn_training
  wandb:
    is_sweep: False
    project: har-experiments
    entity: your_entity
    mode: online
  
  method:
    task: multiclass
    cross_validation:
      type: holdout  # Options: holdout, kfold, loso, fold_json
      params:
        test_ratio: 0.2
        val_ratio: 0.1
    
    model:
      name: cnn_tcn  # Options: cnn_tcn, cnn_bilstm
      dropout: 0.1
    
    optimizer:
      type: Adam  # Options: Adam, SGD, AdamW
      params:
        lr: 0.001
        weight_decay: 0.001
    
    criterion:
      type: CrossEntropyLoss
      params:
        reduction: mean
    
    scheduler:
      type: ReduceLROnPlateau
      params:
        patience: 10
        factor: 0.5
    
    epochs: 100
    early_stop: 20
  
  dataset:
    name: uci_har  # Options: uci_har, pamap2, mhealth
    config:
      shards_glob: artifacts/preprocessed/uci_har/*.npz
      stats_file: artifacts/norm_stats/uci_har.json
      class_names: ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
      num_classes: 6
      input_shape: [6, 128]
  
  dataloader:
    train:
      batch_size: 64
      shuffle: True
      num_workers: 4
    val:
      batch_size: 64
      shuffle: False
      num_workers: 4
    test:
      batch_size: 1
      shuffle: False
      num_workers: 4
  
  evaluation:
    metrics: ["accuracy", "f1", "precision", "recall", "confusion_matrix"]
    plot_dir: artifacts/plots
    save_predictions: true
```

### Sweep Configuration

The sweep configuration defines the hyperparameter search space:

```yaml
method: bayes  # Options: bayes, random, grid
name: uci_har_cnn_tcn_sweep
metric:
  name: val/accuracy
  goal: maximize

early_terminate:
  type: hyperband
  min_iter: 10
  eta: 2

parameters:
  model:
    values:
      - cnn_tcn
      - cnn_bilstm
  
  optimizer:
    values:
      - Adam
      - SGD
      - AdamW
  
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  
  batch_size:
    values:
      - 16
      - 32
      - 64
      - 128
  
  epochs:
    values:
      - 50
      - 100
      - 150
      - 200
  
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
  
  weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-2
```

## Available Scripts

### Core Scripts

1. **`scripts/run_har_experiment.py`** - Runs single HAR experiments
2. **`scripts/run_har_sweep.py`** - Runs HAR sweeps with Weights & Biases
3. **`scripts/run_experiment_from_config.py`** - Universal experiment runner
4. **`scripts/config_experiment_gui.py`** - GUI for config-based experiments

### Example Configurations

1. **`configs/har_experiment_example.yaml`** - Example HAR experiment config
2. **`configs/har_sweep_example.yaml`** - Example sweep configuration

## Features

### Cross-Validation Support

- **Holdout**: Random train/validation/test split
- **K-Fold**: K-fold cross-validation
- **Leave-One-Subject-Out (LOSO)**: Subject-based splits
- **Fold JSON**: Pre-computed fold splits

### Model Support

- **CNN-TCN**: Temporal Convolutional Network
- **CNN-BiLSTM**: Convolutional + Bidirectional LSTM

### Optimizers

- **Adam**: Adaptive learning rate
- **SGD**: Stochastic Gradient Descent
- **AdamW**: Adam with weight decay

### Loss Functions

- **CrossEntropyLoss**: Standard classification loss
- **FocalLoss**: For handling class imbalance (planned)

### Evaluation Metrics

- Accuracy
- F1 Score (macro)
- Precision/Recall
- Expected Calibration Error (ECE)
- Confusion Matrix

## Usage Examples

### 1. Basic HAR Experiment

```bash
# Run a single experiment
python scripts/run_har_experiment.py configs/har_experiment_example.yaml
```

### 2. Hyperparameter Sweep

```bash
# Create and run a sweep
python scripts/run_har_sweep.py configs/har_experiment_example.yaml configs/har_sweep_example.yaml --action both --count 20
```

### 3. Using the Universal Runner

```bash
# Single experiment
python scripts/run_experiment_from_config.py configs/har_experiment_example.yaml

# Sweep experiment
python scripts/run_experiment_from_config.py configs/har_experiment_example.yaml --sweep --sweep-config configs/har_sweep_example.yaml
```

## Output Structure

Experiments create the following output structure:

```
experiments/
└── {experiment_name}/
    ├── evaluation_results.json
    ├── classification_report.json
    ├── predictions.csv
    └── checkpoints/
        ├── {model_name}_best.pth
        └── {model_name}_epoch_{N}.pth

artifacts/
├── plots/
│   ├── {experiment_name}_confusion_matrix.png
│   └── ...
└── checkpoints/
    └── ...
```

## Integration with Existing System

The config-based system integrates with your existing HAR research framework:

- Uses existing dataset classes (`UCIHARDataset`, `PAMAP2Dataset`, `MHealthDataset`)
- Uses existing model classes (`CNNTCN`, `CNNBiLSTM`)
- Uses existing transforms and metrics
- Compatible with existing preprocessing pipeline
- Supports Weights & Biases logging

## Future Extensions

The system is designed to be extensible for other experiment types:

- **Introspection experiments**: Similar structure but for model introspection
- **Detection experiments**: For object detection tasks
- **Extraction experiments**: For feature extraction tasks

Each experiment type will have its own configuration schema and specialized runner scripts.
