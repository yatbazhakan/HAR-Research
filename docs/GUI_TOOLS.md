# HAR GUI Tools

This document describes the separated GUI tools for HAR experiments.

## Overview

The HAR experiment tools have been separated into distinct components for better organization and maintainability:

1. **Experiment Launcher** - Main entry point
2. **Training GUI** - Single experiment configuration and execution
3. **Sweep GUI** - Hyperparameter tuning with Weights & Biases
4. **Command Line Tools** - Advanced users and automation

## Tools

### 1. Experiment Launcher (`experiment_launcher.py`)

**Purpose**: Main entry point for all HAR experiment tools.

**Features**:
- Launch Training GUI
- Launch Sweep GUI
- Run Quick Tests
- Access command line tools
- Open project directory

**Usage**:
```bash
python scripts/experiment_launcher.py
```

### 2. Training GUI (`training_gui.py`)

**Purpose**: Configure and run single HAR experiments.

**Features**:
- Dataset configuration (auto-scan available datasets)
- Model selection (CNN-TCN, CNN-BiLSTM)
- Hyperparameter configuration
- Cross-validation settings (LOSO, Holdout, K-fold)
- Weights & Biases integration
- Tmux support for long-running experiments
- Real-time log output

**Usage**:
```bash
python scripts/training_gui.py
```

**Key Configuration Options**:
- **Dataset**: UCI-HAR, PAMAP2, MHEALTH
- **Model**: CNN-TCN, CNN-BiLSTM
- **CV Method**: Fold JSON, Holdout, K-fold
- **Logging**: WandB integration with plots
- **Execution**: Direct or Tmux session

### 3. Sweep GUI (`sweep_gui.py`)

**Purpose**: Configure and run hyperparameter sweeps using Weights & Biases.

**Features**:
- Base experiment configuration
- Sweep parameter configuration
- Multiple sweep methods (Bayes, Random, Grid)
- Parameter ranges (LR, batch size, epochs, dropout, weight decay)
- Model selection for sweeping
- Sweep creation and execution
- Real-time sweep monitoring

**Usage**:
```bash
python scripts/sweep_gui.py
```

**Sweep Configuration**:
- **Method**: Bayesian optimization, Random search, Grid search
- **Metric**: Validation accuracy, F1, Precision, Recall
- **Parameters**: Learning rate, batch size, epochs, dropout, weight decay
- **Models**: CNN-TCN, CNN-BiLSTM (multiple selection)

### 4. Command Line Tools

**Training Script**:
```bash
python scripts/train_baselines.py --help
```

**Sweep Script**:
```bash
python scripts/run_sweep_new.py --help
```

**Quick Tests**:
```bash
bash scripts/run_quick_tests.sh
```

## File Structure

```
scripts/
├── experiment_launcher.py    # Main launcher GUI
├── training_gui.py          # Single experiment GUI
├── sweep_gui.py             # Hyperparameter sweep GUI
├── train_baselines.py       # Training script
├── run_sweep_new.py         # Enhanced sweep script
├── run_quick_tests.sh       # Quick test suite
└── experiment_gui.py        # Legacy (deprecated)

artifacts/
├── sweep_configs/           # Sweep configuration files
├── plots/                   # Generated plots
└── logs/                    # Experiment logs
```

## Workflow Examples

### Single Experiment

1. Launch `experiment_launcher.py`
2. Click "Open Training GUI"
3. Configure dataset, model, and hyperparameters
4. Click "Run Experiment"

### Hyperparameter Sweep

1. Launch `experiment_launcher.py`
2. Click "Open Sweep GUI"
3. Configure base experiment settings
4. Set sweep parameters and ranges
5. Click "Create Sweep"
6. Click "Run Sweep"

### Quick Testing

1. Launch `experiment_launcher.py`
2. Click "Run Quick Tests"
3. Monitor progress in tmux session

## Configuration Files

### Base Experiment Config
```yaml
dataset: uci_har
shards_glob: artifacts/preprocessed/uci_har/*.npz
stats: artifacts/norm_stats/uci_har.json
fold_json: artifacts/folds/uci_har/loso_fold_subject_1.json
model: cnn_tcn
epochs: 100
batch_size: 32
lr: 0.001
# ... more parameters
```

### Sweep Config
```yaml
method: bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-1
  batch_size:
    values: [16, 32, 64, 128]
  # ... more parameters
```

## Benefits of Separation

1. **Cleaner Code**: Each tool has a single responsibility
2. **Better Maintainability**: Easier to update individual components
3. **Improved UX**: Focused interfaces for specific tasks
4. **Modularity**: Tools can be used independently
5. **Scalability**: Easy to add new tools or features

## Dependencies

- **GUI**: tkinter (built-in)
- **Sweep**: wandb, yaml
- **Training**: torch, torchmetrics, matplotlib
- **Data**: pandas, numpy

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **WandB Errors**: Check your WandB login and project permissions
3. **File Not Found**: Run preprocessing scripts first
4. **Tmux Issues**: Ensure tmux is installed and accessible

### Getting Help

1. Check the log output in the GUI
2. Use command line tools with `--help`
3. Check the documentation in `docs/`
4. Review the example scripts in `examples/`
