# Weights & Biases Sweeps for HAR Experiments

This guide explains how to use Weights & Biases (wandb) sweeps for hyperparameter tuning in your HAR experiments.

## Overview

Wandb sweeps allow you to:
- **Automatically search** for optimal hyperparameters
- **Track and compare** different runs
- **Visualize results** with interactive plots
- **Run multiple experiments** in parallel
- **Early termination** of poor-performing runs

## Setup

1. **Install wandb**:
   ```bash
   pip install wandb
   ```

2. **Login to wandb**:
   ```bash
   wandb login
   ```

## Using the GUI

### 1. Enable Sweep Mode
- Check "Use Sweep" in the WandB Sweep section
- Configure your base experiment settings
- Set the number of sweep runs (default: 10)

### 2. Create Sweep
- Click "Create Sweep" to generate a sweep configuration
- The GUI will create a default sweep config based on your dataset
- You'll get a sweep ID that you can use to start the sweep

### 3. Start Sweep
- Enter the sweep ID (or use the one from step 2)
- Click "Start Sweep" to begin hyperparameter search
- The sweep will run automatically in the background

### 4. Monitor Results
- Check the wandb dashboard: https://wandb.ai
- View real-time metrics and plots
- Compare different hyperparameter combinations

## Using the Command Line

### 1. Create a Sweep
```bash
python scripts/run_sweep.py create \
    --config artifacts/experiments/uci_har_experiment.yaml \
    --dataset uci_har \
    --method bayes \
    --project har-sweeps
```

### 2. Run a Sweep
```bash
python scripts/run_sweep.py run \
    --config artifacts/experiments/uci_har_experiment.yaml \
    --sweep_id YOUR_SWEEP_ID \
    --project har-sweeps \
    --count 20
```

## Sweep Configuration

### Methods
- **bayes**: Bayesian optimization (recommended)
- **random**: Random search
- **grid**: Grid search (exhaustive)

### Parameters
The sweep will optimize these hyperparameters:

#### Learning Rate
- **Distribution**: Log-uniform
- **Range**: 1e-5 to 1e-1
- **Purpose**: Controls training speed and convergence

#### Batch Size
- **Values**: [16, 32, 64, 128] (or [32, 64, 128, 256] for PAMAP2)
- **Purpose**: Affects training stability and memory usage

#### Epochs
- **Values**: [50, 100, 150, 200] (or [100, 200, 300, 400] for PAMAP2)
- **Purpose**: Training duration

#### Model Architecture
- **Values**: [cnn_tcn, cnn_bilstm]
- **Purpose**: Different backbone architectures

#### Dropout
- **Distribution**: Uniform
- **Range**: 0.1 to 0.5 (or 0.2 to 0.6 for PAMAP2)
- **Purpose**: Regularization to prevent overfitting

#### Weight Decay
- **Distribution**: Log-uniform
- **Range**: 1e-6 to 1e-2
- **Purpose**: L2 regularization

### Early Termination
- **Type**: Hyperband
- **Min Iterations**: 10
- **Eta**: 2
- **Purpose**: Stop poor-performing runs early

## Customizing Sweep Configurations

### 1. Edit Existing Config
```yaml
# artifacts/sweep_configs/uci_har_sweep.yaml
method: bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  # ... other parameters
```

### 2. Add New Parameters
```yaml
parameters:
  # ... existing parameters
  hidden_size:
    values: [64, 128, 256, 512]
  num_layers:
    values: [2, 3, 4, 5]
```

### 3. Change Search Method
```yaml
method: random  # or grid, bayes
```

## Best Practices

### 1. Start Small
- Begin with 10-20 runs to get a feel for the search space
- Increase count once you understand the parameter ranges

### 2. Monitor Progress
- Check wandb dashboard regularly
- Look for trends in the parameter importance plots
- Stop early if you find good results

### 3. Refine Search Space
- Narrow down ranges based on initial results
- Focus on the most important parameters
- Use more runs for promising regions

### 4. Use Multiple Metrics
- Consider both accuracy and F1-score
- Look at training stability (loss curves)
- Check for overfitting

### 5. Save Best Configurations
- Export the best hyperparameters
- Use them as starting points for further experiments
- Document what works for each dataset

## Example Workflow

1. **Prepare Data**:
   ```bash
   python scripts/preprocess.py --dataset uci_har
   python scripts/compute_norm_stats.py --shards_glob "artifacts/preprocessed/uci_har/*.npz"
   ```

2. **Create Sweep**:
   ```bash
   python scripts/run_sweep.py create --dataset uci_har --project har-sweeps
   ```

3. **Run Sweep**:
   ```bash
   python scripts/run_sweep.py run --sweep_id abc123 --count 50
   ```

4. **Analyze Results**:
   - Visit https://wandb.ai/your-username/har-sweeps
   - Look at the parallel coordinates plot
   - Check parameter importance
   - Export best configuration

5. **Run Final Experiment**:
   ```bash
   python scripts/train_baselines.py --config best_config.yaml
   ```

## Troubleshooting

### Common Issues

1. **"wandb not installed"**
   - Install: `pip install wandb`
   - Login: `wandb login`

2. **"Sweep ID not found"**
   - Check the sweep ID is correct
   - Ensure you're using the right project

3. **"No runs started"**
   - Check your base configuration
   - Ensure all required files exist
   - Verify wandb project permissions

4. **"Out of memory"**
   - Reduce batch size in sweep config
   - Use smaller models
   - Enable gradient checkpointing

### Getting Help

- Check wandb documentation: https://docs.wandb.ai
- View sweep examples: https://github.com/wandb/examples
- Join wandb community: https://community.wandb.ai

## Advanced Features

### 1. Multi-GPU Sweeps
- Use `wandb.agent()` with multiple processes
- Distribute runs across GPUs
- Use `torch.distributed` for large-scale sweeps

### 2. Custom Metrics
- Add custom metrics to your training script
- Use wandb.log() to track them
- Optimize for multiple objectives

### 3. Conditional Parameters
- Use wandb.config for conditional logic
- Enable/disable features based on parameters
- Create complex search spaces

### 4. Sweep Groups
- Organize related sweeps
- Compare across different datasets
- Track progress over time

## Integration with Existing Workflow

The sweep functionality integrates seamlessly with your existing HAR experiment workflow:

1. **Same training script**: `train_baselines.py`
2. **Same data format**: NPZ shards and normalization stats
3. **Same metrics**: All existing metrics are tracked
4. **Same logging**: Wandb integration works with sweeps
5. **Same visualization**: Plots and confusion matrices

This means you can easily switch between single experiments and sweeps without changing your core training code!
