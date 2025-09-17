# API Reference

Complete API documentation for the HAR project.

## 📦 Core Package (`har/`)

### Dataset Loading (`har.datasets`)

#### `load_dataset(name, root_dir, **kwargs)`
Unified dataset loading function.

**Parameters:**
- `name` (str): Dataset name ("uci_har", "pamap2", "mhealth")
- `root_dir` (str): Path to dataset directory
- `**kwargs`: Dataset-specific parameters

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
from har.datasets import load_dataset

# Load UCI-HAR dataset
df = load_dataset("uci_har", "data/UCI-HAR")

# Load PAMAP2 with custom parameters
df = load_dataset("pamap2", "data/PAMAP2", win_sec=2.56, overlap=0.5)
```

#### `NPZShardsDataset(shards_glob, split, stats=None)`
PyTorch Dataset for loading NPZ shards.

**Parameters:**
- `shards_glob` (str): Glob pattern for NPZ files
- `split` (str): Data split ("train", "val", "test", "all")
- `stats` (NormStats, optional): Normalization statistics

**Example:**
```python
from har.datasets import NPZShardsDataset

dataset = NPZShardsDataset(
    shards_glob="artifacts/preprocessed/uci_har/*.npz",
    split="train",
    stats=norm_stats
)
```

### Neural Network Modules (`har.modules`)

#### CNN Blocks

##### `ConvBlock1D(in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm1d, activation=nn.ReLU, dropout=0.0)`
Basic 1D Convolutional Block.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `kernel_size` (int): Convolution kernel size
- `stride` (int): Convolution stride
- `padding` (int, optional): Padding size
- `dilation` (int): Dilation rate
- `groups` (int): Number of groups
- `bias` (bool): Whether to use bias
- `norm_layer` (nn.Module): Normalization layer
- `activation` (nn.Module): Activation function
- `dropout` (float): Dropout rate

##### `ResidualBlock1D(in_channels, out_channels, kernel_size=3, stride=1, bottleneck=False, expansion=4, dropout=0.1)`
1D Residual Block with optional bottleneck.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `kernel_size` (int): Convolution kernel size
- `stride` (int): Convolution stride
- `bottleneck` (bool): Whether to use bottleneck design
- `expansion` (int): Expansion factor for bottleneck
- `dropout` (float): Dropout rate

##### `SEBlock1D(channels, reduction=16)`
Squeeze-and-Excitation Block for 1D signals.

**Parameters:**
- `channels` (int): Number of channels
- `reduction` (int): Reduction factor

#### RNN Blocks

##### `IndRNN(input_size, hidden_size, bias=True, nonlinearity='relu', recurrent_init_std=0.01)`
Independently Recurrent Neural Network.

**Parameters:**
- `input_size` (int): Input feature size
- `hidden_size` (int): Hidden state size
- `bias` (bool): Whether to use bias
- `nonlinearity` (str): Activation function ('relu', 'tanh')
- `recurrent_init_std` (float): Recurrent weight initialization std

##### `ConvLSTM1D(input_channels, hidden_channels, kernel_size, bias=True)`
1D Convolutional LSTM Cell.

**Parameters:**
- `input_channels` (int): Number of input channels
- `hidden_channels` (int): Number of hidden channels
- `kernel_size` (int): Convolution kernel size
- `bias` (bool): Whether to use bias

##### `LayerNormLSTM(input_size, hidden_size, bias=True)`
LSTM with Layer Normalization.

**Parameters:**
- `input_size` (int): Input feature size
- `hidden_size` (int): Hidden state size
- `bias` (bool): Whether to use bias

#### Transformer Blocks

##### `TimeSeriesTransformer(input_dim, d_model, num_heads, num_layers, max_seq_len=1000, dropout=0.1, output_dim=None, pos_encoding='sinusoidal', activation='relu')`
Complete Transformer model for time series data.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `d_model` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `num_layers` (int): Number of transformer layers
- `max_seq_len` (int): Maximum sequence length
- `dropout` (float): Dropout rate
- `output_dim` (int, optional): Output dimension
- `pos_encoding` (str): Positional encoding type
- `activation` (str): Activation function

##### `MultiHeadAttention(d_model, num_heads, dropout=0.1, batch_first=True)`
Multi-Head Attention mechanism.

**Parameters:**
- `d_model` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `dropout` (float): Dropout rate
- `batch_first` (bool): Whether batch is first dimension

#### Attention Modules

##### `SelfAttention(input_dim, hidden_dim=None, dropout=0.1)`
Self-Attention mechanism for sequence modeling.

**Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dim` (int, optional): Hidden dimension
- `dropout` (float): Dropout rate

##### `ChannelAttention(num_channels, reduction=16)`
Channel Attention Module (CAM).

**Parameters:**
- `num_channels` (int): Number of channels
- `reduction` (int): Reduction factor

##### `CBAM(num_channels, reduction=16, kernel_size=7)`
Convolutional Block Attention Module.

**Parameters:**
- `num_channels` (int): Number of channels
- `reduction` (int): Reduction factor
- `kernel_size` (int): Spatial attention kernel size

### Model Builders (`har.modules.model_builders`)

#### `create_har_model(architecture, input_dim, num_classes, sequence_length, **kwargs)`
Factory function to create HAR models.

**Parameters:**
- `architecture` (str): Model architecture ("cnn", "lstm", "transformer", "hybrid", "graph")
- `input_dim` (int): Input feature dimension
- `num_classes` (int): Number of output classes
- `sequence_length` (int): Input sequence length
- `**kwargs`: Architecture-specific parameters

**Returns:**
- `nn.Module`: PyTorch model

**Example:**
```python
from har.modules import create_har_model

# Create CNN model
model = create_har_model(
    architecture='cnn',
    input_dim=6,
    num_classes=6,
    sequence_length=128,
    conv_channels=[64, 128, 256],
    use_residual=True
)

# Create hybrid model
model = create_har_model(
    architecture='hybrid',
    input_dim=6,
    num_classes=6,
    sequence_length=128,
    conv_channels=[64, 128],
    lstm_hidden=128,
    transformer_layers=4
)
```

#### `create_model_from_config(config_name, input_dim, num_classes, sequence_length)`
Create model from predefined configuration.

**Parameters:**
- `config_name` (str): Configuration name
- `input_dim` (int): Input feature dimension
- `num_classes` (int): Number of output classes
- `sequence_length` (int): Input sequence length

**Available Configurations:**
- `cnn_small`: Small CNN model
- `cnn_large`: Large CNN model with attention
- `lstm_simple`: Simple LSTM model
- `lstm_advanced`: Advanced LSTM with multiple types
- `transformer_small`: Small transformer model
- `transformer_large`: Large transformer model
- `hybrid_balanced`: Balanced hybrid model

### Data Transforms (`har.transforms`)

#### `NormStats(mean, std)`
Normalization statistics dataclass.

**Methods:**
- `from_data(data)`: Compute stats from data
- `apply(data)`: Apply normalization to data
- `save(path)`: Save stats to file
- `load(path)`: Load stats from file

**Example:**
```python
from har.transforms import NormStats

# Compute normalization stats
stats = NormStats.from_data(train_data)

# Apply normalization
normalized_data = stats.apply(data)

# Save/load stats
stats.save("artifacts/norm_stats/dataset.json")
stats = NormStats.load("artifacts/norm_stats/dataset.json")
```

## 🛠️ Scripts (`scripts/`)

### Training Scripts

#### `train_baselines.py`
Main training script for HAR experiments.

**Key Arguments:**
- `--shards_glob`: Glob pattern for NPZ files
- `--stats`: Path to normalization stats JSON
- `--model`: Model architecture ("cnn_tcn", "cnn_bilstm")
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--cv`: Cross-validation method ("fold_json", "holdout", "kfold")
- `--wandb`: Enable Weights & Biases logging
- `--amp`: Enable mixed precision training

**Example:**
```bash
python scripts/train_baselines.py \
    --shards_glob "artifacts/preprocessed/uci_har/*.npz" \
    --stats "artifacts/norm_stats/uci_har.json" \
    --model "cnn_tcn" \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --cv "fold_json" \
    --fold_json "artifacts/folds/uci_har/loso_fold_subject_1.json" \
    --wandb
```

#### `run_sweep_new.py`
Enhanced sweep script for hyperparameter optimization.

**Actions:**
- `create`: Create a new sweep
- `run`: Run sweep agent

**Key Arguments:**
- `--action`: Action to perform
- `--config`: Base experiment configuration
- `--sweep_config`: Sweep configuration file
- `--project`: WandB project name
- `--sweep_id`: Sweep ID (for run action)
- `--count`: Number of sweep runs

**Example:**
```bash
# Create sweep
python scripts/run_sweep_new.py \
    --action create \
    --config "artifacts/experiments/base_config.yaml" \
    --project "har-sweeps"

# Run sweep
python scripts/run_sweep_new.py \
    --action run \
    --config "artifacts/experiments/base_config.yaml" \
    --sweep_id "sweep_id" \
    --project "har-sweeps" \
    --count 20
```

### Data Processing Scripts

#### `preprocess.py`
Dataset preprocessing script.

**Arguments:**
- `--dataset`: Dataset name ("uci_har", "pamap2", "mhealth")
- `--data_root`: Path to raw dataset
- `--win_sec`: Window size in seconds
- `--overlap`: Window overlap ratio
- `--fs`: Target sampling frequency

**Example:**
```bash
python scripts/preprocess.py \
    --dataset uci_har \
    --data_root data/UCI-HAR \
    --win_sec 2.56 \
    --overlap 0.5
```

#### `compute_norm_stats.py`
Compute normalization statistics.

**Arguments:**
- `--shards_glob`: Glob pattern for NPZ files
- `--split`: Data split ("train", "test", "all")
- `--output`: Output JSON file path

**Example:**
```bash
python scripts/compute_norm_stats.py \
    --shards_glob "artifacts/preprocessed/uci_har/*.npz" \
    --split train \
    --output "artifacts/norm_stats/uci_har.json"
```

#### `generate_loso_folds.py`
Generate Leave-One-Subject-Out cross-validation folds.

**Arguments:**
- `--shards_glob`: Glob pattern for NPZ files
- `--output_dir`: Output directory for fold files

**Example:**
```bash
python scripts/generate_loso_folds.py \
    --shards_glob "artifacts/preprocessed/uci_har/*.npz" \
    --output_dir "artifacts/folds/uci_har"
```

### GUI Scripts

#### `experiment_launcher.py`
Main launcher GUI for all tools.

**Usage:**
```bash
python scripts/experiment_launcher.py
```

#### `training_gui.py`
GUI for single experiment configuration and execution.

**Usage:**
```bash
python scripts/training_gui.py
```

#### `sweep_gui.py`
GUI for hyperparameter sweep configuration and execution.

**Usage:**
```bash
python scripts/sweep_gui.py
```

## 📊 Configuration Files

### Base Experiment Configuration
```yaml
# Dataset settings
dataset: uci_har
shards_glob: artifacts/preprocessed/uci_har/*.npz
stats: artifacts/norm_stats/uci_har.json
fold_json: artifacts/folds/uci_har/loso_fold_subject_1.json
class_names: "WALKING,WALKING_UPSTAIRS,WALKING_DOWNSTAIRS,SITTING,STANDING,LAYING"

# Model settings
model: cnn_tcn
epochs: 100
batch_size: 32
lr: 0.001
dropout: 0.2
weight_decay: 1e-4

# Training settings
cv: fold_json
num_workers: 4
amp: false

# Logging settings
wandb: true
wandb_project: har-experiments
plot_dir: artifacts/plots
```

### Sweep Configuration
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
  epochs:
    values: [50, 100, 150, 200]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
  weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-2
  model:
    values: ["cnn_tcn", "cnn_bilstm"]
```

## 🔧 Environment Variables

### WandB Configuration
```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="har-experiments"
export WANDB_ENTITY="your_entity"
```

### CUDA Configuration
```bash
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
```

### PyTorch Configuration
```bash
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDA_DSA=1
```

## 📈 Metrics and Evaluation

### Available Metrics
- **Accuracy**: `torchmetrics.MulticlassAccuracy`
- **F1-Score**: `torchmetrics.MulticlassF1Score`
- **Precision**: `torchmetrics.MulticlassPrecision`
- **Recall**: `torchmetrics.MulticlassRecall`
- **AUROC**: `torchmetrics.MulticlassAUROC`
- **Calibration Error**: `torchmetrics.MulticlassCalibrationError`
- **Confusion Matrix**: `torchmetrics.MulticlassConfusionMatrix`
- **Cohen's Kappa**: `torchmetrics.MulticlassCohenKappa`

### Model Calibration
```python
from har.train.metrics import TemperatureScaler

# Fit temperature scaler on validation set
scaler = TemperatureScaler()
scaler.fit(model, val_loader)

# Apply calibration to test predictions
calibrated_probs = scaler.predict_proba(test_logits)
```

This API reference provides comprehensive documentation for all major components of the HAR project. For more detailed examples, see the `examples/` directory and the individual module docstrings.
