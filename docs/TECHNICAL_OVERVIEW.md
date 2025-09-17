# Technical Overview

This document provides a comprehensive technical overview of the HAR project architecture, implementation details, and advanced usage patterns.

## 🏗️ System Architecture

### Core Components

#### 1. Data Pipeline (`har/datasets/`)
```python
# Unified dataset loading
from har.datasets import load_dataset

# Load any supported dataset
df_uci = load_dataset("uci_har", "data/UCI-HAR")
df_pamap2 = load_dataset("pamap2", "data/PAMAP2") 
df_mhealth = load_dataset("mhealth", "data/MHEALTH")
```

**Features:**
- **Unified API**: Single interface for all datasets
- **Automatic Preprocessing**: Windowing, resampling, normalization
- **Memory Efficient**: NPZ sharding for large datasets
- **Flexible Configuration**: Customizable window sizes and overlap rates

#### 2. Neural Network Modules (`har/modules/`)

**CNN Blocks:**
```python
from har.modules import ConvBlock1D, ResidualBlock1D, SEBlock1D

# Basic convolution
conv = ConvBlock1D(in_channels=6, out_channels=64, kernel_size=5)

# Residual connection
res_block = ResidualBlock1D(64, 128, dropout=0.2)

# Squeeze-and-Excitation
se_block = SEBlock1D(128, reduction=16)
```

**RNN Blocks:**
```python
from har.modules import IndRNN, ConvLSTM1D, LayerNormLSTM

# Independently Recurrent Neural Network
indrnn = IndRNN(input_size=64, hidden_size=128)

# Convolutional LSTM
convlstm = ConvLSTM1D(input_channels=64, hidden_channels=128, kernel_size=3)

# Layer Normalized LSTM
ln_lstm = LayerNormLSTM(input_size=64, hidden_size=128)
```

**Transformer Blocks:**
```python
from har.modules import TimeSeriesTransformer, MultiHeadAttention

# Complete transformer for time series
transformer = TimeSeriesTransformer(
    input_dim=6, d_model=256, num_heads=8, num_layers=6
)

# Multi-head attention
attention = MultiHeadAttention(d_model=256, num_heads=8)
```

**Graph Neural Networks:**
```python
from har.modules import GraphConvLayer, GraphAttentionLayer

# Graph convolution
gcn = GraphConvLayer(in_features=6, out_features=64)

# Graph attention
gat = GraphAttentionLayer(in_features=6, out_features=64, num_heads=4)
```

#### 3. Model Builders (`har/modules/model_builders.py`)

**High-Level Model Creation:**
```python
from har.modules import create_har_model, create_model_from_config

# Create model from predefined config
model = create_model_from_config(
    config_name='transformer_large',
    input_dim=6, num_classes=6, sequence_length=128
)

# Create custom hybrid model
model = create_har_model(
    architecture='hybrid',
    input_dim=6, num_classes=6, sequence_length=128,
    conv_channels=[64, 128],
    lstm_hidden=128,
    transformer_layers=4
)
```

### Data Processing Pipeline

#### 1. Raw Data Loading
```python
# UCI-HAR: 6 activities, 30 subjects, 50Hz
# PAMAP2: 12 activities, 9 subjects, 100Hz  
# MHEALTH: 12 activities, 10 subjects, 50Hz
```

#### 2. Windowing
```python
# Sliding window with configurable overlap
window_size = 128  # samples
overlap = 0.5      # 50% overlap
```

#### 3. Normalization
```python
# Per-channel normalization
norm_stats = NormStats.from_data(train_data)
normalized_data = norm_stats.apply(data)
```

#### 4. Cross-Validation
- **LOSO**: Leave-One-Subject-Out (subject-independent)
- **Holdout**: Random train/validation split
- **K-Fold**: Stratified k-fold cross-validation

## 🧠 Model Architectures

### CNN-TCN (Temporal Convolutional Network)
```python
class CNN_TCN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBlock1D(input_dim, 64, kernel_size=5),
            ConvBlock1D(64, 128, kernel_size=3),
            ConvBlock1D(128, 256, kernel_size=3)
        )
        
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            num_inputs=256,
            num_channels=[256, 256, 256],
            kernel_size=2,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
```

### CNN-BiLSTM
```python
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size=128):
        super().__init__()
        # CNN feature extraction
        self.conv_layers = nn.Sequential(
            ConvBlock1D(input_dim, 64, kernel_size=5),
            ConvBlock1D(64, 128, kernel_size=3)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
```

## 🔧 Training Configuration

### Optimizer Settings
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)
```

### Mixed Precision Training
```python
# Automatic Mixed Precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📊 Evaluation Metrics

### Classification Metrics
```python
from torchmetrics import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassCalibrationError,
    MulticlassConfusionMatrix, MulticlassCohenKappa
)

# Initialize metrics
accuracy = MulticlassAccuracy(num_classes=num_classes)
f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')
auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
```

### Model Calibration
```python
from har.train.metrics import TemperatureScaler

# Temperature scaling for calibration
scaler = TemperatureScaler()
scaler.fit(model, val_loader)
calibrated_probs = scaler.predict_proba(test_logits)
```

## 🎯 Hyperparameter Optimization

### Weights & Biases Sweep Configuration
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

### Sweep Execution
```python
# Create sweep
sweep_id = wandb.sweep(sweep_config, project="har-sweeps")

# Run sweep agent
wandb.agent(sweep_id, function=train_function, count=20)
```

## 🐳 Docker Environment

### Base Configuration
```dockerfile
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget unzip curl ca-certificates

# Install Python packages
RUN pip install --no-cache-dir \
    torchvision==0.17.2 \
    torchaudio==2.2.2 \
    scikit-learn pandas numpy matplotlib seaborn \
    jupyterlab scipy wandb torchmetrics tqdm pyyaml
```

### GPU Support
```bash
# Run with GPU support
docker run -it --gpus all -v $(pwd):/workspace har-project

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Performance Optimization

### Data Loading
```python
# Optimized DataLoader configuration
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### Model Optimization
```python
# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Set high precision for matrix operations
torch.set_float32_matmul_precision("high")

# Use non-blocking transfers
x = x.to(device, non_blocking=True)
```

### Memory Management
```python
# Gradient accumulation for large models
accumulation_steps = 4
for i, (x, y) in enumerate(dataloader):
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = criterion(logits, y) / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🔍 Debugging & Monitoring

### Logging Configuration
```python
import wandb

# Initialize WandB
wandb.init(
    project="har-experiments",
    config=config,
    tags=["cnn_tcn", "uci_har", "loso"]
)

# Log metrics
wandb.log({
    "train/loss": train_loss,
    "val/accuracy": val_accuracy,
    "val/f1": val_f1
})
```

### Visualization
```python
# Confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=pred_probs, y_true=targets, class_names=class_names
)})

# ROC curves
wandb.log({"roc_curves": wandb.plot.roc_curve(
    y_true=targets, y_probas=pred_probs, classes=class_names
)})
```

## 🚀 Advanced Usage

### Custom Model Architecture
```python
from har.modules import *

class CustomHARModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # CNN feature extraction
        self.conv_layers = nn.Sequential(
            ConvBlock1D(input_dim, 64, kernel_size=5),
            ResidualBlock1D(64, 128),
            SEBlock1D(128)
        )
        
        # Attention mechanism
        self.attention = SelfAttention(128, hidden_dim=64)
        
        # Classification head
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x, _ = self.attention(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
```

### Custom Loss Functions
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### Custom Metrics
```python
from torchmetrics import Metric

class CustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
    
    def compute(self):
        return self.correct.float() / self.total
```

This technical overview provides the foundation for understanding and extending the HAR project. For specific implementation details, refer to the source code and example scripts.
