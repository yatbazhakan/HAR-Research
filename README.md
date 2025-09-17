# Human Activity Recognition (HAR) Project

A comprehensive deep learning framework for Human Activity Recognition using wearable sensor data, featuring state-of-the-art neural network architectures, automated preprocessing pipelines, and advanced hyperparameter optimization.

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t har-project .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace har-project

# Launch the experiment launcher
python scripts/experiment_launcher.py
```

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd HAR_Docker

# Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install wandb torchmetrics tqdm pyyaml

# Download datasets
python download_har_datasets.py

# Preprocess data
python scripts/preprocess.py --dataset uci_har --data_root data/UCI-HAR

# Run experiments
python scripts/experiment_launcher.py
```

## 📊 Supported Datasets

| Dataset | Description | Subjects | Activities | Sensors | Sampling Rate |
|---------|-------------|----------|------------|---------|---------------|
| **UCI-HAR** | Smartphone-based activity recognition | 30 | 6 | 3-axis accelerometer, gyroscope | 50Hz |
| **PAMAP2** | Physical Activity Monitoring | 9 | 12 | 3-axis accelerometer, gyroscope, magnetometer, heart rate | 100Hz |
| **MHEALTH** | Mobile health monitoring | 10 | 12 | 3-axis accelerometer, gyroscope, magnetometer | 50Hz |

## 🏗️ Project Architecture

```
HAR_Docker/
├── 📁 har/                          # Core HAR package
│   ├── 📁 datasets/                 # Dataset loaders
│   ├── 📁 models/                   # Model architectures
│   ├── 📁 modules/                  # Neural network building blocks
│   ├── 📁 transforms/               # Data preprocessing
│   └── 📁 train/                    # Training utilities
├── 📁 scripts/                      # Experiment scripts
│   ├── 🎯 training_gui.py          # Single experiment GUI
│   ├── 🔍 sweep_gui.py             # Hyperparameter sweep GUI
│   ├── 🚀 experiment_launcher.py   # Main launcher
│   ├── 🏋️ train_baselines.py      # Training script
│   └── ⚡ run_sweep_new.py         # Sweep execution
├── 📁 artifacts/                    # Generated outputs
│   ├── 📁 preprocessed/            # Processed datasets
│   ├── 📁 norm_stats/              # Normalization statistics
│   ├── 📁 folds/                   # Cross-validation folds
│   ├── 📁 plots/                   # Generated plots
│   └── 📁 sweep_configs/           # Sweep configurations
├── 📁 data/                         # Raw datasets
├── 📁 docs/                         # Documentation
└── 📁 examples/                     # Usage examples
```

## 🧠 Neural Network Architectures

### Core Models
- **CNN-TCN**: Convolutional Neural Network with Temporal Convolutional Network
- **CNN-BiLSTM**: Convolutional Neural Network with Bidirectional LSTM

### Advanced Modules
- **CNN Blocks**: ResNet, MobileNet, EfficientNet, Inception, SE-Net
- **RNN Blocks**: IndRNN, ConvLSTM, LayerNorm LSTM, Peephole LSTM, QRNN
- **Transformer Blocks**: Multi-head attention, positional encoding, feed-forward networks
- **GNN Blocks**: Graph Convolution, Graph Attention, GraphSAGE, ChebConv
- **Attention Modules**: Self-attention, cross-attention, CBAM, ECA, coordinate attention

## 🎯 Key Features

### 🔬 Comprehensive Experimentation
- **Single Experiments**: Configure and run individual experiments with custom hyperparameters
- **Hyperparameter Sweeps**: Automated optimization using Weights & Biases
- **Cross-Validation**: LOSO (Leave-One-Subject-Out), Holdout, K-fold
- **Quick Tests**: Predefined test suites for rapid validation

### 📊 Advanced Metrics & Visualization
- **Classification Metrics**: Accuracy, F1, Precision, Recall, AUROC, AUPR
- **Calibration**: Temperature scaling for model calibration
- **Visualizations**: Confusion matrices, ROC/PR curves, calibration plots
- **Logging**: Comprehensive experiment tracking with WandB

### 🛠️ Production-Ready Tools
- **GUI Interfaces**: User-friendly graphical interfaces for all operations
- **Docker Support**: Containerized environment for reproducibility
- **Modular Design**: Reusable components and building blocks
- **Comprehensive Logging**: Detailed experiment tracking and debugging

## 🚀 Usage Examples

### 1. Single Experiment

```python
# Using the Training GUI
python scripts/training_gui.py

# Or via command line
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

### 2. Hyperparameter Sweep

```python
# Using the Sweep GUI
python scripts/sweep_gui.py

# Or via command line
python scripts/run_sweep_new.py \
    --action create \
    --config "artifacts/experiments/base_config.yaml" \
    --project "har-sweeps" \
    --method "bayes" \
    --metric "val/accuracy"
```

### 3. Quick Testing

```bash
# Run comprehensive tests
bash scripts/run_quick_tests.sh

# Or with tmux for detached execution
bash scripts/run_quick_tests.sh --tmux har_tests
```

### 4. Using Neural Network Modules

```python
from har.modules import *

# Create a CNN-based model
model = create_model_from_config(
    config_name='cnn_large',
    input_dim=6,
    num_classes=6,
    sequence_length=128
)

# Create a hybrid CNN-LSTM-Transformer model
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

## 📋 Workflow

### 1. Data Preparation
```bash
# Download datasets
python download_har_datasets.py

# Preprocess data
python scripts/preprocess.py --dataset uci_har --data_root data/UCI-HAR
python scripts/preprocess.py --dataset pamap2 --data_root data/PAMAP2
python scripts/preprocess.py --dataset mhealth --data_root data/MHEALTH

# Compute normalization statistics
python scripts/compute_norm_stats.py --shards_glob "artifacts/preprocessed/uci_har/*.npz" --split train

# Generate cross-validation folds
python scripts/generate_loso_folds.py --shards_glob "artifacts/preprocessed/uci_har/*.npz"
```

### 2. Experiment Configuration
```bash
# Launch the main GUI
python scripts/experiment_launcher.py

# Choose your tool:
# - Training GUI: Single experiments
# - Sweep GUI: Hyperparameter optimization
# - Quick Tests: Validation suite
```

### 3. Model Training
- Configure dataset, model, and hyperparameters
- Select cross-validation strategy
- Enable Weights & Biases logging
- Run experiment (direct or in tmux)

### 4. Analysis
- View results in WandB dashboard
- Examine generated plots and metrics
- Analyze model performance and calibration

## 🔧 Configuration

### Environment Variables
```bash
# WandB configuration
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="har-experiments"

# CUDA configuration
export CUDA_VISIBLE_DEVICES="0,1"
```

### Configuration Files
- **Base Config**: `artifacts/experiments/base_config.yaml`
- **Sweep Config**: `artifacts/sweep_configs/sweep_config.yaml`
- **Model Configs**: `har/modules/model_builders.py`

## 📊 Results & Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area under the ROC curve
- **AUPR**: Area under the Precision-Recall curve
- **Calibration Error**: Model confidence calibration
- **Cohen's Kappa**: Inter-rater agreement

### Visualizations
- **Confusion Matrix**: Class-wise performance
- **ROC Curves**: True positive vs false positive rates
- **PR Curves**: Precision vs recall curves
- **Calibration Plots**: Predicted vs actual probabilities

## 🐳 Docker Details

### Base Image
- **PyTorch**: 2.2.2 with CUDA 12.1 and cuDNN 8
- **Python**: 3.10+
- **CUDA**: 12.1 support

### Pre-installed Datasets
- UCI-HAR Dataset
- PAMAP2 Dataset
- MHEALTH Dataset

### Dependencies
- PyTorch ecosystem (torch, torchvision, torchaudio)
- Scientific computing (numpy, pandas, scipy, scikit-learn)
- Visualization (matplotlib, seaborn)
- Experiment tracking (wandb)
- Jupyter Lab for interactive development

## 📚 Documentation

- **[GUI Tools](docs/GUI_TOOLS.md)**: Complete guide to graphical interfaces
- **[WandB Sweeps](docs/WANDB_SWEEPS.md)**: Hyperparameter optimization guide
- **[Module Usage](examples/module_usage_example.py)**: Neural network modules examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **UCI-HAR**: Human Activity Recognition Using Smartphones Dataset
- **PAMAP2**: Physical Activity Monitoring for Aging People Dataset
- **MHEALTH**: Mobile Health Dataset
- **PyTorch**: Deep learning framework
- **Weights & Biases**: Experiment tracking platform

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review example scripts in `examples/`

---

**Built with ❤️ for the Human Activity Recognition community**
