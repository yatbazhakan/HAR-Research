# Quick Start Guide

Get up and running with the HAR project in minutes!

## 🚀 5-Minute Setup

### Option 1: Docker (Recommended)

```bash
# 1. Build the image
docker build -t har-project .

# 2. Run the container
docker run -it --gpus all -v $(pwd):/workspace har-project

# 3. Launch the GUI
python scripts/experiment_launcher.py
```

### Option 2: Local Installation

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install wandb torchmetrics tqdm pyyaml

# 2. Download datasets
python download_har_datasets.py

# 3. Launch the GUI
python scripts/experiment_launcher.py
```

## 🎯 First Experiment

### Step 1: Launch the GUI
```bash
python scripts/experiment_launcher.py
```

### Step 2: Choose Your Tool
- **Training GUI**: For single experiments
- **Sweep GUI**: For hyperparameter tuning
- **Quick Tests**: For validation

### Step 3: Configure Your Experiment

#### Using Training GUI:
1. Click "Open Training GUI"
2. Select dataset (UCI-HAR, PAMAP2, or MHEALTH)
3. Choose model (CNN-TCN or CNN-BiLSTM)
4. Set hyperparameters (epochs, batch size, learning rate)
5. Click "Run Experiment"

#### Using Sweep GUI:
1. Click "Open Sweep GUI"
2. Configure base experiment settings
3. Set sweep parameters and ranges
4. Click "Create Sweep"
5. Click "Run Sweep"

## 📊 Quick Test Suite

Run comprehensive tests on UCI-HAR dataset:

```bash
# Run quick tests
bash scripts/run_quick_tests.sh

# Or with tmux for detached execution
bash scripts/run_quick_tests.sh --tmux har_tests
```

This will:
- Preprocess UCI-HAR dataset
- Compute normalization statistics
- Generate cross-validation folds
- Run experiments with different models and CV strategies
- Generate plots and metrics

## 🔧 Command Line Usage

### Single Experiment
```bash
python scripts/train_baselines.py \
    --shards_glob "artifacts/preprocessed/uci_har/*.npz" \
    --stats "artifacts/norm_stats/uci_har.json" \
    --model "cnn_tcn" \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --cv "fold_json" \
    --fold_json "artifacts/folds/uci_har/loso_fold_subject_1.json" \
    --wandb
```

### Hyperparameter Sweep
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
    --sweep_id "your_sweep_id" \
    --project "har-sweeps" \
    --count 10
```

## 📈 View Results

### Weights & Biases Dashboard
1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. View experiment results, plots, and metrics

### Generated Plots
- **Confusion Matrix**: `artifacts/plots/confusion_matrix_*.png`
- **ROC Curves**: `artifacts/plots/roc_curves_*.png`
- **PR Curves**: `artifacts/plots/pr_curves_*.png`
- **Calibration Plots**: `artifacts/plots/calibration_*.png`

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure you're in the project root
cd HAR_Docker
python scripts/experiment_launcher.py
```

#### 2. CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run without GPU
CUDA_VISIBLE_DEVICES="" python scripts/train_baselines.py ...
```

#### 3. Dataset Not Found
```bash
# Download datasets first
python download_har_datasets.py

# Or preprocess manually
python scripts/preprocess.py --dataset uci_har --data_root data/UCI-HAR
```

#### 4. WandB Issues
```bash
# Login to WandB
wandb login

# Or run without WandB
python scripts/train_baselines.py ... --no-wandb
```

### Getting Help

1. **Check Logs**: Look at the log output in the GUI
2. **Command Line**: Use `--help` flag for any script
3. **Documentation**: Check `docs/` folder
4. **Examples**: Look at `examples/` folder

## 🎉 Next Steps

### Explore the Codebase
- **Core Package**: `har/` - Main HAR functionality
- **Scripts**: `scripts/` - Experiment scripts
- **Examples**: `examples/` - Usage examples
- **Docs**: `docs/` - Detailed documentation

### Try Different Models
- **CNN-TCN**: Good for temporal patterns
- **CNN-BiLSTM**: Good for sequential dependencies
- **Custom Models**: Use the modular components

### Experiment with Datasets
- **UCI-HAR**: 6 activities, smartphone data
- **PAMAP2**: 12 activities, multiple sensors
- **MHEALTH**: 12 activities, mobile health data

### Advanced Features
- **Hyperparameter Sweeps**: Automated optimization
- **Cross-Validation**: Different CV strategies
- **Model Calibration**: Temperature scaling
- **Attention Mechanisms**: Various attention modules

## 📚 Learn More

- **[Technical Overview](TECHNICAL_OVERVIEW.md)**: Deep dive into architecture
- **[GUI Tools](GUI_TOOLS.md)**: Complete GUI guide
- **[WandB Sweeps](WANDB_SWEEPS.md)**: Hyperparameter optimization
- **[Module Usage](examples/module_usage_example.py)**: Neural network modules

---

**Happy Experimenting! 🚀**
