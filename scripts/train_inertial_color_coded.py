#!/usr/bin/env python3
"""
Training script for inertial HAR models using color-coded inputs.

This script loads preprocessed NPZ shards, converts them to color-coded RGB images,
and trains a CNN model for human activity recognition.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    from sklearn.metrics import calibration_curve
import torchmetrics
from torchmetrics import (
    Accuracy, F1Score, Precision, Recall, Specificity, 
    AUROC, AveragePrecision, ConfusionMatrix, CalibrationError,
    CohenKappa, MatthewsCorrCoef
)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from har.transforms.color_coding import ColorCodingTransform
from har.models.color_coding_cnn import InertialColorCNN, ECGColorCNN, MultiModalColorCNN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(model_name, input_shape, num_classes, config):
    """Create the specified model architecture"""
    if model_name == "InertialColorCNN":
        return InertialColorCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_blocks=4,
            fc_layers=3,
            dropout=config['model']['dropout'],
        )
    elif model_name == "ECGColorCNN":
        return ECGColorCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_blocks=4,
            fc_layers=3,
            dropout=config['model']['dropout'],
        )
    elif model_name == "MultiModalColorCNN":
        return MultiModalColorCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_blocks=4,
            fc_layers=3,
            dropout=config['model']['dropout'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Activity mapping
ACTIVITY_MAPPING = {
    1: "standing_still",
    2: "sitting_relaxing", 
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_of_arms",
    8: "knees_bending_crouching",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back"
}

# Target activities (8 classes)
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

class ColorCodedHARDataset(Dataset):
    """Dataset for color-coded HAR data."""
    
    def __init__(self, 
                 shard_files: List[Path],
                 sensors_order: List[str],
                 target_activities: Dict[int, str],
                 time_steps_per_window: int = 150,
                 sensor_band_height_px: int = 10,
                 input_range: str = "0_1",
                 normalize_mode: str = "none",
                 height_compress: bool = True,
                 keep_full_height: bool = False,
                 ignore_class0: bool = False):
        
        self.sensors_order = sensors_order
        self.target_activities = target_activities
        self.time_steps_per_window = time_steps_per_window
        self.sensor_band_height_px = sensor_band_height_px
        self.input_range = input_range
        self.normalize_mode = normalize_mode
        self.height_compress = height_compress
        self.keep_full_height = keep_full_height
        self.ignore_class0 = ignore_class0
        
        # Create activity to index mapping
        self.activity_to_idx = {activity_id: idx for idx, activity_id in enumerate(sorted(target_activities.keys()))}
        self.idx_to_activity = {idx: activity_id for activity_id, idx in self.activity_to_idx.items()}
        
        # Load all data
        self.data = []
        self.labels = []
        self.subject_ids = []
        
        self._load_data(shard_files)
        
        logger.info(f"Loaded {len(self.data)} samples with {len(self.activity_to_idx)} classes")
        logger.info(f"Activity mapping: {self.activity_to_idx}")
    
    def _load_data(self, shard_files: List[Path]):
        """Load data from shard files."""
        for shard_file in shard_files:
            try:
                data = np.load(shard_file, allow_pickle=False)
                X = data["X"]  # (N, C, T)
                y = data["y"]  # (N,)
                subject_ids = data.get("subject_id", np.zeros(len(y), dtype=int))
                channel_names = data.get("channel_names", None)
                
                if channel_names is not None:
                    channel_names = channel_names.tolist()
                
                for i in range(len(X)):
                    activity = int(y[i])
                    
                    # Skip class 0 if ignore_class0 is True
                    if self.ignore_class0 and activity == 0:
                        continue
                    
                    if activity in self.target_activities:
                        self.data.append({
                            'window_data': X[i],  # (C, T)
                            'channel_names': channel_names,
                            'subject_id': int(subject_ids[i])
                        })
                        self.labels.append(self.activity_to_idx[activity])
                        self.subject_ids.append(int(subject_ids[i]))
                        
            except Exception as e:
                logger.error(f"Error loading shard {shard_file}: {e}")
    
    def _group_channels_to_sensors(self, window_data: np.ndarray, 
                                 channel_names: Optional[List[str]]) -> Dict[str, np.ndarray]:
        """Group channels into sensors."""
        C, T = window_data.shape
        
        if channel_names is not None:
            # Group by channel names
            sensor_data = {}
            for sensor in self.sensors_order:
                sensor_channels = []
                for axis in ['x', 'y', 'z']:
                    channel_name = f"{sensor}_{axis}"
                    if channel_name in channel_names:
                        channel_idx = channel_names.index(channel_name)
                        sensor_channels.append(window_data[channel_idx, :])
                    else:
                        sensor_channels.append(np.zeros(T))
                
                if len(sensor_channels) == 3:
                    sensor_data[sensor] = np.stack(sensor_channels, axis=1)  # (T, 3)
                else:
                    sensor_data[sensor] = np.zeros((T, 3))
        else:
            # Assume consecutive triples
            sensor_data = {}
            channels_per_sensor = C // len(self.sensors_order)
            
            for i, sensor in enumerate(self.sensors_order):
                start_idx = i * channels_per_sensor
                end_idx = min(start_idx + 3, C)
                
                if end_idx - start_idx == 3:
                    sensor_data[sensor] = window_data[start_idx:end_idx, :].T  # (T, 3)
                else:
                    sensor_data[sensor] = np.zeros((T, 3))
        
        return sensor_data
    
    def _apply_color_coding(self, window_data: np.ndarray, 
                          channel_names: Optional[List[str]]) -> np.ndarray:
        """Apply color coding transform to window data."""
        # Group channels into sensors
        sensor_data = self._group_channels_to_sensors(window_data, channel_names)
        
        # Scale data if needed
        if self.input_range == "0_1":
            # Scale from [0,1] to [0,255]
            for sensor in sensor_data:
                sensor_data[sensor] = np.clip(255.0 * sensor_data[sensor], 0, 255).astype(np.uint8)
        
        # Create ColorCodingTransform
        transform = ColorCodingTransform(
            sensors_order=self.sensors_order,
            time_steps_per_window=self.time_steps_per_window,
            sensor_band_height_px=self.sensor_band_height_px,
            output_format='HWC',
            normalize_mode=self.normalize_mode
        )
        
        # Apply transform
        rgb_image = transform(sensor_data, subject_minmax=None)  # (H, W, 3)
        
        # Compress height if enabled
        if self.height_compress and not self.keep_full_height:
            # Mean pool each 10-px band to 1 px
            compressed = np.zeros((len(self.sensors_order), self.time_steps_per_window, 3), dtype=np.uint8)
            for i in range(len(self.sensors_order)):
                start_h = i * self.sensor_band_height_px
                end_h = (i + 1) * self.sensor_band_height_px
                compressed[i] = np.mean(rgb_image[start_h:end_h], axis=0)
            return compressed  # (num_sensors, W, 3)
        
        return rgb_image  # (H, W, 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply color coding
        rgb_image = self._apply_color_coding(
            sample['window_data'], 
            sample['channel_names']
        )
        
        # Convert to tensor and normalize to [0,1]
        if rgb_image.dtype == np.uint8:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        
        # Convert to CHW format for PyTorch
        if rgb_image.ndim == 3:
            rgb_image = np.transpose(rgb_image, (2, 0, 1))  # (C, H, W)
        
        return torch.tensor(rgb_image, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Model will be loaded from YAML configuration

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_shard_files(shards_dir: str) -> List[Path]:
    """Get list of shard files."""
    shards_dir = Path(shards_dir)
    shard_files = sorted(list(shards_dir.glob("*.npz")))
    if not shard_files:
        raise ValueError(f"No NPZ files found in {shards_dir}")
    return shard_files

def create_kfold_splits(dataset: ColorCodedHARDataset, k: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Create k-fold cross-validation splits."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    
    indices = list(range(len(dataset)))
    for train_idx, val_idx in kfold.split(indices):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    return splits

def create_loso_splits(dataset: ColorCodedHARDataset) -> List[Tuple[List[int], List[int]]]:
    """Create Leave-One-Subject-Out splits."""
    splits = []
    unique_subjects = sorted(set(dataset.subject_ids))
    
    for test_subject in unique_subjects:
        train_idx = [i for i, subj in enumerate(dataset.subject_ids) if subj != test_subject]
        val_idx = [i for i, subj in enumerate(dataset.subject_ids) if subj == test_subject]
        splits.append((train_idx, val_idx))
    
    return splits

def create_holdout_splits(dataset: ColorCodedHARDataset, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Create holdout train/validation/test splits."""
    from sklearn.model_selection import train_test_split
    
    # Get all indices
    all_indices = list(range(len(dataset)))
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        all_indices, 
        test_size=test_ratio, 
        random_state=seed,
        stratify=[dataset.labels[i] for i in all_indices]
    )
    
    # Second split: separate train and validation from remaining data
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio/(1-test_ratio),  # Adjust val_ratio for remaining data
        random_state=seed,
        stratify=[dataset.labels[i] for i in train_val_idx]
    )
    
    return train_idx, val_idx, test_idx

def train_epoch(model, dataloader, criterion, optimizer, device, metrics, epoch=None, total_epochs=None, hide_batch_progress=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    
    # Reset metrics for this epoch
    metrics.reset()
    
    # Create progress bar only if not hidden
    if hide_batch_progress:
        pbar = dataloader
    else:
        progress_desc = f"Epoch {epoch+1}/{total_epochs}" if epoch is not None else "Training"
        pbar = tqdm(dataloader, desc=progress_desc, leave=False, ncols=100)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Check for NaN/Inf in input data
        if not torch.isfinite(data).all():
            logger.error(f"NaN/Inf detected in input data at batch {batch_idx}")
            continue
            
        optimizer.zero_grad()
        output = model(data)
        
        # Check for NaN/Inf in model output
        if not torch.isfinite(output).all():
            logger.error(f"NaN/Inf detected in model output at batch {batch_idx}")
            logger.error(f"Data stats: min={data.min().item():.6f}, max={data.max().item():.6f}")
            continue
            
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions and probabilities
        preds = output.argmax(dim=1)
        probs = torch.softmax(output, dim=1)
        
        predictions.extend(preds.detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
        probabilities.extend(probs.detach().cpu().numpy())
        
        # Update metrics
        metrics.update(preds, target, probs)
        
        # Update progress bar with current loss only if not hidden
        if not hide_batch_progress:
            current_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Check for NaN/Inf values
        if not np.isfinite(loss.item()):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx}")
            logger.error(f"Output stats: min={output.min().item():.6f}, max={output.max().item():.6f}")
            logger.error(f"Target stats: min={target.min().item()}, max={target.max().item()}")
            break
    
    # Compute final metrics
    epoch_metrics = metrics.compute()
    
    return total_loss / len(dataloader), epoch_metrics['accuracy'], epoch_metrics['f1_macro'], predictions, targets, probabilities

def validate_epoch(model, dataloader, criterion, device, metrics, epoch=None, total_epochs=None, hide_batch_progress=False):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    
    # Reset metrics for this epoch
    metrics.reset()
    
    # Create progress bar only if not hidden
    if hide_batch_progress:
        pbar = dataloader
    else:
        progress_desc = f"Val {epoch+1}/{total_epochs}" if epoch is not None else "Validation"
        pbar = tqdm(dataloader, desc=progress_desc, leave=False, ncols=100)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            preds = output.argmax(dim=1)
            probs = torch.softmax(output, dim=1)
            
            predictions.extend(preds.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            probabilities.extend(probs.detach().cpu().numpy())
            
            # Update metrics
            metrics.update(preds, target, probs)
            
            # Update progress bar with current loss only if not hidden
            if not hide_batch_progress:
                current_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
    
    # Compute final metrics
    epoch_metrics = metrics.compute()
    
    return total_loss / len(dataloader), epoch_metrics['accuracy'], epoch_metrics['f1_macro'], predictions, targets, probabilities

class HARMetrics:
    """Comprehensive metrics collection for HAR using torchmetrics."""
    
    def __init__(self, num_classes, class_names, device='cpu'):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.f1_macro = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.f1_micro = F1Score(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.f1_weighted = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        self.f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average='none').to(device)
        
        self.precision_macro = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.precision_micro = Precision(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.precision_per_class = Precision(task='multiclass', num_classes=num_classes, average='none').to(device)
        
        self.recall_macro = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.recall_micro = Recall(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.recall_per_class = Recall(task='multiclass', num_classes=num_classes, average='none').to(device)
        
        self.specificity = Specificity(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        # AUROC and AUPR
        self.auroc_macro = AUROC(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.auroc_per_class = AUROC(task='multiclass', num_classes=num_classes, average='none').to(device)
        self.aupr_macro = AveragePrecision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.aupr_per_class = AveragePrecision(task='multiclass', num_classes=num_classes, average='none').to(device)
        
        # Additional metrics
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
        self.cohen_kappa = CohenKappa(task='multiclass', num_classes=num_classes).to(device)
        self.matthews_corr = MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device)
        
        # Calibration error
        self.calibration_error = CalibrationError(task='multiclass', num_classes=num_classes).to(device)
    
    def update(self, preds, targets, probs=None):
        """Update metrics with batch predictions."""
        # Convert to tensors if needed
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)
        
        # Update all metrics
        self.accuracy.update(preds, targets)
        self.f1_macro.update(preds, targets)
        self.f1_micro.update(preds, targets)
        self.f1_weighted.update(preds, targets)
        self.f1_per_class.update(preds, targets)
        
        self.precision_macro.update(preds, targets)
        self.precision_micro.update(preds, targets)
        self.precision_per_class.update(preds, targets)
        
        self.recall_macro.update(preds, targets)
        self.recall_micro.update(preds, targets)
        self.recall_per_class.update(preds, targets)
        
        self.specificity.update(preds, targets)
        
        if probs is not None:
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs, device=self.device)
            
            self.auroc_macro.update(probs, targets)
            self.auroc_per_class.update(probs, targets)
            self.aupr_macro.update(probs, targets)
            self.aupr_per_class.update(probs, targets)
            self.calibration_error.update(probs, targets)
        
        self.confusion_matrix.update(preds, targets)
        self.cohen_kappa.update(preds, targets)
        self.matthews_corr.update(preds, targets)
    
    def compute(self):
        """Compute and return all metrics."""
        metrics = {
            # Basic metrics
            'accuracy': self.accuracy.compute().item(),
            'f1_macro': self.f1_macro.compute().item(),
            'f1_micro': self.f1_micro.compute().item(),
            'f1_weighted': self.f1_weighted.compute().item(),
            'f1_per_class': self.f1_per_class.compute().cpu().numpy(),
            
            'precision_macro': self.precision_macro.compute().item(),
            'precision_micro': self.precision_micro.compute().item(),
            'precision_per_class': self.precision_per_class.compute().cpu().numpy(),
            
            'recall_macro': self.recall_macro.compute().item(),
            'recall_micro': self.recall_micro.compute().item(),
            'recall_per_class': self.recall_per_class.compute().cpu().numpy(),
            
            'specificity': self.specificity.compute().item(),
            
            # Additional metrics
            'cohen_kappa': self.cohen_kappa.compute().item(),
            'matthews_corr': self.matthews_corr.compute().item(),
            
            # Confusion matrix
            'confusion_matrix': self.confusion_matrix.compute().cpu().numpy(),
        }
        
        # Add AUROC and AUPR if probabilities were provided
        try:
            metrics['auroc_macro'] = self.auroc_macro.compute().item()
            metrics['auroc_per_class'] = self.auroc_per_class.compute().cpu().numpy()
            metrics['aupr_macro'] = self.aupr_macro.compute().item()
            metrics['aupr_per_class'] = self.aupr_per_class.compute().cpu().numpy()
            metrics['calibration_error'] = self.calibration_error.compute().item()
        except:
            # If no probabilities were provided, set to None
            metrics['auroc_macro'] = None
            metrics['auroc_per_class'] = None
            metrics['aupr_macro'] = None
            metrics['aupr_per_class'] = None
            metrics['calibration_error'] = None
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.accuracy.reset()
        self.f1_macro.reset()
        self.f1_micro.reset()
        self.f1_weighted.reset()
        self.f1_per_class.reset()
        self.precision_macro.reset()
        self.precision_micro.reset()
        self.precision_per_class.reset()
        self.recall_macro.reset()
        self.recall_micro.reset()
        self.recall_per_class.reset()
        self.specificity.reset()
        self.auroc_macro.reset()
        self.auroc_per_class.reset()
        self.aupr_macro.reset()
        self.aupr_per_class.reset()
        self.confusion_matrix.reset()
        self.cohen_kappa.reset()
        self.matthews_corr.reset()
        self.calibration_error.reset()


def save_checkpoint(model, optimizer, epoch, best_metric, config, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }
    torch.save(checkpoint, filepath)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_auroc_aupr(y_true, y_pred_proba, class_names):
    """Calculate AUROC and AUPR for each class and overall."""
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Binarize labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Calculate AUROC for each class
    auroc_scores = []
    for i in range(len(class_names)):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            auroc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            auroc_scores.append(auroc)
        else:
            auroc_scores.append(0.0)
    
    # Calculate macro-averaged AUROC
    macro_auroc = np.mean(auroc_scores)
    
    # Calculate AUPR for each class
    aupr_scores = []
    for i in range(len(class_names)):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            aupr = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            aupr_scores.append(aupr)
        else:
            aupr_scores.append(0.0)
    
    # Calculate macro-averaged AUPR
    macro_aupr = np.mean(aupr_scores)
    
    return {
        'auroc_per_class': auroc_scores,
        'aupr_per_class': aupr_scores,
        'macro_auroc': macro_auroc,
        'macro_aupr': macro_aupr
    }

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot and save ROC curves for all classes."""
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Binarize labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(len(class_names)):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auroc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUROC = {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot and save Precision-Recall curves for all classes."""
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Binarize labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    for i in range(len(class_names)):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            aupr = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'{class_names[i]} (AUPR = {aupr:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Get predicted class and confidence
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    
    # Calculate ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def plot_calibration_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot calibration curves for all classes."""
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Binarize labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 8))
    
    # Plot calibration curve for each class
    for i in range(len(class_names)):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_bin[:, i], y_pred_proba[:, i], n_bins=10
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', linewidth=2, label=f'{class_names[i]}')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reliability_diagram(y_true, y_pred_proba, class_names, save_path, n_bins=10):
    """Plot reliability diagram showing calibration across confidence bins."""
    from sklearn.preprocessing import label_binarize
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Get predicted class and confidence
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    
    # Calculate bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    # Create reliability diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
    
    # Main calibration plot
    bin_centers = (bin_lowers + bin_uppers) / 2
    width = (bin_uppers - bin_lowers) * 0.8
    
    bars = ax1.bar(bin_centers, bin_accuracies, width=width, alpha=0.7, 
                   color='skyblue', edgecolor='navy', linewidth=1)
    
    # Add confidence values on top of bars
    for i, (acc, conf, count) in enumerate(zip(bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            ax1.text(bin_centers[i], acc + 0.01, f'{conf:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Count histogram
    ax2.bar(bin_centers, bin_counts, width=width, alpha=0.7, 
            color='lightcoral', edgecolor='darkred', linewidth=1)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Count per Bin')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_iterative_cv(args, config, dataset, device):
    """Run iterative cross-validation for all folds/subjects."""
    results = []
    
    # Initialize W&B for the entire CV run
    if args.wandb:
        try:
            import wandb
            from datetime import datetime
            
            # Create a single run name for the entire CV
            if args.protocol == "kfold":
                run_name = f"kfold_cv_{config['dataset']['splits']['k']}fold"
            elif args.protocol == "loso":
                unique_subjects = sorted(set(dataset.subject_ids))
                run_name = f"loso_cv_{len(unique_subjects)}subjects"
            else:  # holdout
                run_name = f"holdout_cv_{args.holdout_runs}runs"
            
            # Add timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_name}_{timestamp}"
            
            wandb.init(
                project=config['logging']['project'],
                entity=config['logging']['entity'],
                name=run_name,
                config=config,
                tags=[args.protocol, "cross_validation", "color_coded_har"]
            )
            
            logger.info(f"Initialized W&B run: {run_name}")
            
        except ImportError:
            logger.warning("wandb not available, falling back to offline logging")
            args.wandb = False
    
    if args.protocol == "kfold":
        k = config['dataset']['splits']['k']
        logger.info(f"Running {k}-fold cross-validation iteratively...")
        
        for fold in range(k):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold + 1}/{k} - Starting training...")
            logger.info(f"{'='*60}")
            
            # Create temporary args for this fold
            fold_args = argparse.Namespace(**vars(args))
            fold_args.fold = fold
            fold_args.run_all = False  # Prevent recursion
            fold_args.wandb = False  # Disable individual W&B runs
            
            # Run training for this fold
            fold_results = run_single_fold(fold_args, config, dataset, device, fold)
            results.append(fold_results)
            
            # Log to W&B with fold-specific naming
            if args.wandb:
                try:
                    import wandb
                    wandb.log({
                        f"fold_{fold+1}/best_val_loss": fold_results['best_val_loss'],
                        f"fold_{fold+1}/best_val_acc": fold_results['best_val_acc'],
                        f"fold_{fold+1}/best_val_f1": fold_results['best_val_f1'],
                        f"fold_{fold+1}/best_epoch": fold_results['best_epoch'],
                        f"fold_{fold+1}/final_train_loss": fold_results['final_train_loss'],
                        f"fold_{fold+1}/final_train_acc": fold_results['final_train_acc'],
                        f"fold_{fold+1}/final_train_f1": fold_results['final_train_f1'],
                        f"fold_{fold+1}/final_val_loss": fold_results['final_val_loss'],
                        f"fold_{fold+1}/final_val_acc": fold_results['final_val_acc'],
                        f"fold_{fold+1}/final_val_f1": fold_results['final_val_f1'],
                    })
                except:
                    pass
            
            # Log fold results immediately
            logger.info(f"FOLD {fold + 1} COMPLETED:")
            logger.info(f"  Best Epoch: {fold_results['best_epoch'] + 1}")
            logger.info(f"  Best Val Loss: {fold_results['best_val_loss']:.4f}")
            logger.info(f"  Final Train - Loss: {fold_results['final_train_loss']:.4f}, Acc: {fold_results['final_train_acc']:.4f}, F1: {fold_results['final_train_f1']:.4f}")
            logger.info(f"  Final Val   - Loss: {fold_results['final_val_loss']:.4f}, Acc: {fold_results['final_val_acc']:.4f}, F1: {fold_results['final_val_f1']:.4f}")
            logger.info(f"  Model saved: checkpoints/best_{fold_results['run_name']}.pt")
            logger.info(f"  Confusion matrix: checkpoints/confusion_matrix_{fold_results['run_name']}.png")
            
            # Show running statistics
            if len(results) > 1:
                val_accs = [r['final_val_acc'] for r in results]
                val_f1s = [r['final_val_f1'] for r in results]
                val_losses = [r['final_val_loss'] for r in results]
                logger.info(f"  Running Stats - Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}, Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
                
                # Log running averages to W&B
                if args.wandb:
                    try:
                        import wandb
                        wandb.log({
                            f"running_avg/val_loss": np.mean(val_losses),
                            f"running_avg/val_acc": np.mean(val_accs),
                            f"running_avg/val_f1": np.mean(val_f1s),
                            f"running_avg/val_loss_std": np.std(val_losses),
                            f"running_avg/val_acc_std": np.std(val_accs),
                            f"running_avg/val_f1_std": np.std(val_f1s),
                            f"running_avg/folds_completed": len(results),
                        })
                    except:
                        pass
            
    elif args.protocol == "loso":
        unique_subjects = sorted(set(dataset.subject_ids))
        logger.info(f"Running LOSO cross-validation for {len(unique_subjects)} subjects...")
        
        for i, subject in enumerate(unique_subjects):
            logger.info(f"\n{'='*60}")
            logger.info(f"SUBJECT {i + 1}/{len(unique_subjects)}: Subject {subject} - Starting training...")
            logger.info(f"{'='*60}")
            
            # Create temporary args for this subject
            subject_args = argparse.Namespace(**vars(args))
            subject_args.subject = subject
            subject_args.run_all = False  # Prevent recursion
            subject_args.wandb = False  # Disable individual W&B runs
            
            # Run training for this subject
            subject_results = run_single_fold(subject_args, config, dataset, device, subject)
            results.append(subject_results)
            
            # Log to W&B with subject-specific naming
            if args.wandb:
                try:
                    import wandb
                    wandb.log({
                        f"subject_{subject}/best_val_loss": subject_results['best_val_loss'],
                        f"subject_{subject}/best_val_acc": subject_results['best_val_acc'],
                        f"subject_{subject}/best_val_f1": subject_results['best_val_f1'],
                        f"subject_{subject}/best_epoch": subject_results['best_epoch'],
                        f"subject_{subject}/final_train_loss": subject_results['final_train_loss'],
                        f"subject_{subject}/final_train_acc": subject_results['final_train_acc'],
                        f"subject_{subject}/final_train_f1": subject_results['final_train_f1'],
                        f"subject_{subject}/final_val_loss": subject_results['final_val_loss'],
                        f"subject_{subject}/final_val_acc": subject_results['final_val_acc'],
                        f"subject_{subject}/final_val_f1": subject_results['final_val_f1'],
                    })
                except:
                    pass
            
            # Log subject results immediately
            logger.info(f"SUBJECT {subject} COMPLETED:")
            logger.info(f"  Best Epoch: {subject_results['best_epoch'] + 1}")
            logger.info(f"  Best Val Loss: {subject_results['best_val_loss']:.4f}")
            logger.info(f"  Final Train - Loss: {subject_results['final_train_loss']:.4f}, Acc: {subject_results['final_train_acc']:.4f}, F1: {subject_results['final_train_f1']:.4f}")
            logger.info(f"  Final Val   - Loss: {subject_results['final_val_loss']:.4f}, Acc: {subject_results['final_val_acc']:.4f}, F1: {subject_results['final_val_f1']:.4f}")
            logger.info(f"  Model saved: checkpoints/best_{subject_results['run_name']}.pt")
            logger.info(f"  Confusion matrix: checkpoints/confusion_matrix_{subject_results['run_name']}.png")
            
            # Show running statistics
            if len(results) > 1:
                val_accs = [r['final_val_acc'] for r in results]
                val_f1s = [r['final_val_f1'] for r in results]
                val_losses = [r['final_val_loss'] for r in results]
                logger.info(f"  Running Stats - Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}, Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
                
                # Log running averages to W&B
                if args.wandb:
                    try:
                        import wandb
                        wandb.log({
                            f"running_avg/val_loss": np.mean(val_losses),
                            f"running_avg/val_acc": np.mean(val_accs),
                            f"running_avg/val_f1": np.mean(val_f1s),
                            f"running_avg/val_loss_std": np.std(val_losses),
                            f"running_avg/val_acc_std": np.std(val_accs),
                            f"running_avg/val_f1_std": np.std(val_f1s),
                            f"running_avg/subjects_completed": len(results),
                        })
                    except:
                        pass
    
    elif args.protocol == "holdout":
        num_runs = args.holdout_runs
        logger.info(f"Running {num_runs} holdout experiments with random shuffling...")
        
        for run in range(num_runs):
            logger.info(f"\n{'='*60}")
            logger.info(f"HOLDOUT RUN {run + 1}/{num_runs} - Starting training...")
            logger.info(f"{'='*60}")
            
            # Create temporary args for this holdout run
            holdout_args = argparse.Namespace(**vars(args))
            holdout_args.run_all = False  # Prevent recursion
            holdout_args.wandb = False  # Disable individual W&B runs
            # Use different seed for each run to get different shuffles
            holdout_args.holdout_seed = config['training']['seed'] + run
            
            # Run training for this holdout run
            holdout_results = run_single_fold(holdout_args, config, dataset, device, run)
            results.append(holdout_results)
            
            # Log to W&B with run-specific naming
            if args.wandb:
                try:
                    import wandb
                    wandb.log({
                        f"holdout_run_{run+1}/best_val_loss": holdout_results['best_val_loss'],
                        f"holdout_run_{run+1}/best_val_acc": holdout_results['best_val_acc'],
                        f"holdout_run_{run+1}/best_val_f1": holdout_results['best_val_f1'],
                        f"holdout_run_{run+1}/best_epoch": holdout_results['best_epoch'],
                        f"holdout_run_{run+1}/final_train_loss": holdout_results['final_train_loss'],
                        f"holdout_run_{run+1}/final_train_acc": holdout_results['final_train_acc'],
                        f"holdout_run_{run+1}/final_train_f1": holdout_results['final_train_f1'],
                        f"holdout_run_{run+1}/final_val_loss": holdout_results['final_val_loss'],
                        f"holdout_run_{run+1}/final_val_acc": holdout_results['final_val_acc'],
                        f"holdout_run_{run+1}/final_val_f1": holdout_results['final_val_f1'],
                    })
                    
                    # Add test metrics if available
                    if 'test_loss' in holdout_results:
                        wandb.log({
                            f"holdout_run_{run+1}/test_loss": holdout_results['test_loss'],
                            f"holdout_run_{run+1}/test_acc": holdout_results['test_acc'],
                            f"holdout_run_{run+1}/test_f1": holdout_results['test_f1'],
                        })
                except:
                    pass
            
            # Log holdout results immediately
            logger.info(f"HOLDOUT RUN {run + 1} COMPLETED:")
            logger.info(f"  Best Epoch: {holdout_results['best_epoch'] + 1}")
            logger.info(f"  Best Val Loss: {holdout_results['best_val_loss']:.4f}")
            logger.info(f"  Final Train - Loss: {holdout_results['final_train_loss']:.4f}, Acc: {holdout_results['final_train_acc']:.4f}, F1: {holdout_results['final_train_f1']:.4f}")
            logger.info(f"  Final Val   - Loss: {holdout_results['final_val_loss']:.4f}, Acc: {holdout_results['final_val_acc']:.4f}, F1: {holdout_results['final_val_f1']:.4f}")
            logger.info(f"  Final Test  - Loss: {holdout_results.get('test_loss', 'N/A'):.4f}, Acc: {holdout_results.get('test_acc', 'N/A'):.4f}, F1: {holdout_results.get('test_f1', 'N/A'):.4f}")
            logger.info(f"  Model saved: checkpoints/best_{holdout_results['run_name']}.pt")
            logger.info(f"  Confusion matrix: checkpoints/confusion_matrix_{holdout_results['run_name']}.png")
            
            # Show running statistics
            if len(results) > 1:
                val_accs = [r['final_val_acc'] for r in results]
                val_f1s = [r['final_val_f1'] for r in results]
                val_losses = [r['final_val_loss'] for r in results]
                test_accs = [r.get('test_acc', 0) for r in results if 'test_acc' in r]
                test_f1s = [r.get('test_f1', 0) for r in results if 'test_f1' in r]
                test_losses = [r.get('test_loss', 0) for r in results if 'test_loss' in r]
                logger.info(f"  Running Stats - Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}, Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
                if test_accs:
                    logger.info(f"  Running Stats - Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}, Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
                
                # Log running averages to W&B
                if args.wandb:
                    try:
                        import wandb
                        log_dict = {
                            f"running_avg/val_loss": np.mean(val_losses),
                            f"running_avg/val_acc": np.mean(val_accs),
                            f"running_avg/val_f1": np.mean(val_f1s),
                            f"running_avg/val_loss_std": np.std(val_losses),
                            f"running_avg/val_acc_std": np.std(val_accs),
                            f"running_avg/val_f1_std": np.std(val_f1s),
                            f"running_avg/runs_completed": len(results),
                        }
                        
                        # Add test metrics if available
                        if test_accs:
                            log_dict.update({
                                f"running_avg/test_loss": np.mean(test_losses),
                                f"running_avg/test_acc": np.mean(test_accs),
                                f"running_avg/test_f1": np.mean(test_f1s),
                                f"running_avg/test_loss_std": np.std(test_losses),
                                f"running_avg/test_acc_std": np.std(test_accs),
                                f"running_avg/test_f1_std": np.std(test_f1s),
                            })
                        
                        wandb.log(log_dict)
                    except:
                        pass
    
    # Generate aggregated results and plots
    if results:
        generate_aggregated_results(results, args, config, dataset)
    
    # Finalize W&B run
    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    # Aggregate and save results
    aggregate_results(results, args.protocol)
    return results

def generate_aggregated_results(results, args, config, dataset):
    """Generate aggregated results, plots, and comprehensive logging."""
    logger.info("\n" + "="*80)
    logger.info("GENERATING AGGREGATED RESULTS AND PLOTS")
    logger.info("="*80)
    
    # Collect all predictions and targets
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []
    all_test_preds = []
    all_test_targets = []
    
    # Collect metrics for each fold/subject/run
    fold_metrics = []
    
    for i, result in enumerate(results):
        # Collect predictions and targets
        if 'train_predictions' in result:
            all_train_preds.extend(result['train_predictions'])
            all_train_targets.extend(result['train_targets'])
        
        if 'val_predictions' in result:
            all_val_preds.extend(result['val_predictions'])
            all_val_targets.extend(result['val_targets'])
        
        if 'test_predictions' in result:
            all_test_preds.extend(result['test_predictions'])
            all_test_targets.extend(result['test_targets'])
        
        # Store fold metrics
        fold_metrics.append({
            'fold_id': i,
            'val_acc': result['final_val_acc'],
            'val_f1': result['final_val_f1'],
            'val_loss': result['final_val_loss'],
            'test_acc': result.get('test_acc', None),
            'test_f1': result.get('test_f1', None),
            'test_loss': result.get('test_loss', None),
        })
    
    # Convert to numpy arrays
    all_train_preds = np.array(all_train_preds)
    all_train_targets = np.array(all_train_targets)
    all_val_preds = np.array(all_val_preds)
    all_val_targets = np.array(all_val_targets)
    
    if all_test_preds:
        all_test_preds = np.array(all_test_preds)
        all_test_targets = np.array(all_test_targets)
    
    # Calculate overall aggregated metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    
    # Overall metrics
    overall_train_acc = accuracy_score(all_train_targets, all_train_preds)
    overall_train_f1 = f1_score(all_train_targets, all_train_preds, average='macro')
    overall_val_acc = accuracy_score(all_val_targets, all_val_preds)
    overall_val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
    
    overall_test_acc = None
    overall_test_f1 = None
    if all_test_preds.size > 0:
        overall_test_acc = accuracy_score(all_test_targets, all_test_preds)
        overall_test_f1 = f1_score(all_test_targets, all_test_preds, average='macro')
    
    # Per-activity metrics
    activity_names = config['dataset']['labels']
    per_activity_metrics = {}
    
    for i, activity_name in enumerate(activity_names):
        # Filter predictions and targets for this activity
        train_mask = all_train_targets == i
        val_mask = all_val_targets == i
        test_mask = all_test_targets == i if all_test_preds.size > 0 else np.array([])
        
        if np.sum(train_mask) > 0:
            per_activity_metrics[activity_name] = {
                'train_acc': accuracy_score(all_train_targets[train_mask], all_train_preds[train_mask]),
                'train_f1': f1_score(all_train_targets[train_mask], all_train_preds[train_mask], average='macro'),
                'train_samples': np.sum(train_mask)
            }
        
        if np.sum(val_mask) > 0:
            per_activity_metrics[activity_name]['val_acc'] = accuracy_score(all_val_targets[val_mask], all_val_preds[val_mask])
            per_activity_metrics[activity_name]['val_f1'] = f1_score(all_val_targets[val_mask], all_val_preds[val_mask], average='macro')
            per_activity_metrics[activity_name]['val_samples'] = np.sum(val_mask)
        
        if all_test_preds.size > 0 and np.sum(test_mask) > 0:
            per_activity_metrics[activity_name]['test_acc'] = accuracy_score(all_test_targets[test_mask], all_test_preds[test_mask])
            per_activity_metrics[activity_name]['test_f1'] = f1_score(all_test_targets[test_mask], all_test_preds[test_mask], average='macro')
            per_activity_metrics[activity_name]['test_samples'] = np.sum(test_mask)
    
    # Generate plots
    create_aggregated_plots(all_val_preds, all_val_targets, all_test_preds, all_test_targets, 
                           activity_names, args.protocol, fold_metrics)
    
    # Log comprehensive results
    log_aggregated_results(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1, 
                          overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics, 
                          args, config)
    
    # Log to W&B
    if args.wandb:
        log_aggregated_to_wandb(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1,
                               overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics,
                               all_val_preds, all_val_targets, all_test_preds, all_test_targets,
                               activity_names, args.protocol)

def create_aggregated_plots(val_preds, val_targets, test_preds, test_targets, activity_names, protocol, fold_metrics):
    """Create aggregated confusion matrices and performance plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Create output directory
    os.makedirs('artifacts/plots', exist_ok=True)
    
    # 1. Overall Confusion Matrix (Validation)
    plt.figure(figsize=(10, 8))
    cm_val = confusion_matrix(val_targets, val_preds)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names)
    plt.title(f'Aggregated Confusion Matrix - Validation Set ({protocol.upper()})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'artifacts/plots/confusion_matrix_aggregated_{protocol}_val.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Overall Confusion Matrix (Test) - if available
    if test_preds.size > 0:
        plt.figure(figsize=(10, 8))
        cm_test = confusion_matrix(test_targets, test_preds)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=activity_names, yticklabels=activity_names)
        plt.title(f'Aggregated Confusion Matrix - Test Set ({protocol.upper()})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'artifacts/plots/confusion_matrix_aggregated_{protocol}_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Per-fold/subject Performance Comparison
    plt.figure(figsize=(12, 8))
    fold_ids = [f"Fold {i+1}" if protocol == "kfold" else f"Subject {i+1}" if protocol == "loso" else f"Run {i+1}" for i in range(len(fold_metrics))]
    val_accs = [m['val_acc'] for m in fold_metrics]
    val_f1s = [m['val_f1'] for m in fold_metrics]
    
    x = np.arange(len(fold_ids))
    width = 0.35
    
    plt.bar(x - width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
    plt.bar(x + width/2, val_f1s, width, label='Validation F1-Score', alpha=0.8)
    
    plt.xlabel('Fold/Subject/Run')
    plt.ylabel('Score')
    plt.title(f'Performance Comparison Across {protocol.upper()} Runs')
    plt.xticks(x, fold_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'artifacts/plots/performance_comparison_{protocol}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot of performance distribution
    plt.figure(figsize=(8, 6))
    plt.boxplot([val_accs, val_f1s], labels=['Validation Accuracy', 'Validation F1-Score'])
    plt.title(f'Performance Distribution Across {protocol.upper()} Runs')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'artifacts/plots/performance_distribution_{protocol}.png', dpi=300, bbox_inches='tight')
    plt.close()

def log_aggregated_results(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1,
                          overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics,
                          args, config):
    """Log comprehensive aggregated results to console and file."""
    
    # Console logging
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS SUMMARY")
    logger.info("="*80)
    
    # Overall results
    logger.info(f"\nOVERALL RESULTS ({args.protocol.upper()}):")
    logger.info(f"  Training   - Accuracy: {overall_train_acc:.4f}, F1-Score: {overall_train_f1:.4f}")
    logger.info(f"  Validation - Accuracy: {overall_val_acc:.4f}, F1-Score: {overall_val_f1:.4f}")
    if overall_test_acc is not None:
        logger.info(f"  Test       - Accuracy: {overall_test_acc:.4f}, F1-Score: {overall_test_f1:.4f}")
    
    # Per-fold/subject statistics
    val_accs = [m['val_acc'] for m in fold_metrics]
    val_f1s = [m['val_f1'] for m in fold_metrics]
    
    logger.info(f"\nPER-FOLD/SUBJECT STATISTICS:")
    logger.info(f"  Validation Accuracy - Mean: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    logger.info(f"  Validation F1-Score  - Mean: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    logger.info(f"  Best Validation Acc: {np.max(val_accs):.4f}")
    logger.info(f"  Best Validation F1:  {np.max(val_f1s):.4f}")
    
    # Per-activity results
    logger.info(f"\nPER-ACTIVITY RESULTS:")
    logger.info(f"{'Activity':<20} {'Train Acc':<10} {'Val Acc':<10} {'Val F1':<10} {'Samples':<10}")
    logger.info("-" * 70)
    
    for activity_name, metrics in per_activity_metrics.items():
        train_acc = metrics.get('train_acc', 0.0)
        val_acc = metrics.get('val_acc', 0.0)
        val_f1 = metrics.get('val_f1', 0.0)
        val_samples = metrics.get('val_samples', 0)
        
        logger.info(f"{activity_name:<20} {train_acc:<10.4f} {val_acc:<10.4f} {val_f1:<10.4f} {val_samples:<10}")
    
    # Save detailed results to file
    save_aggregated_results_to_file(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1,
                                   overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics,
                                   args, config)

def save_aggregated_results_to_file(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1,
                                   overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics,
                                   args, config):
    """Save detailed aggregated results to JSON file."""
    import json
    from datetime import datetime
    
    results_data = {
        'protocol': args.protocol,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'overall_results': {
            'train_accuracy': float(overall_train_acc),
            'train_f1': float(overall_train_f1),
            'val_accuracy': float(overall_val_acc),
            'val_f1': float(overall_val_f1),
        },
        'per_activity_results': per_activity_metrics,
        'fold_metrics': fold_metrics,
        'statistics': {
            'val_acc_mean': float(np.mean([m['val_acc'] for m in fold_metrics])),
            'val_acc_std': float(np.std([m['val_acc'] for m in fold_metrics])),
            'val_f1_mean': float(np.mean([m['val_f1'] for m in fold_metrics])),
            'val_f1_std': float(np.std([m['val_f1'] for m in fold_metrics])),
        }
    }
    
    if overall_test_acc is not None:
        results_data['overall_results']['test_accuracy'] = float(overall_test_acc)
        results_data['overall_results']['test_f1'] = float(overall_test_f1)
    
    # Save to file
    os.makedirs('artifacts/results', exist_ok=True)
    filename = f'artifacts/results/aggregated_results_{args.protocol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {filename}")

def log_aggregated_to_wandb(overall_train_acc, overall_train_f1, overall_val_acc, overall_val_f1,
                           overall_test_acc, overall_test_f1, per_activity_metrics, fold_metrics,
                           val_preds, val_targets, test_preds, test_targets, activity_names, protocol):
    """Log aggregated results to W&B."""
    try:
        import wandb
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Overall metrics
        wandb.log({
            "aggregated/overall_train_acc": overall_train_acc,
            "aggregated/overall_train_f1": overall_train_f1,
            "aggregated/overall_val_acc": overall_val_acc,
            "aggregated/overall_val_f1": overall_val_f1,
        })
        
        if overall_test_acc is not None:
            wandb.log({
                "aggregated/overall_test_acc": overall_test_acc,
                "aggregated/overall_test_f1": overall_test_f1,
            })
        
        # Per-activity metrics
        for activity_name, metrics in per_activity_metrics.items():
            activity_key = activity_name.lower().replace(' ', '_')
            wandb.log({
                f"aggregated/per_activity/{activity_key}_train_acc": metrics.get('train_acc', 0.0),
                f"aggregated/per_activity/{activity_key}_val_acc": metrics.get('val_acc', 0.0),
                f"aggregated/per_activity/{activity_key}_val_f1": metrics.get('val_f1', 0.0),
            })
        
        # Statistics
        val_accs = [m['val_acc'] for m in fold_metrics]
        val_f1s = [m['val_f1'] for m in fold_metrics]
        
        wandb.log({
            "aggregated/stats/val_acc_mean": np.mean(val_accs),
            "aggregated/stats/val_acc_std": np.std(val_accs),
            "aggregated/stats/val_f1_mean": np.mean(val_f1s),
            "aggregated/stats/val_f1_std": np.std(val_f1s),
            "aggregated/stats/val_acc_max": np.max(val_accs),
            "aggregated/stats/val_f1_max": np.max(val_f1s),
        })
        
        # Confusion matrices
        cm_val = confusion_matrix(val_targets, val_preds)
        wandb.log({"aggregated/confusion_matrix_val": wandb.plot.confusion_matrix(
            probs=None, y_true=val_targets, preds=val_preds, class_names=activity_names
        )})
        
        if test_preds.size > 0:
            cm_test = confusion_matrix(test_targets, test_preds)
            wandb.log({"aggregated/confusion_matrix_test": wandb.plot.confusion_matrix(
                probs=None, y_true=test_targets, preds=test_preds, class_names=activity_names
            )})
        
        # Classification report
        val_report = classification_report(val_targets, val_preds, target_names=activity_names, output_dict=True)
        wandb.log({"aggregated/classification_report_val": val_report})
        
        if test_preds.size > 0:
            test_report = classification_report(test_targets, test_preds, target_names=activity_names, output_dict=True)
            wandb.log({"aggregated/classification_report_test": test_report})
        
    except Exception as e:
        logger.warning(f"Failed to log aggregated results to W&B: {e}")

def run_single_fold(args, config, dataset, device, fold_id):
    """Run training for a single fold/subject."""
    # Create splits
    if args.protocol == "kfold":
        splits = create_kfold_splits(dataset, config['dataset']['splits']['k'], config['dataset']['splits']['seed'])
        train_idx, val_idx = splits[args.fold]
        run_name = f"kfold_fold_{args.fold}"
    elif args.protocol == "loso":
        splits = create_loso_splits(dataset)
        unique_subjects = sorted(set(dataset.subject_ids))
        subject_index = unique_subjects.index(args.subject)
        train_idx, val_idx = splits[subject_index]
        run_name = f"loso_subject_{args.subject}"
    else:  # holdout
        # Use different seed for each holdout run if specified
        holdout_seed = getattr(args, 'holdout_seed', config['training']['seed'])
        train_idx, val_idx, test_idx = create_holdout_splits(
            dataset, 
            test_ratio=args.test_ratio, 
            val_ratio=args.val_ratio, 
            seed=holdout_seed
        )
        run_name = f"holdout_run_{fold_id}_test{args.test_ratio}_val{args.val_ratio}"
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create test loader for holdout validation
    test_loader = None
    if args.protocol == "holdout":
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if args.protocol == "holdout":
        logger.info(f"Test samples: {len(test_dataset)}")
    
    # Log class distributions if not hidden
    if not args.hide_class_distribution:
        logger.info("\n" + "="*60)
        logger.info("CLASS DISTRIBUTIONS")
        logger.info("="*60)
        
        # Get class distributions for train set
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.cpu().numpy())
        train_labels = np.array(train_labels)
        
        # Get class distributions for val set
        val_labels = []
        for _, labels in val_loader:
            val_labels.extend(labels.cpu().numpy())
        val_labels = np.array(val_labels)
        
        # Get unique classes and their counts
        unique_classes = sorted(np.unique(np.concatenate([train_labels, val_labels])))
        
        logger.info(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Train%':<10} {'Val%':<10}")
        logger.info("-" * 60)
        
        for class_id in unique_classes:
            train_count = np.sum(train_labels == class_id)
            val_count = np.sum(val_labels == class_id)
            train_pct = (train_count / len(train_labels)) * 100
            val_pct = (val_count / len(val_labels)) * 100
            
            logger.info(f"{class_id:<15} {train_count:<10} {val_count:<10} {train_pct:<10.1f} {val_pct:<10.1f}")
        
        # Add test set distribution if available
        if test_loader:
            test_labels = []
            for _, labels in test_loader:
                test_labels.extend(labels.cpu().numpy())
            test_labels = np.array(test_labels)
            
            logger.info(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Train%':<10} {'Val%':<10} {'Test%':<10}")
            logger.info("-" * 80)
            
            for class_id in unique_classes:
                train_count = np.sum(train_labels == class_id)
                val_count = np.sum(val_labels == class_id)
                test_count = np.sum(test_labels == class_id)
                train_pct = (train_count / len(train_labels)) * 100
                val_pct = (val_count / len(val_labels)) * 100
                test_pct = (test_count / len(test_labels)) * 100
                
                logger.info(f"{class_id:<15} {train_count:<10} {val_count:<10} {test_count:<10} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:<10.1f}")
        
        logger.info("="*60)
    
    # Create model
    num_sensors = len(config['preprocessing']['sensors_order'])
    time_steps = config['preprocessing']['time_steps_per_window']
    num_classes = len(config['dataset']['labels'])
    
    input_shape = (num_sensors, time_steps, 3)
    
    model = create_model(args.model, input_shape, num_classes, config).to(device)
    
    # Create metrics objects
    train_metrics = HARMetrics(num_classes, config['dataset']['labels'], device)
    val_metrics = HARMetrics(num_classes, config['dataset']['labels'], device)
    test_metrics = HARMetrics(num_classes, config['dataset']['labels'], device) if test_loader else None
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize Weights & Biases if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config['logging']['project'],
                entity=config['logging']['entity'],
                name=run_name,
                config=config
            )
        except ImportError:
            logger.warning("wandb not available, falling back to offline logging")
    
    # Log class distributions to W&B if enabled
    if args.wandb:
        try:
            import wandb
            class_dist = {}
            for class_id in unique_classes:
                train_count = np.sum(train_labels == class_id)
                val_count = np.sum(val_labels == class_id)
                train_pct = (train_count / len(train_labels)) * 100
                val_pct = (val_count / len(val_labels)) * 100
                
                class_dist[f'train_class_{class_id}_count'] = train_count
                class_dist[f'train_class_{class_id}_pct'] = train_pct
                class_dist[f'val_class_{class_id}_count'] = val_count
                class_dist[f'val_class_{class_id}_pct'] = val_pct
                
                if test_loader:
                    test_count = np.sum(test_labels == class_id)
                    test_pct = (test_count / len(test_labels)) * 100
                    class_dist[f'test_class_{class_id}_count'] = test_count
                    class_dist[f'test_class_{class_id}_pct'] = test_pct
            
            wandb.log(class_dist)
        except:
            pass
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    min_delta = args.min_delta if args.min_delta != 1e-6 else config['training']['early_stopping'].get('min_delta', 1e-6)
    
    epoch_pbar = tqdm(range(config['training']['epochs']), desc=f"Training {run_name}", ncols=120)
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc, train_f1, train_preds, train_targets, train_probs = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets, val_probs = validate_epoch(
            model, val_loader, criterion, device, val_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Val F1': f'{val_f1:.4f}',
            'LR': f'{current_lr:.6f}'
        })
        
        # Log to wandb
        if args.wandb:
            try:
                # Get comprehensive metrics
                train_metrics_dict = train_metrics.compute()
                val_metrics_dict = val_metrics.compute()
                
                # Basic metrics
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    
                    # Training metrics
                    'train_accuracy': train_metrics_dict['accuracy'],
                    'train_f1_macro': train_metrics_dict['f1_macro'],
                    'train_f1_micro': train_metrics_dict['f1_micro'],
                    'train_f1_weighted': train_metrics_dict['f1_weighted'],
                    'train_precision_macro': train_metrics_dict['precision_macro'],
                    'train_recall_macro': train_metrics_dict['recall_macro'],
                    'train_specificity': train_metrics_dict['specificity'],
                    'train_cohen_kappa': train_metrics_dict['cohen_kappa'],
                    'train_matthews_corr': train_metrics_dict['matthews_corr'],
                    
                    # Validation metrics
                    'val_accuracy': val_metrics_dict['accuracy'],
                    'val_f1_macro': val_metrics_dict['f1_macro'],
                    'val_f1_micro': val_metrics_dict['f1_micro'],
                    'val_f1_weighted': val_metrics_dict['f1_weighted'],
                    'val_precision_macro': val_metrics_dict['precision_macro'],
                    'val_recall_macro': val_metrics_dict['recall_macro'],
                    'val_specificity': val_metrics_dict['specificity'],
                    'val_cohen_kappa': val_metrics_dict['cohen_kappa'],
                    'val_matthews_corr': val_metrics_dict['matthews_corr'],
                }
                
                # Add AUROC/AUPR if available
                if val_metrics_dict['auroc_macro'] is not None:
                    log_dict.update({
                        'val_auroc_macro': val_metrics_dict['auroc_macro'],
                        'val_aupr_macro': val_metrics_dict['aupr_macro'],
                        'val_calibration_error': val_metrics_dict['calibration_error'],
                    })
                
                wandb.log(log_dict)
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")
        
        # Early stopping with epsilon-based improvement check
        improvement = best_val_loss - val_loss
        if improvement > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(model, optimizer, epoch, best_val_loss, config, f"checkpoints/best_{run_name}.pt")
            
            # Plot confusion matrix
            class_names = [TARGET_ACTIVITIES[dataset.idx_to_activity[i]] for i in range(len(dataset.idx_to_activity))]
            plot_confusion_matrix(val_targets, val_preds, class_names, f"checkpoints/confusion_matrix_{run_name}.png")
            
            # Calculate and plot AUROC and AUPR
            auroc_aupr_results = calculate_auroc_aupr(val_targets, val_probs, class_names)
            plot_roc_curves(val_targets, val_probs, class_names, f"checkpoints/roc_curves_{run_name}.png")
            plot_pr_curves(val_targets, val_probs, class_names, f"checkpoints/pr_curves_{run_name}.png")
            
            # Calculate and plot calibration metrics
            ece = calculate_ece(val_targets, val_probs)
            plot_calibration_curves(val_targets, val_probs, class_names, f"checkpoints/calibration_curves_{run_name}.png")
            plot_reliability_diagram(val_targets, val_probs, class_names, f"checkpoints/reliability_diagram_{run_name}.png")
            
            # Log AUROC, AUPR, and calibration metrics to W&B
            if args.wandb:
                try:
                    wandb.log({
                        'macro_auroc': auroc_aupr_results['macro_auroc'],
                        'macro_aupr': auroc_aupr_results['macro_aupr'],
                        'ece': ece
                    })
                    # Log per-class metrics
                    for i, class_name in enumerate(class_names):
                        wandb.log({
                            f'auroc_{class_name}': auroc_aupr_results['auroc_per_class'][i],
                            f'aupr_{class_name}': auroc_aupr_results['aupr_per_class'][i]
                        })
                except Exception as e:
                    logger.warning(f"W&B AUROC/AUPR/ECE logging failed: {e}")
            
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping at epoch {epoch+1} (patience exceeded)")
                break
    
    # Final evaluation on train and validation sets to get predictions
    logger.info("Performing final evaluation...")
    final_train_loss, final_train_acc, final_train_f1, final_train_preds, final_train_targets, final_train_probs = train_epoch(
        model, train_loader, criterion, optimizer, device, train_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
    )
    
    final_val_loss, final_val_acc, final_val_f1, final_val_preds, final_val_targets, final_val_probs = validate_epoch(
        model, val_loader, criterion, device, val_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
    )
    
    # Evaluate on test set for holdout validation
    test_results = {}
    if args.protocol == "holdout" and test_loader is not None:
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1, test_preds, test_targets, test_probs = validate_epoch(
            model, test_loader, criterion, device, test_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Get comprehensive test metrics
        test_metrics_dict = test_metrics.compute()
        test_results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_metrics': test_metrics_dict,
            'test_predictions': test_preds,
            'test_targets': test_targets
        }
        logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Finish W&B logging
    if args.wandb:
        try:
            wandb.finish()
        except:
            pass
    
    # Return results for this fold
    return {
        'fold_id': fold_id,
        'run_name': run_name,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'final_train_f1': final_train_f1,
        'final_val_loss': final_val_loss,
        'final_val_acc': final_val_acc,
        'final_val_f1': final_val_f1,
        'train_predictions': final_train_preds,
        'train_targets': final_train_targets,
        'val_predictions': final_val_preds,
        'val_targets': final_val_targets,
        **test_results
    }

def aggregate_results(results, protocol):
    """Aggregate results from all folds/subjects."""
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION RESULTS SUMMARY ({protocol.upper()})")
    logger.info(f"{'='*80}")
    
    # Calculate statistics
    val_accs = [r['final_val_acc'] for r in results]
    val_f1s = [r['final_val_f1'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]
    
    # Print individual results with more detail
    logger.info(f"\nIndividual Results:")
    if protocol == "holdout":
        logger.info(f"{'ID':<4} {'Best Epoch':<10} {'Best Val Loss':<12} {'Final Val Loss':<13} {'Val Acc':<8} {'Val F1':<8} {'Test Acc':<8} {'Test F1':<8} {'Train Acc':<9} {'Train F1':<8}")
        logger.info(f"{'-'*4} {'-'*10} {'-'*12} {'-'*13} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
        
        for i, result in enumerate(results):
            test_acc = result.get('test_acc', 0.0)
            test_f1 = result.get('test_f1', 0.0)
            logger.info(f"{protocol.upper()} {i+1:2d} {result['best_epoch']+1:9d} {result['best_val_loss']:11.4f} "
                       f"{result['final_val_loss']:12.4f} {result['final_val_acc']:7.4f} {result['final_val_f1']:7.4f} "
                       f"{test_acc:7.4f} {test_f1:7.4f} {result['final_train_acc']:8.4f} {result['final_train_f1']:7.4f}")
    else:
        logger.info(f"{'ID':<4} {'Best Epoch':<10} {'Best Val Loss':<12} {'Final Val Loss':<13} {'Val Acc':<8} {'Val F1':<8} {'Train Acc':<9} {'Train F1':<8}")
        logger.info(f"{'-'*4} {'-'*10} {'-'*12} {'-'*13} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
        
        for i, result in enumerate(results):
            logger.info(f"{protocol.upper()} {i+1:2d} {result['best_epoch']+1:9d} {result['best_val_loss']:11.4f} "
                       f"{result['final_val_loss']:12.4f} {result['final_val_acc']:7.4f} {result['final_val_f1']:7.4f} "
                       f"{result['final_train_acc']:8.4f} {result['final_train_f1']:7.4f}")
    
    # Print aggregated statistics
    logger.info(f"\nAggregated Statistics:")
    logger.info(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    logger.info(f"Validation F1:       {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    logger.info(f"Validation Loss:     {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    
    # Add test set statistics for holdout
    if protocol == "holdout":
        test_accs = [r.get('test_acc', 0.0) for r in results if 'test_acc' in r]
        test_f1s = [r.get('test_f1', 0.0) for r in results if 'test_f1' in r]
        test_losses = [r.get('test_loss', 0.0) for r in results if 'test_loss' in r]
        
        if test_accs:
            logger.info(f"Test Accuracy:       {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
            logger.info(f"Test F1:             {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
            logger.info(f"Test Loss:           {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    results_file = f"checkpoints/cv_results_{protocol}.csv"
    df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save summary statistics
    summary = {
        'protocol': protocol,
        'num_folds': len(results),
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'val_f1_mean': np.mean(val_f1s),
        'val_f1_std': np.std(val_f1s),
        'val_loss_mean': np.mean(val_losses),
        'val_loss_std': np.std(val_losses)
    }
    
    # Add test set statistics for holdout
    if protocol == "holdout":
        test_accs = [r.get('test_acc', 0.0) for r in results if 'test_acc' in r]
        test_f1s = [r.get('test_f1', 0.0) for r in results if 'test_f1' in r]
        test_losses = [r.get('test_loss', 0.0) for r in results if 'test_loss' in r]
        
        if test_accs:
            summary.update({
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'test_f1_mean': np.mean(test_f1s),
                'test_f1_std': np.std(test_f1s),
                'test_loss_mean': np.mean(test_losses),
                'test_loss_std': np.std(test_losses)
            })
    
    summary_file = f"checkpoints/cv_summary_{protocol}.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Train color-coded HAR model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--model", type=str, default="InertialColorCNN", 
                       choices=["InertialColorCNN", "ECGColorCNN", "MultiModalColorCNN"],
                       help="Model architecture to use")
    parser.add_argument("--protocol", type=str, choices=["kfold", "loso", "holdout"], required=True, help="Cross-validation protocol")
    parser.add_argument("--fold", type=int, help="Fold number for k-fold (0-indexed)")
    parser.add_argument("--subject", type=int, help="Subject number for LOSO")
    parser.add_argument("--run_all", action="store_true", help="Run all folds/subjects iteratively and aggregate results")
    parser.add_argument("--holdout_runs", type=int, default=5, help="Number of holdout runs to perform (default: 5)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio for holdout (default: 0.2)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio for holdout (default: 0.1)")
    parser.add_argument("--min_delta", type=float, default=1e-6, help="Minimum change in loss to qualify as improvement (default: 1e-6)")
    parser.add_argument("--keep_full_height", action="store_true", help="Keep full height images")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--ignore_class0", action="store_true", help="Ignore class 0 samples (null class)")
    parser.add_argument("--hide_batch_progress", action="store_true", help="Hide batch-level progress bars during training")
    parser.add_argument("--hide_class_distribution", action="store_true", help="Hide class distribution logging")
    parser.add_argument("--tmux", action="store_true", help="Print tmux commands")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if we should run iterative cross-validation
    if args.run_all:
        if args.protocol == "holdout" and args.holdout_runs <= 1:
            logger.error("--run_all with holdout requires --holdout_runs > 1")
            return 1
        
        # Validate arguments for iterative runs
        if args.fold is not None or args.subject is not None:
            logger.warning("Ignoring --fold and --subject arguments when using --run_all")
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set random seed
        set_seed(config['training']['seed'])
        
        # Load dataset
        shard_files = get_shard_files(config['dataset']['root'])
        if not shard_files:
            logger.error(f"No shard files found in {config['dataset']['root']}")
            return 1
        
        dataset = ColorCodedHARDataset(
            shard_files=shard_files,
            sensors_order=config['preprocessing']['sensors_order'],
            target_activities=TARGET_ACTIVITIES,
            time_steps_per_window=config['preprocessing']['time_steps_per_window'],
            sensor_band_height_px=config['preprocessing']['sensor_band_height_px'],
            input_range=config['preprocessing']['input_range'],
            normalize_mode=config['preprocessing']['color_transform']['normalize_mode'],
            height_compress=config['preprocessing']['height_compress']['enabled'],
            keep_full_height=args.keep_full_height,
            ignore_class0=args.ignore_class0
        )
        
        # Run iterative cross-validation
        results = run_iterative_cv(args, config, dataset, device)
        logger.info("Iterative cross-validation completed!")
        return 0
    
    # Print tmux commands if requested
    if args.tmux:
        cmd = f"python scripts/train_inertial_color_coded.py --config {args.config} --protocol {args.protocol}"
        if args.run_all:
            cmd += " --run_all"
        if args.fold is not None:
            cmd += f" --fold {args.fold}"
        if args.subject is not None:
            cmd += f" --subject {args.subject}"
        if args.protocol == "holdout":
            cmd += f" --test_ratio {args.test_ratio} --val_ratio {args.val_ratio} --holdout_runs {args.holdout_runs}"
        if args.keep_full_height:
            cmd += " --keep_full_height"
        if args.wandb:
            cmd += " --wandb"
        if args.ignore_class0:
            cmd += " --ignore_class0"
        
        print(f"tmux new-session -d -s har_colorcoded_{args.protocol}_{args.fold or args.subject}")
        print(f"tmux send-keys -t har_colorcoded_{args.protocol}_{args.fold or args.subject} '{cmd}' Enter")
        return
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    shard_files = get_shard_files(config['dataset']['root'])
    logger.info(f"Found {len(shard_files)} shard files")
    
    # Create dataset
    dataset = ColorCodedHARDataset(
        shard_files=shard_files,
        sensors_order=config['preprocessing']['sensors_order'],
        target_activities=TARGET_ACTIVITIES,
        time_steps_per_window=config['preprocessing']['time_steps_per_window'],
        sensor_band_height_px=config['preprocessing']['sensor_band_height_px'],
        input_range=config['preprocessing']['input_range'],
        normalize_mode=config['preprocessing']['color_transform']['normalize_mode'],
        height_compress=config['preprocessing']['height_compress']['enabled'],
        keep_full_height=args.keep_full_height
    )
    
    # Create splits
    if args.protocol == "kfold":
        splits = create_kfold_splits(dataset, config['dataset']['splits']['k'], config['dataset']['splits']['seed'])
        if args.fold is None and not args.run_all:
            raise ValueError("--fold is required for kfold protocol. Use --run_all to run all folds iteratively.")
        train_idx, val_idx = splits[args.fold]
        run_name = f"kfold_fold_{args.fold}"
    elif args.protocol == "loso":
        splits = create_loso_splits(dataset)
        if args.subject is None and not args.run_all:
            raise ValueError("--subject is required for loso protocol. Use --run_all to run all subjects iteratively.")
        train_idx, val_idx = splits[args.subject]
        run_name = f"loso_subject_{args.subject}"
    else:  # holdout
        train_idx, val_idx, test_idx = create_holdout_splits(
            dataset, 
            test_ratio=args.test_ratio, 
            val_ratio=args.val_ratio, 
            seed=config['training']['seed']
        )
        run_name = f"holdout_test{args.test_ratio}_val{args.val_ratio}"
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create test loader for holdout validation
    test_loader = None
    if args.protocol == "holdout":
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    if args.protocol == "holdout":
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    else:
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Log class distributions if not hidden
    if not args.hide_class_distribution:
        logger.info("\n" + "="*60)
        logger.info("CLASS DISTRIBUTIONS")
        logger.info("="*60)
    
    # Get class distributions for train set
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.cpu().numpy())
    train_labels = np.array(train_labels)
    
    # Get class distributions for val set
    val_labels = []
    for _, labels in val_loader:
        val_labels.extend(labels.cpu().numpy())
    val_labels = np.array(val_labels)
    
    # Get unique classes and their counts
    unique_classes = sorted(np.unique(np.concatenate([train_labels, val_labels])))
    
    logger.info(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Train%':<10} {'Val%':<10}")
    logger.info("-" * 60)
    
    for class_id in unique_classes:
        train_count = np.sum(train_labels == class_id)
        val_count = np.sum(val_labels == class_id)
        train_pct = (train_count / len(train_labels)) * 100
        val_pct = (val_count / len(val_labels)) * 100
        
        logger.info(f"{class_id:<15} {train_count:<10} {val_count:<10} {train_pct:<10.1f} {val_pct:<10.1f}")
    
    # Add test set distribution if available
    if test_loader:
        test_labels = []
        for _, labels in test_loader:
            test_labels.extend(labels.cpu().numpy())
        test_labels = np.array(test_labels)
        
        logger.info(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Train%':<10} {'Val%':<10} {'Test%':<10}")
        logger.info("-" * 80)
        
        for class_id in unique_classes:
            train_count = np.sum(train_labels == class_id)
            val_count = np.sum(val_labels == class_id)
            test_count = np.sum(test_labels == class_id)
            train_pct = (train_count / len(train_labels)) * 100
            val_pct = (val_count / len(val_labels)) * 100
            test_pct = (test_count / len(test_labels)) * 100
            
            logger.info(f"{class_id:<15} {train_count:<10} {val_count:<10} {test_count:<10} {train_pct:<10.1f} {val_pct:<10.1f} {test_pct:<10.1f}")
    
    logger.info("="*60)
    
    # Determine input dimensions
    sample_data, _ = next(iter(train_loader))
    input_height = sample_data.shape[2]  # H dimension
    input_width = sample_data.shape[3]   # W dimension
    
    logger.info(f"Input shape: {sample_data.shape}")
    
    # Create model using InertialColorCNN
    num_sensors = len(config['preprocessing']['sensors_order'])
    time_steps = config['preprocessing']['time_steps_per_window']
    num_classes = len(config['dataset']['labels'])
    
    # Input shape: (num_sensors, time_steps, channels)
    input_shape = (num_sensors, time_steps, 3)
    
    model = create_model(args.model, input_shape, num_classes, config).to(device)
    
    # Create metrics objects
    train_metrics = HARMetrics(num_classes, config['dataset']['labels'], device)
    val_metrics = HARMetrics(num_classes, config['dataset']['labels'], device)
    test_metrics = HARMetrics(num_classes, config['dataset']['labels'], device) if test_loader else None
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Early stopping: patience={config['training']['early_stopping']['patience']}, min_delta={min_delta:.2e}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize Weights & Biases if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config['logging']['project'],
                entity=config['logging']['entity'],
                name=run_name,
                config=config
            )
        except ImportError:
            logger.warning("wandb not available, falling back to offline logging")
    
    # Log class distributions to W&B if enabled
    if args.wandb:
        try:
            import wandb
            class_dist = {}
            for class_id in unique_classes:
                train_count = np.sum(train_labels == class_id)
                val_count = np.sum(val_labels == class_id)
                train_pct = (train_count / len(train_labels)) * 100
                val_pct = (val_count / len(val_labels)) * 100
                
                class_dist[f'train_class_{class_id}_count'] = train_count
                class_dist[f'train_class_{class_id}_pct'] = train_pct
                class_dist[f'val_class_{class_id}_count'] = val_count
                class_dist[f'val_class_{class_id}_pct'] = val_pct
                
                if test_loader:
                    test_count = np.sum(test_labels == class_id)
                    test_pct = (test_count / len(test_labels)) * 100
                    class_dist[f'test_class_{class_id}_count'] = test_count
                    class_dist[f'test_class_{class_id}_pct'] = test_pct
            
            wandb.log(class_dist)
        except:
            pass
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    min_delta = args.min_delta if args.min_delta != 1e-6 else config['training']['early_stopping'].get('min_delta', 1e-6)
    
    # Main training progress bar
    epoch_pbar = tqdm(range(config['training']['epochs']), desc="Training Progress", ncols=120)
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc, train_f1, train_preds, train_targets, train_probs = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets, val_probs = validate_epoch(
            model, val_loader, criterion, device, val_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update main progress bar
        current_lr = optimizer.param_groups[0]['lr']
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Val F1': f'{val_f1:.4f}',
            'LR': f'{current_lr:.6f}'
        })
        
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Log to wandb
        if args.wandb:
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'learning_rate': current_lr
                })
            except:
                pass
        
        # Early stopping with epsilon-based improvement check
        improvement = best_val_loss - val_loss
        if improvement > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(model, optimizer, epoch, best_val_loss, config, f"checkpoints/best_{run_name}.pt")
            
            # Plot confusion matrix
            class_names = [TARGET_ACTIVITIES[dataset.idx_to_activity[i]] for i in range(len(dataset.idx_to_activity))]
            plot_confusion_matrix(val_targets, val_preds, class_names, f"checkpoints/confusion_matrix_{run_name}.png")
            
            # Calculate and plot AUROC and AUPR
            auroc_aupr_results = calculate_auroc_aupr(val_targets, val_probs, class_names)
            plot_roc_curves(val_targets, val_probs, class_names, f"checkpoints/roc_curves_{run_name}.png")
            plot_pr_curves(val_targets, val_probs, class_names, f"checkpoints/pr_curves_{run_name}.png")
            
            # Calculate and plot calibration metrics
            ece = calculate_ece(val_targets, val_probs)
            plot_calibration_curves(val_targets, val_probs, class_names, f"checkpoints/calibration_curves_{run_name}.png")
            plot_reliability_diagram(val_targets, val_probs, class_names, f"checkpoints/reliability_diagram_{run_name}.png")
            
            # Log AUROC, AUPR, and calibration metrics to W&B
            if args.wandb:
                try:
                    wandb.log({
                        'macro_auroc': auroc_aupr_results['macro_auroc'],
                        'macro_aupr': auroc_aupr_results['macro_aupr'],
                        'ece': ece
                    })
                    # Log per-class metrics
                    for i, class_name in enumerate(class_names):
                        wandb.log({
                            f'auroc_{class_name}': auroc_aupr_results['auroc_per_class'][i],
                            f'aupr_{class_name}': auroc_aupr_results['aupr_per_class'][i]
                        })
                except Exception as e:
                    logger.warning(f"W&B AUROC/AUPR/ECE logging failed: {e}")
            
            logger.info(f"Improvement: {improvement:.6f} (>{min_delta:.6f}) - New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if improvement <= min_delta and improvement >= 0:
                logger.info(f"Minimal improvement: {improvement:.6f} (≤{min_delta:.6f})")
            elif improvement < 0:
                logger.info(f"Validation loss increased: {abs(improvement):.6f}")
            
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping at epoch {epoch+1} (patience exceeded)")
                break
    
    # Save final model
    save_checkpoint(model, optimizer, epoch, best_val_loss, config, f"checkpoints/last_{run_name}.pt")
    
    # Evaluate on test set for holdout validation
    if args.protocol == "holdout" and test_loader is not None:
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1, test_preds, test_targets, test_probs = validate_epoch(
            model, test_loader, criterion, device, test_metrics, epoch, config['training']['epochs'], args.hide_batch_progress
        )
        
        # Get comprehensive test metrics
        test_metrics_dict = test_metrics.compute()
        
        logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Plot test confusion matrix
        class_names = [TARGET_ACTIVITIES[dataset.idx_to_activity[i]] for i in range(len(dataset.idx_to_activity))]
        plot_confusion_matrix(test_targets, test_preds, class_names, f"checkpoints/confusion_matrix_test_{run_name}.png")
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    if args.wandb:
        try:
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()
