#!/usr/bin/env python3
"""
HAR Experiment Runner
Runs HAR experiments directly from YAML configuration files.
Supports both single experiments and Weights & Biases sweeps.
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
import logging
import time
import json
from typing import Dict, Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from har.train.metrics import accuracy, macro_f1, expected_calibration_error
from har.datasets.ucihar import UCIHARDataset
from har.datasets.pamap2 import PAMAP2Dataset
from har.datasets.mhealth import MHealthDataset
from har.models.cnn_tcn import CNN_TCN
from har.models.cnn_bilstm import CNN_BiLSTM
from har.transforms.normalize import NormStats
from har.transforms.stats_io import load_stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HARExperimentRunner:
    """Main class for running HAR experiments from config files."""
    
    def __init__(self, config_path: str):
        """Initialize the experiment runner with a config file."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Setup experiment directory
        self.experiment_name = self.config['har_experiment']['experiment_name']
        self.experiment_dir = Path(self.config.get('experiment_path', './')) / 'experiments' / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.transform = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"Initialized HAR experiment: {self.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_dataset(self):
        """Setup dataset and data loaders."""
        har_config = self.config['har_experiment']
        dataset_config = har_config['dataset']['config']
        dataloader_config = har_config['dataloader']
        
        # Load normalization stats
        stats_file = dataset_config.get('stats_file')
        if stats_file and Path(stats_file).exists():
            self.stats = load_stats(stats_file)
            logger.info(f"Loaded normalization stats from: {stats_file}")
        else:
            logger.warning("No normalization stats found, using raw data")
            self.stats = None
        
        # Create dataset
        dataset_name = har_config['dataset']['name'].lower()
        shards_glob = dataset_config['shards_glob']
        
        if dataset_name == 'uci_har':
            self.dataset = UCIHARDataset(shards_glob, transform=self.stats)
        elif dataset_name == 'pamap2':
            self.dataset = PAMAP2Dataset(shards_glob, transform=self.stats)
        elif dataset_name == 'mhealth':
            self.dataset = MHealthDataset(shards_glob, transform=self.stats)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        logger.info(f"Created {dataset_name} dataset with {len(self.dataset)} samples")
        
        # Setup cross-validation splits
        self.setup_cross_validation()
        
        # Create data loaders
        self.setup_dataloaders(dataloader_config)
    
    def setup_cross_validation(self):
        """Setup cross-validation splits based on configuration."""
        cv_config = self.config['har_experiment']['cross_validation']
        cv_type = cv_config['type']
        
        if cv_type == 'holdout':
            test_ratio = cv_config['params']['test_ratio']
            val_ratio = cv_config['params']['val_ratio']
            random_seed = cv_config['params'].get('random_seed', 42)
            
            # Calculate split sizes
            total_size = len(self.dataset)
            test_size = int(total_size * test_ratio)
            val_size = int(total_size * val_ratio)
            train_size = total_size - test_size - val_size
            
            # Split dataset
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed)
            )
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            
            logger.info(f"Holdout split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
            
        elif cv_type == 'fold_json':
            fold_file = cv_config['params']['fold_file']
            self.load_fold_splits(fold_file)
            
        else:
            raise NotImplementedError(f"Cross-validation type {cv_type} not implemented yet")
    
    def load_fold_splits(self, fold_file: str):
        """Load pre-computed fold splits from JSON file."""
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
        
        train_indices = fold_data['train']
        val_indices = fold_data.get('val', [])
        test_indices = fold_data['test']
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices) if val_indices else None
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        logger.info(f"Loaded fold splits: train={len(self.train_dataset)}, val={len(self.val_dataset) if self.val_dataset else 0}, test={len(self.test_dataset)}")
    
    def setup_dataloaders(self, dataloader_config: Dict[str, Any]):
        """Setup data loaders for training, validation, and testing."""
        # Common dataloader parameters
        common_params = {
            'num_workers': dataloader_config['all']['num_workers'],
            'pin_memory': dataloader_config['all']['pin_memory']
        }
        
        # Training loader
        train_params = {**common_params, **dataloader_config['train']}
        self.train_loader = DataLoader(self.train_dataset, **train_params)
        
        # Validation loader
        if self.val_dataset:
            val_params = {**common_params, **dataloader_config['val']}
            self.val_loader = DataLoader(self.val_dataset, **val_params)
        
        # Test loader
        test_params = {**common_params, **dataloader_config['test']}
        self.test_loader = DataLoader(self.test_dataset, **test_params)
        
        logger.info("Data loaders created successfully")
    
    def setup_model(self):
        """Setup the neural network model."""
        har_config = self.config['har_experiment']
        model_config = har_config['model']
        
        # Get dataset info
        dataset_config = har_config['dataset']['config']
        input_shape = dataset_config['input_shape']  # [channels, time_steps]
        num_classes = dataset_config['num_classes']
        in_channels = input_shape[0]  # Extract number of channels
        
        # Create model
        model_name = model_config['name']
        if model_name == 'cnn_tcn':
            self.model = CNN_TCN(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden=model_config.get('hidden', 64),
                num_blocks=model_config.get('num_blocks', 3)
            )
        elif model_name == 'cnn_bilstm':
            self.model = CNN_BiLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                cnn_hidden=model_config.get('cnn_hidden', 64),
                lstm_hidden=model_config.get('lstm_hidden', 64),
                lstm_layers=model_config.get('lstm_layers', 1)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        logger.info(f"Created {model_name} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        har_config = self.config['har_experiment']
        opt_config = har_config['optimizer']
        
        # Create optimizer
        opt_type = opt_config['type']
        opt_params = opt_config['params']
        
        if opt_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), **opt_params)
        elif opt_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **opt_params)
        elif opt_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), **opt_params)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        # Create scheduler
        sched_config = har_config['scheduler']
        sched_type = sched_config['type']
        sched_params = sched_config['params']
        
        if sched_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **sched_params
            )
        elif sched_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **sched_params
            )
        elif sched_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **sched_params
            )
        else:
            logger.warning(f"Unknown scheduler: {sched_type}, using no scheduler")
            self.scheduler = None
        
        logger.info(f"Created {opt_type} optimizer with {sched_type} scheduler")
    
    def setup_criterion(self):
        """Setup loss function."""
        har_config = self.config['har_experiment']
        criterion_config = har_config['criterion']
        
        criterion_type = criterion_config['type']
        criterion_params = criterion_config['params']
        
        if criterion_type == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss(**criterion_params)
        elif criterion_type == 'FocalLoss':
            # You would need to implement FocalLoss or import it
            raise NotImplementedError("FocalLoss not implemented yet")
        else:
            raise ValueError(f"Unknown criterion: {criterion_type}")
        
        logger.info(f"Created {criterion_type} loss function")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config['har_experiment']['method'].get('clip_grad'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config['har_experiment']['method']['clip_grad'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % self.config['har_experiment']['log_interval'] == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self):
        """Main training loop."""
        har_config = self.config['har_experiment']
        epochs = har_config['epochs']
        early_stop = har_config.get('early_stop', None)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                       f'Train Acc: {train_metrics["accuracy"]:.2f}%')
            if val_metrics:
                logger.info(f'Epoch {epoch}: Val Loss: {val_metrics["loss"]:.4f}, '
                           f'Val Acc: {val_metrics["accuracy"]:.2f}%')
            
            # Early stopping
            if early_stop and val_metrics:
                val_acc = val_metrics['accuracy']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    self.save_model(epoch, is_best=True)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Save checkpoint
            if epoch % har_config.get('save_interval', 10) == 0:
                self.save_model(epoch)
        
        logger.info("Training completed")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy(targets, predictions),
            'f1': macro_f1(targets, predictions),
            'ece': expected_calibration_error(probabilities, targets)
        }
        
        # Generate classification report
        class_names = self.config['har_experiment']['dataset']['config']['class_names']
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        
        # Save results
        self.save_evaluation_results(metrics, report, predictions, targets, probabilities)
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {metrics['f1']:.4f}")
        logger.info(f"Test ECE: {metrics['ece']:.4f}")
        
        return metrics
    
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        har_config = self.config['har_experiment']
        save_path = Path(har_config['save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if is_best:
            filename = f"{har_config['save_name']}_best.pth"
        else:
            filename = f"{har_config['save_name']}_epoch_{epoch}.pth"
        
        torch.save(checkpoint, save_path / filename)
        logger.info(f"Saved model checkpoint: {save_path / filename}")
    
    def save_evaluation_results(self, metrics: Dict[str, float], report: Dict, 
                              predictions: np.ndarray, targets: np.ndarray, 
                              probabilities: np.ndarray):
        """Save evaluation results and plots."""
        har_config = self.config['har_experiment']
        eval_config = har_config.get('evaluation', {})
        
        # Save metrics
        results_file = self.experiment_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed report
        report_file = self.experiment_dir / 'classification_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save predictions if requested
        if eval_config.get('save_predictions', False):
            predictions_file = self.experiment_dir / 'predictions.csv'
            import pandas as pd
            df = pd.DataFrame({
                'true_label': targets,
                'predicted_label': predictions,
                'confidence': probabilities.max(axis=1)
            })
            df.to_csv(predictions_file, index=False)
        
        # Generate plots
        if eval_config.get('plot_dir'):
            self.generate_plots(targets, predictions, probabilities)
    
    def generate_plots(self, targets: np.ndarray, predictions: np.ndarray, 
                      probabilities: np.ndarray):
        """Generate evaluation plots."""
        har_config = self.config['har_experiment']
        plot_dir = Path(har_config['evaluation']['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        class_names = har_config['dataset']['config']['class_names']
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.experiment_name}_confusion_matrix.png')
        plt.close()
        
        logger.info(f"Saved plots to {plot_dir}")
    
    def run(self):
        """Run the complete experiment."""
        logger.info("Starting HAR experiment")
        
        # Setup all components
        self.setup_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        
        # Run training
        self.train()
        
        # Run evaluation
        metrics = self.evaluate()
        
        logger.info("Experiment completed successfully")
        return metrics


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run HAR experiments from config files')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--device', help='Device to use (cuda/cpu)', default=None)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run experiment
    runner = HARExperimentRunner(args.config)
    
    if args.device:
        runner.device = torch.device(args.device)
    
    try:
        metrics = runner.run()
        print(f"Experiment completed successfully!")
        print(f"Final metrics: {metrics}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
