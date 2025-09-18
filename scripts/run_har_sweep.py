#!/usr/bin/env python3
"""
HAR Sweep Runner
Runs Weights & Biases sweeps for HAR experiments using YAML configuration files.
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

import wandb
import torch
from scripts.run_har_experiment import HARExperimentRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HARSweepRunner:
    """Main class for running HAR sweeps from config files."""
    
    def __init__(self, config_path: str, sweep_config_path: str):
        """Initialize the sweep runner with config files."""
        self.config_path = Path(config_path)
        self.sweep_config_path = Path(sweep_config_path)
        
        # Load configurations
        self.config = self.load_config(self.config_path)
        self.sweep_config = self.load_config(self.sweep_config_path)
        
        # Setup sweep
        self.sweep_id = None
        self.project = self.config['har_experiment']['wandb']['project']
        self.entity = self.config['har_experiment']['wandb'].get('entity')
        
        logger.info(f"Initialized HAR sweep runner")
        logger.info(f"Project: {self.project}")
        logger.info(f"Entity: {self.entity}")
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def create_sweep(self) -> str:
        """Create a new Weights & Biases sweep."""
        try:
            # Initialize wandb
            wandb.login()
            
            # Create sweep
            sweep_id = wandb.sweep(
                self.sweep_config,
                project=self.project,
                entity=self.entity
            )
            
            self.sweep_id = sweep_id
            logger.info(f"Created sweep with ID: {sweep_id}")
            logger.info(f"Project: {self.project}")
            logger.info(f"Entity: {self.entity}")
            
            return sweep_id
            
        except Exception as e:
            logger.error(f"Failed to create sweep: {e}")
            raise
    
    def run_sweep(self, count: int = 10):
        """Run the sweep with specified number of runs."""
        if not self.sweep_id:
            logger.error("No sweep ID available. Create a sweep first.")
            return
        
        try:
            # Run sweep
            wandb.agent(
                self.sweep_id,
                function=self.sweep_function,
                project=self.project,
                entity=self.entity,
                count=count
            )
            
            logger.info(f"Completed sweep with {count} runs")
            
        except Exception as e:
            logger.error(f"Failed to run sweep: {e}")
            raise
    
    def sweep_function(self):
        """Function to be called for each sweep run."""
        # Initialize wandb run
        run = wandb.init()
        
        try:
            # Get hyperparameters from wandb
            config_updates = dict(wandb.config)
            
            # Update the base config with sweep parameters
            updated_config = self.update_config_with_sweep_params(config_updates)
            
            # Create temporary config file for this run
            temp_config_path = self.create_temp_config(updated_config, run.id)
            
            # Run the experiment
            runner = HARExperimentRunner(str(temp_config_path))
            runner.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Setup components
            runner.setup_dataset()
            runner.setup_model()
            runner.setup_optimizer()
            runner.setup_criterion()
            
            # Run training with wandb logging
            metrics = self.run_training_with_wandb(runner, run)
            
            # Run evaluation
            eval_metrics = runner.evaluate()
            
            # Log final metrics
            wandb.log({
                "final/train_accuracy": metrics.get('train_accuracy', 0),
                "final/val_accuracy": metrics.get('val_accuracy', 0),
                "final/test_accuracy": eval_metrics.get('accuracy', 0),
                "final/test_f1": eval_metrics.get('f1', 0),
                "final/test_ece": eval_metrics.get('ece', 0)
            })
            
            # Clean up temporary config
            temp_config_path.unlink()
            
        except Exception as e:
            logger.error(f"Sweep run failed: {e}")
            wandb.log({"error": str(e)})
            raise
        finally:
            wandb.finish()
    
    def update_config_with_sweep_params(self, sweep_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update base config with sweep parameters."""
        import copy
        updated_config = copy.deepcopy(self.config)
        
        # Update model parameters
        if 'model' in sweep_params:
            updated_config['har_experiment']['model']['name'] = sweep_params['model']
        
        if 'dropout' in sweep_params:
            updated_config['har_experiment']['model']['dropout'] = sweep_params['dropout']
        
        # Update optimizer parameters
        if 'optimizer' in sweep_params:
            updated_config['har_experiment']['optimizer']['type'] = sweep_params['optimizer']
        
        if 'lr' in sweep_params:
            updated_config['har_experiment']['optimizer']['params']['lr'] = sweep_params['lr']
        
        if 'weight_decay' in sweep_params:
            updated_config['har_experiment']['optimizer']['params']['weight_decay'] = sweep_params['weight_decay']
        
        # Update training parameters
        if 'batch_size' in sweep_params:
            updated_config['har_experiment']['dataloader']['train']['batch_size'] = sweep_params['batch_size']
            updated_config['har_experiment']['dataloader']['val']['batch_size'] = sweep_params['batch_size']
            updated_config['har_experiment']['dataloader']['test']['batch_size'] = sweep_params['batch_size']
        
        if 'epochs' in sweep_params:
            updated_config['har_experiment']['epochs'] = sweep_params['epochs']
        
        return updated_config
    
    def create_temp_config(self, config: Dict[str, Any], run_id: str) -> Path:
        """Create temporary config file for a sweep run."""
        temp_dir = Path("temp_sweep_configs")
        temp_dir.mkdir(exist_ok=True)
        
        temp_config_path = temp_dir / f"temp_config_{run_id}.yaml"
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return temp_config_path
    
    def run_training_with_wandb(self, runner, wandb_run) -> Dict[str, float]:
        """Run training with Weights & Biases logging."""
        har_config = runner.config['har_experiment']
        epochs = har_config['epochs']
        early_stop = har_config.get('early_stop', None)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = runner.train_epoch(epoch)
            
            # Validation
            val_metrics = runner.validate_epoch(epoch)
            
            # Update scheduler
            if runner.scheduler:
                if isinstance(runner.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    runner.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    runner.scheduler.step()
            
            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy']
            }
            
            if val_metrics:
                log_dict.update({
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy']
                })
            
            wandb.log(log_dict)
            
            # Early stopping
            if early_stop and val_metrics:
                val_acc = val_metrics['accuracy']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        return {
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics.get('accuracy', 0) if val_metrics else 0
        }


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run HAR sweeps from config files')
    parser.add_argument('config', help='Path to base HAR experiment YAML configuration file')
    parser.add_argument('sweep_config', help='Path to sweep configuration YAML file')
    parser.add_argument('--action', choices=['create', 'run', 'both'], default='both',
                       help='Action to perform: create sweep, run sweep, or both')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of sweep runs to execute')
    parser.add_argument('--sweep_id', help='Existing sweep ID to run (if not creating new)')
    parser.add_argument('--project', help='WandB project name (overrides config)')
    parser.add_argument('--entity', help='WandB entity name (overrides config)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize sweep runner
    runner = HARSweepRunner(args.config, args.sweep_config)
    
    # Override project/entity if provided
    if args.project:
        runner.project = args.project
    if args.entity:
        runner.entity = args.entity
    
    try:
        if args.action in ['create', 'both']:
            sweep_id = runner.create_sweep()
            print(f"Created sweep with ID: {sweep_id}")
            
            if args.action == 'create':
                print(f"To run this sweep, use:")
                print(f"python scripts/run_har_sweep.py {args.config} {args.sweep_config} --action run --sweep_id {sweep_id}")
        
        if args.action in ['run', 'both']:
            if args.sweep_id:
                runner.sweep_id = args.sweep_id
            elif not runner.sweep_id:
                logger.error("No sweep ID available. Use --action create first or provide --sweep_id")
                sys.exit(1)
            
            runner.run_sweep(args.count)
            print(f"Completed sweep with {args.count} runs")
    
    except Exception as e:
        logger.error(f"Sweep failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
