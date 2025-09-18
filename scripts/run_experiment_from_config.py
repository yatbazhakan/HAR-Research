#!/usr/bin/env python3
"""
Universal Experiment Runner
Runs experiments directly from YAML configuration files.
Supports both single experiments and Weights & Biases sweeps.
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
import logging
import subprocess
from typing import Dict, Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentConfigRunner:
    """Main class for running experiments from config files."""
    
    def __init__(self, config_path: str):
        """Initialize the experiment runner with a config file."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        logger.info(f"Initialized experiment runner with config: {self.config_path}")
    
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
    
    def determine_experiment_type(self) -> str:
        """Determine the type of experiment from the config."""
        if 'har_experiment' in self.config:
            return 'har'
        elif 'introspection' in self.config:
            return 'introspection'
        elif 'detection' in self.config:
            return 'detection'
        elif 'extraction' in self.config:
            return 'extraction'
        else:
            raise ValueError("Unknown experiment type in configuration")
    
    def run_har_experiment(self, is_sweep: bool = False, sweep_config_path: Optional[str] = None) -> int:
        """Run HAR experiment or sweep."""
        if is_sweep:
            if not sweep_config_path:
                raise ValueError("Sweep config path required for sweep runs")
            
            # Run HAR sweep
            cmd = [
                sys.executable, 
                str(REPO_ROOT / "scripts" / "run_har_sweep.py"),
                str(self.config_path),
                sweep_config_path,
                "--action", "both"
            ]
        else:
            # Run single HAR experiment
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_har_experiment.py"),
                str(self.config_path)
            ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return result.returncode
    
    def run_introspection_experiment(self, is_sweep: bool = False, sweep_config_path: Optional[str] = None) -> int:
        """Run introspection experiment or sweep."""
        # This would call your existing introspection scripts
        # For now, we'll just log that it's not implemented
        logger.warning("Introspection experiments not yet implemented in config runner")
        return 1
    
    def run_detection_experiment(self, is_sweep: bool = False, sweep_config_path: Optional[str] = None) -> int:
        """Run detection experiment or sweep."""
        # This would call your existing detection scripts
        logger.warning("Detection experiments not yet implemented in config runner")
        return 1
    
    def run_extraction_experiment(self, is_sweep: bool = False, sweep_config_path: Optional[str] = None) -> int:
        """Run extraction experiment or sweep."""
        # This would call your existing extraction scripts
        logger.warning("Extraction experiments not yet implemented in config runner")
        return 1
    
    def run(self, is_sweep: bool = False, sweep_config_path: Optional[str] = None) -> int:
        """Run the experiment based on configuration type."""
        experiment_type = self.determine_experiment_type()
        logger.info(f"Running {experiment_type} experiment (sweep: {is_sweep})")
        
        if experiment_type == 'har':
            return self.run_har_experiment(is_sweep, sweep_config_path)
        elif experiment_type == 'introspection':
            return self.run_introspection_experiment(is_sweep, sweep_config_path)
        elif experiment_type == 'detection':
            return self.run_detection_experiment(is_sweep, sweep_config_path)
        elif experiment_type == 'extraction':
            return self.run_extraction_experiment(is_sweep, sweep_config_path)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run experiments from YAML configuration files')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--sweep', action='store_true', help='Run as a Weights & Biases sweep')
    parser.add_argument('--sweep-config', help='Path to sweep configuration file (required for sweeps)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.sweep and not args.sweep_config:
        logger.error("Sweep config path required when running sweeps")
        sys.exit(1)
    
    if args.sweep_config and not Path(args.sweep_config).exists():
        logger.error(f"Sweep config file not found: {args.sweep_config}")
        sys.exit(1)
    
    # Run experiment
    try:
        runner = ExperimentConfigRunner(args.config)
        return_code = runner.run(args.sweep, args.sweep_config)
        
        if return_code == 0:
            logger.info("Experiment completed successfully")
        else:
            logger.error(f"Experiment failed with return code {return_code}")
        
        sys.exit(return_code)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
