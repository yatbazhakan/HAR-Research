#!/usr/bin/env python3
"""
Wandb Sweep Runner for HAR Experiments
Runs hyperparameter sweeps using wandb.
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
import time

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def create_sweep_config(output_file, dataset="uci_har", method="bayes"):
    """Create a sweep configuration file"""
    
    # Dataset-specific configurations
    dataset_configs = {
        "uci_har": {
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-1
                },
                "batch_size": {
                    "values": [16, 32, 64, 128]
                },
                "epochs": {
                    "values": [50, 100, 150, 200]
                },
                "model": {
                    "values": ["cnn_tcn", "cnn_bilstm"]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 0.5
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2
                }
            }
        },
        "pamap2": {
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-1
                },
                "batch_size": {
                    "values": [32, 64, 128, 256]
                },
                "epochs": {
                    "values": [100, 200, 300, 400]
                },
                "model": {
                    "values": ["cnn_tcn", "cnn_bilstm"]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.2,
                    "max": 0.6
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2
                }
            }
        },
        "mhealth": {
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-1
                },
                "batch_size": {
                    "values": [16, 32, 64, 128]
                },
                "epochs": {
                    "values": [50, 100, 150, 200]
                },
                "model": {
                    "values": ["cnn_tcn", "cnn_bilstm"]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 0.5
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-2
                }
            }
        }
    }
    
    # Base configuration
    sweep_config = {
        "method": method,
        "metric": {
            "name": "val/accuracy",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,
            "eta": 2
        }
    }
    
    # Add dataset-specific parameters
    if dataset in dataset_configs:
        sweep_config["parameters"] = dataset_configs[dataset]["parameters"]
    else:
        # Default parameters
        sweep_config["parameters"] = dataset_configs["uci_har"]["parameters"]
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    print(f"Created sweep config: {output_path}")
    return output_path

def run_sweep_agent(sweep_id, project, count, config_file):
    """Run a single sweep agent iteration"""
    
    # Load the base configuration
    with open(config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Generate command
    cmd_parts = ["python", "scripts/train_baselines.py"]
    
    # Add required arguments
    required_args = ["shards_glob", "stats", "model", "epochs", "batch_size", "lr"]
    for arg in required_args:
        if arg in base_config:
            if arg == "shards_glob":
                # Escape glob patterns
                escaped_value = str(base_config[arg]).replace('*', '\\*')
                cmd_parts.extend([f"--{arg}", escaped_value])
            else:
                cmd_parts.extend([f"--{arg}", str(base_config[arg])])
    
    # Add wandb arguments
    cmd_parts.extend(["--wandb", "--wandb_project", project])
    
    # Add other arguments
    optional_args = ["cv", "fold_json", "val_split", "k_folds", "plot_dir", "class_names", "calibrate", "amp"]
    for arg in optional_args:
        if arg in base_config:
            cmd_parts.extend([f"--{arg}", str(base_config[arg])])
    
    command = " ".join(cmd_parts)
    print(f"Running: {command}")
    
    # Run the command
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Sweep run completed successfully")
        else:
            print(f"Sweep run failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
    except Exception as e:
        print(f"ERROR running sweep agent: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run wandb sweeps for HAR experiments")
    parser.add_argument("--action", choices=["create", "run"], required=True,
                       help="Action to perform: create sweep config or run sweep")
    parser.add_argument("--config", type=str, required=True,
                       help="Base experiment configuration YAML file")
    parser.add_argument("--sweep_config", type=str,
                       help="Sweep configuration YAML file (for create action)")
    parser.add_argument("--dataset", type=str, default="uci_har",
                       choices=["uci_har", "pamap2", "mhealth"],
                       help="Dataset name")
    parser.add_argument("--method", type=str, default="bayes",
                       choices=["bayes", "random", "grid"],
                       help="Sweep method")
    parser.add_argument("--project", type=str, default="har-sweeps",
                       help="Wandb project name")
    parser.add_argument("--sweep_id", type=str,
                       help="Sweep ID (for run action)")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of sweep runs")
    
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.sweep_config:
            args.sweep_config = f"artifacts/sweep_configs/{args.dataset}_sweep.yaml"
        
        # Create sweep config
        sweep_config_path = create_sweep_config(args.sweep_config, args.dataset, args.method)
        
        # Create sweep
        try:
            import wandb
            wandb.login()
            sweep_id = wandb.sweep(sweep_config_path, project=args.project)
            print(f"Created sweep with ID: {sweep_id}")
            print(f"Project: {args.project}")
            print(f"Config: {sweep_config_path}")
        except ImportError:
            print("ERROR: wandb not installed. Please install with: pip install wandb")
            return 1
        except Exception as e:
            print(f"ERROR creating sweep: {e}")
            return 1
    
    elif args.action == "run":
        if not args.sweep_id:
            print("ERROR: --sweep_id required for run action")
            return 1
        
        try:
            import wandb
            wandb.login()
            
            # Run sweep agent
            wandb.agent(args.sweep_id, 
                       function=lambda: run_sweep_agent(args.sweep_id, args.project, args.count, args.config),
                       count=args.count, 
                       project=args.project)
        except ImportError:
            print("ERROR: wandb not installed. Please install with: pip install wandb")
            return 1
        except Exception as e:
            print(f"ERROR running sweep: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
