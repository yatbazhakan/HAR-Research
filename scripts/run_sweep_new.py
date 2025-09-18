#!/usr/bin/env python3
"""
Enhanced Wandb Sweep Runner for HAR Experiments

This script provides improved functionality for creating and running
Weights & Biases sweeps with better integration with the GUI tools.
"""

import argparse
import yaml
import subprocess
import sys
import wandb
from pathlib import Path
import time

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def create_sweep_config(output_file, method="bayes", metric="val/accuracy", goal="maximize",
                       lr_range=(1e-5, 1e-1), batch_sizes=[16, 32, 64, 128],
                       epochs=[50, 100, 150, 200], dropout_range=(0.1, 0.5),
                       weight_decay_range=(1e-6, 1e-2), models=["cnn_tcn", "cnn_bilstm"]):
    """Create a sweep configuration file with custom parameters"""
    
    config = {
        "method": method,
        "metric": {
            "name": metric,
            "goal": goal
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": lr_range[0],
                "max": lr_range[1]
            },
            "batch_size": {
                "values": batch_sizes
            },
            "epochs": {
                "values": epochs
            },
            "model": {
                "values": models
            },
            "dropout": {
                "distribution": "uniform",
                "min": dropout_range[0],
                "max": dropout_range[1]
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": weight_decay_range[0],
                "max": weight_decay_range[1]
            }
        }
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created sweep config: {output_file}")
    return output_file


def create_sweep(base_config_file, sweep_config_file, project="har-sweeps"):
    """Create a new sweep using wandb"""
    try:
        # Load base configuration
        with open(base_config_file, 'r') as f:
            base_config = yaml.safe_load(f)
            
        # Load sweep configuration
        with open(sweep_config_file, 'r') as f:
            sweep_config = yaml.safe_load(f)
            
        # Initialize wandb
        wandb.login()
        
        # Create sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=project
        )
        
        print(f"Sweep created successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(f"Project: {project}")
        print(f"View at: https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error creating sweep: {str(e)}")
        return None


def run_sweep_agent(base_config_file, sweep_id=None, project="har-sweeps", count=10):
    """Run sweep agent"""
    try:
        # Load base configuration
        with open(base_config_file, 'r') as f:
            base_config = yaml.safe_load(f)
            
        # Create command for train_baselines.py
        cmd = [
            "python", "scripts/train_baselines.py",
            "--shards_glob", base_config["shards_glob"],
            "--stats", base_config["stats"],
            "--epochs", str(base_config.get("epochs", 100)),
            "--batch_size", str(base_config.get("batch_size", 32)),
            "--lr", str(base_config.get("lr", 0.001)),
            "--dropout", str(base_config.get("dropout", 0.2)),
            "--weight_decay", str(base_config.get("weight_decay", 1e-4)),
            "--cv", base_config.get("cv", "fold_json"),
            "--num_workers", str(base_config.get("num_workers", 4)),
            "--plot_dir", base_config.get("plot_dir", "artifacts/plots"),
            "--wandb"
        ]
        
        # Add optional arguments
        if base_config.get("fold_json"):
            cmd.extend(["--fold_json", base_config["fold_json"]])
        if base_config.get("class_names"):
            cmd.extend(["--class_names", base_config["class_names"]])
        if base_config.get("amp"):
            cmd.append("--amp")
            
        # Add wandb project
        cmd.extend(["--wandb_project", project])
        
        print(f"Starting sweep agent...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run sweep agent
        if sweep_id:
            wandb.agent(sweep_id, function=lambda: subprocess.run(cmd), count=count)
        else:
            print("No sweep ID provided. Please create a sweep first.")
            
    except Exception as e:
        print(f"Error running sweep agent: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Run wandb sweeps for HAR experiments")
    parser.add_argument("--action", choices=["create", "run"], required=True,
                       help="Action to perform: create sweep config or run sweep")
    parser.add_argument("--config", type=str, required=True,
                       help="Base experiment configuration YAML file")
    parser.add_argument("--sweep_config", type=str,
                       help="Sweep configuration YAML file (for create action)")
    parser.add_argument("--project", type=str, default="har-sweeps",
                       help="Wandb project name")
    parser.add_argument("--sweep_id", type=str,
                       help="Sweep ID (for run action)")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of sweep runs")
    
    # Sweep configuration parameters
    parser.add_argument("--method", type=str, default="bayes",
                       choices=["bayes", "random", "grid"],
                       help="Sweep method")
    parser.add_argument("--metric", type=str, default="val/accuracy",
                       help="Metric to optimize")
    parser.add_argument("--goal", type=str, default="maximize",
                       choices=["maximize", "minimize"],
                       help="Optimization goal")
    
    # Parameter ranges
    parser.add_argument("--lr_min", type=float, default=1e-5,
                       help="Minimum learning rate")
    parser.add_argument("--lr_max", type=float, default=1e-1,
                       help="Maximum learning rate")
    parser.add_argument("--batch_sizes", type=str, default="16,32,64,128",
                       help="Batch sizes (comma-separated)")
    parser.add_argument("--epochs", type=str, default="50,100,150,200",
                       help="Epochs (comma-separated)")
    parser.add_argument("--dropout_min", type=float, default=0.1,
                       help="Minimum dropout")
    parser.add_argument("--dropout_max", type=float, default=0.5,
                       help="Maximum dropout")
    parser.add_argument("--wd_min", type=float, default=1e-6,
                       help="Minimum weight decay")
    parser.add_argument("--wd_max", type=float, default=1e-2,
                       help="Maximum weight decay")
    parser.add_argument("--models", type=str, default="cnn_tcn,cnn_bilstm",
                       help="Models to sweep (comma-separated)")
    
    args = parser.parse_args()
    
    if args.action == "create":
        # Create sweep configuration
        if not args.sweep_config:
            # Generate default sweep config
            sweep_config_file = Path("artifacts/sweep_configs") / f"sweep_{int(time.time())}.yaml"
            sweep_config_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            sweep_config_file = args.sweep_config
            
        # Parse parameter lists
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        epochs = [int(x.strip()) for x in args.epochs.split(",")]
        models = [x.strip() for x in args.models.split(",")]
        
        # Create sweep config
        create_sweep_config(
            output_file=sweep_config_file,
            method=args.method,
            metric=args.metric,
            goal=args.goal,
            lr_range=(args.lr_min, args.lr_max),
            batch_sizes=batch_sizes,
            epochs=epochs,
            dropout_range=(args.dropout_min, args.dropout_max),
            weight_decay_range=(args.wd_min, args.wd_max),
            models=models
        )
        
        # Create sweep
        sweep_id = create_sweep(args.config, sweep_config_file, args.project)
        if sweep_id:
            print(f"\nTo run the sweep, use:")
            print(f"python scripts/run_sweep_new.py --action run --config {args.config} --sweep_id {sweep_id} --project {args.project}")
            
    elif args.action == "run":
        # Run sweep
        if not args.sweep_id:
            print("Error: --sweep_id is required for run action")
            return
            
        run_sweep_agent(args.config, args.sweep_id, args.project, args.count)


if __name__ == "__main__":
    main()
