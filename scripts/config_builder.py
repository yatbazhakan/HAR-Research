#!/usr/bin/env python3
"""
HAR Experiment Configuration Builder
Builds command line arguments from YAML configuration files.
"""

import argparse
import yaml
import sys
from pathlib import Path

def build_command_from_config(config_file, output_file=None):
    """Build command line from YAML config file"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build command
    cmd_parts = ["python", "scripts/train_baselines.py"]
    
    # Required arguments (escape glob patterns for shell)
    required_args = ["shards_glob", "stats", "model", "epochs", "batch_size", "lr"]
    for arg in required_args:
        if arg in config:
            if arg == "shards_glob":
                escaped_value = str(config[arg]).replace('*', '\\*')
                cmd_parts.extend([f"--{arg}", escaped_value])
            else:
                cmd_parts.extend([f"--{arg}", str(config[arg])])
    
    # CV arguments
    if "cv" in config:
        cmd_parts.extend(["--cv", config["cv"]])
        
        if config["cv"] == "fold_json" and "fold_json" in config and config["fold_json"]:
            cmd_parts.extend(["--fold_json", config["fold_json"]])
        elif config["cv"] == "holdout":
            if "holdout_test_ratio" in config:
                cmd_parts.extend(["--holdout_test_ratio", str(config["holdout_test_ratio"])])
            if "holdout_val_ratio" in config:
                cmd_parts.extend(["--holdout_val_ratio", str(config["holdout_val_ratio"])])
        elif config["cv"] == "kfold":
            if "kfold_k" in config:
                cmd_parts.extend(["--kfold_k", str(config["kfold_k"])])
            if "kfold_idx" in config:
                cmd_parts.extend(["--kfold_idx", str(config["kfold_idx"])])
    
    # Optional arguments
    optional_args = ["num_workers", "plot_dir", "class_names", "wandb_project", "wandb_run"]
    for arg in optional_args:
        if arg in config and config[arg]:
            cmd_parts.extend([f"--{arg}", str(config[arg])])
    
    # Boolean flags
    boolean_flags = ["calibrate", "amp", "wandb"]
    for flag in boolean_flags:
        if flag in config and config[flag]:
            cmd_parts.append(f"--{flag}")
    
    command = " ".join(cmd_parts)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"#!/bin/bash\n{command}\n")
        print(f"Command saved to {output_file}")
    else:
        print(command)
    
    return command

def create_example_configs():
    """Create example configuration files"""
    
    # UCI-HAR LOSO example
    uci_loso_config = {
        "shards_glob": "artifacts/preprocessed/uci_har/*.npz",
        "stats": "artifacts/norm_stats/uci_har.json",
        "model": "cnn_tcn",
        "epochs": 10,
        "batch_size": 128,
        "lr": 0.001,
        "cv": "fold_json",
        "fold_json": "artifacts/folds/uci_har/loso_fold_subject_1.json",
        "num_workers": 4,
        "plot_dir": "artifacts/plots",
        "class_names": "Walking,Upstairs,Downstairs,Sitting,Standing,Laying",
        "calibrate": True,
        "amp": False,
        "wandb": False,
        "wandb_project": "har-baselines",
        "wandb_run": "uci-loso-subject-1"
    }
    
    # UCI-HAR Holdout example
    uci_holdout_config = {
        "shards_glob": "artifacts/preprocessed/uci_har/*.npz",
        "stats": "artifacts/norm_stats/uci_har.json",
        "model": "cnn_bilstm",
        "epochs": 15,
        "batch_size": 256,
        "lr": 0.0005,
        "cv": "holdout",
        "holdout_test_ratio": 0.2,
        "holdout_val_ratio": 0.1,
        "num_workers": 8,
        "plot_dir": "artifacts/plots",
        "class_names": "Walking,Upstairs,Downstairs,Sitting,Standing,Laying",
        "calibrate": True,
        "amp": True,
        "wandb": True,
        "wandb_project": "har-baselines",
        "wandb_run": "uci-holdout-bilstm"
    }
    
    # PAMAP2 K-fold example
    pamap2_kfold_config = {
        "shards_glob": "artifacts/preprocessed/pamap2/*.npz",
        "stats": "artifacts/norm_stats/pamap2.json",
        "model": "cnn_tcn",
        "epochs": 20,
        "batch_size": 128,
        "lr": 0.001,
        "cv": "kfold",
        "kfold_k": 5,
        "kfold_idx": 0,
        "num_workers": 4,
        "plot_dir": "artifacts/plots",
        "class_names": "Lying,Sitting,Standing,Walking,Running,Cycling,Nordic_walking,Ascending_stairs,Descending_stairs,Vacuum_cleaning,Ironing,Rope_jumping",
        "calibrate": True,
        "amp": False,
        "wandb": True,
        "wandb_project": "har-baselines",
        "wandb_run": "pamap2-kfold-fold-0"
    }
    
    # Save example configs
    configs = {
        "uci_har_loso.yaml": uci_loso_config,
        "uci_har_holdout.yaml": uci_holdout_config,
        "pamap2_kfold.yaml": pamap2_kfold_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created example config: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Build HAR experiment commands from YAML configs")
    parser.add_argument("config_file", help="YAML configuration file")
    parser.add_argument("--output", "-o", help="Output script file (optional)")
    parser.add_argument("--create-examples", action="store_true", help="Create example configuration files")
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_configs()
        return
    
    if not Path(args.config_file).exists():
        print(f"Error: Configuration file {args.config_file} not found")
        sys.exit(1)
    
    try:
        build_command_from_config(args.config_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
