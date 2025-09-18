"""
Training GUI for HAR Experiments

This GUI provides a clean interface for configuring and running single HAR experiments
without sweep functionality. Focused on regular training operations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import yaml
import subprocess
import threading
import time
from pathlib import Path
import os
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TrainingConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAR Training Configuration")
        self.root.geometry("800x900")
        
        # Track active tmux sessions
        self.active_tmux_sessions = set()
        
        # Configuration storage
        self.config = {}
        self.reset_config()
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create GUI
        self.create_widgets()
        
    def reset_config(self):
        """Reset configuration to default values"""
        self.config = {
            # Dataset settings
            "dataset": "",
            "shards_glob": "",
            "stats": "",
            "fold_json": "",
            "class_names": "",
            
            # Model settings
            "model": "cnn_tcn",
            "epochs": "100",
            "batch_size": "32",
            "lr": "0.001",
            # Note: dropout and weight_decay are not supported by train_baselines.py
            
            # Training settings
            "cv": "fold_json",
            "holdout_ratio": "0.2",
            "kfold_splits": "5",
            "num_workers": "4",
            "amp": False,
            
            # Logging settings
            "wandb": False,
            "wandb_project": "har-training",
            "wandb_run": "",
            "plot_dir": "artifacts/plots",
            
            # Execution settings
            "use_tmux": False,
            "tmux_session": "har_training",
            "log_dir": "logs"
        }
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Dataset Configuration
        self.create_dataset_section(scrollable_frame)
        
        # Model Configuration
        self.create_model_section(scrollable_frame)
        
        # Training Configuration
        self.create_training_section(scrollable_frame)
        
        # Cross-Validation Configuration
        self.create_cv_section(scrollable_frame)
        
        # Logging Configuration
        self.create_logging_section(scrollable_frame)
        
        # Execution Configuration
        self.create_execution_section(scrollable_frame)
        
        # Control Buttons
        self.create_control_buttons(scrollable_frame)
        
        # Log Output
        self.create_log_section(scrollable_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_dataset_section(self, parent):
        """Create dataset configuration section"""
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding=10)
        dataset_frame.pack(fill=tk.X, pady=5)
        
        # Dataset selection
        self.create_field(dataset_frame, "dataset", "Dataset", "", 0, 
                         combobox_values=["uci_har", "pamap2", "mhealth"])
        
        # Bind dataset change to auto-fill paths
        self.dataset_var.trace('w', lambda *args: self.on_dataset_change())
        
        # Shards glob pattern
        self.create_field(dataset_frame, "shards_glob", "Shards Glob Pattern", "", 1)
        ttk.Button(dataset_frame, text="Browse", 
                  command=self.browse_shards).grid(row=1, column=2, padx=5)
        
        # Stats JSON path
        self.create_field(dataset_frame, "stats", "Stats JSON Path", "", 2)
        ttk.Button(dataset_frame, text="Browse", 
                  command=self.browse_stats).grid(row=2, column=2, padx=5)
        
        # Fold JSON path
        self.create_field(dataset_frame, "fold_json", "Fold JSON Path", "", 3)
        ttk.Button(dataset_frame, text="Browse", 
                  command=self.browse_folds).grid(row=3, column=2, padx=5)
        
        # Class names
        self.create_field(dataset_frame, "class_names", "Class Names", "", 4)
        
        # Auto-scan button
        ttk.Button(dataset_frame, text="Auto-Scan Datasets", 
                  command=self.auto_scan_datasets).grid(row=5, column=0, columnspan=3, pady=5)
        
    def create_model_section(self, parent):
        """Create model configuration section"""
        model_frame = ttk.LabelFrame(parent, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Model selection
        self.create_field(model_frame, "model", "Model", "cnn_tcn", 0,
                         combobox_values=["cnn_tcn", "cnn_bilstm"])
        
        # Hyperparameters
        self.create_field(model_frame, "epochs", "Epochs", "100", 1, "int")
        self.create_field(model_frame, "batch_size", "Batch Size", "32", 2, "int")
        self.create_field(model_frame, "lr", "Learning Rate", "0.001", 3, "float")
        # Note: dropout and weight_decay are not supported by train_baselines.py
        
    def create_training_section(self, parent):
        """Create training configuration section"""
        training_frame = ttk.LabelFrame(parent, text="Training Configuration", padding=10)
        training_frame.pack(fill=tk.X, pady=5)
        
        # Number of workers
        self.create_field(training_frame, "num_workers", "Num Workers", "4", 0, "int")
        
        # Mixed precision
        self.create_checkbox(training_frame, "amp", "Use Mixed Precision (AMP)", 1)
        
    def create_cv_section(self, parent):
        """Create cross-validation configuration section"""
        cv_frame = ttk.LabelFrame(parent, text="Cross-Validation Configuration", padding=10)
        cv_frame.pack(fill=tk.X, pady=5)
        
        # CV method
        self.create_field(cv_frame, "cv", "CV Method", "fold_json", 0,
                         combobox_values=["fold_json", "holdout", "kfold", "loso"])
        
        # Holdout ratio
        self.create_field(cv_frame, "holdout_ratio", "Holdout Ratio", "0.2", 1, "float")
        
        # K-fold splits
        self.create_field(cv_frame, "kfold_splits", "K-Fold Splits", "5", 2, "int")
        
    def create_logging_section(self, parent):
        """Create logging configuration section"""
        logging_frame = ttk.LabelFrame(parent, text="Logging Configuration", padding=10)
        logging_frame.pack(fill=tk.X, pady=5)
        
        # WandB settings
        self.create_checkbox(logging_frame, "wandb", "Use Weights & Biases", 0)
        self.create_field(logging_frame, "wandb_project", "WandB Project", "har-training", 1)
        self.create_field(logging_frame, "wandb_run", "WandB Run Name", "", 2)
        
        # Plot directory
        self.create_field(logging_frame, "plot_dir", "Plot Directory", "artifacts/plots", 3)
        
    def create_execution_section(self, parent):
        """Create execution configuration section"""
        execution_frame = ttk.LabelFrame(parent, text="Execution Configuration", padding=10)
        execution_frame.pack(fill=tk.X, pady=5)
        
        # Tmux settings
        self.create_checkbox(execution_frame, "use_tmux", "Run in Tmux Session", 0)
        self.create_field(execution_frame, "tmux_session", "Tmux Session Name", "har_training", 1)
        self.create_field(execution_frame, "log_dir", "Log Directory", "logs", 2)
        
    def create_control_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        ttk.Button(left_buttons, text="Reset Config", command=self.reset_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        
        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="Generate Command", command=self.generate_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Run Experiment", command=self.run_experiment).pack(side=tk.LEFT, padx=5)
        
    def create_log_section(self, parent):
        """Create log output section"""
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Log text widget
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tmux management section
        tmux_frame = ttk.LabelFrame(parent, text="Tmux Session Management", padding=10)
        tmux_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(tmux_frame, text="List Active Sessions", 
                  command=self.list_active_sessions).pack(side=tk.LEFT, padx=5)
        ttk.Button(tmux_frame, text="Kill All Sessions", 
                  command=self.kill_all_tmux_sessions).pack(side=tk.LEFT, padx=5)
        ttk.Button(tmux_frame, text="Debug Tmux", 
                  command=self.debug_tmux).pack(side=tk.LEFT, padx=5)
        
    def create_field(self, parent, key, label, default, row, field_type="str", column_offset=0, combobox_values=None):
        """Create a labeled input field"""
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=column_offset, sticky=tk.W, padx=5, pady=2)
        
        if field_type == "int":
            var = tk.StringVar(value=default)
            entry = ttk.Entry(parent, textvariable=var, width=20)
            entry.grid(row=row, column=column_offset+1, sticky=tk.W, padx=5, pady=2)
        elif field_type == "float":
            var = tk.StringVar(value=default)
            entry = ttk.Entry(parent, textvariable=var, width=20)
            entry.grid(row=row, column=column_offset+1, sticky=tk.W, padx=5, pady=2)
        elif field_type == "str" and combobox_values is not None:
            var = tk.StringVar(value=default)
            combobox = ttk.Combobox(parent, textvariable=var, values=combobox_values, width=17)
            combobox.grid(row=row, column=column_offset+1, sticky=tk.W, padx=5, pady=2)
        else:
            var = tk.StringVar(value=default)
            entry = ttk.Entry(parent, textvariable=var, width=20)
            entry.grid(row=row, column=column_offset+1, sticky=tk.W, padx=5, pady=2)
        
        # Store reference for later access
        setattr(self, f"{key}_var", var)
        self.config[key] = var.get()
        
        # Bind update event
        var.trace('w', lambda *args: self.update_config(key, var.get()))
        
    def create_checkbox(self, parent, key, label, row):
        """Create a checkbox"""
        var = tk.BooleanVar(value=self.config.get(key, False))
        checkbox = ttk.Checkbutton(parent, text=label, variable=var)
        checkbox.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        setattr(self, f"{key}_var", var)
        var.trace('w', lambda *args: self.update_config(key, var.get()))
        
    def update_config(self, key, value):
        """Update configuration when values change"""
        self.config[key] = value
        
    def on_dataset_change(self):
        """Called when dataset selection changes - auto-fill paths"""
        dataset_name = self.dataset_var.get()
        if dataset_name:
            self.log_message(f"Dataset changed to: {dataset_name}")
            self.update_dataset_paths(dataset_name)
        
    def auto_scan_datasets(self):
        """Auto-scan for available datasets"""
        self.log_message("Scanning for available datasets...")
        
        # Look for preprocessed datasets
        preprocessed_dir = Path("artifacts/preprocessed")
        if preprocessed_dir.exists():
            datasets = []
            for dataset_dir in preprocessed_dir.iterdir():
                if dataset_dir.is_dir():
                    datasets.append(dataset_dir.name)
            
            if datasets:
                self.dataset_var.set(datasets[0])
                self.update_dataset_paths(datasets[0])
                self.log_message(f"Found datasets: {datasets}")
            else:
                self.log_message("No datasets found in artifacts/preprocessed/")
        else:
            self.log_message("Preprocessed directory not found")
            
    def update_dataset_paths(self, dataset_name):
        """Update paths based on selected dataset with existence checking"""
        if dataset_name:
            # Update shards glob
            shards_pattern = f"artifacts/preprocessed/{dataset_name}/*.npz"
            self.shards_glob_var.set(shards_pattern)
            
            # Check if shards exist
            shards_exist = len(list(Path("artifacts/preprocessed").glob(f"{dataset_name}/*.npz"))) > 0
            if shards_exist:
                self.log_message(f"✓ Found NPZ shards for {dataset_name}")
            else:
                self.log_message(f"⚠ No NPZ shards found for {dataset_name}")
            
            # Update stats path
            stats_path = f"artifacts/norm_stats/{dataset_name}.json"
            self.stats_var.set(stats_path)
            
            # Check if stats exist
            if Path(stats_path).exists():
                self.log_message(f"✓ Found normalization stats for {dataset_name}")
            else:
                self.log_message(f"⚠ No normalization stats found for {dataset_name}")
                self.log_message(f"  Run: python scripts/compute_norm_stats.py --shards_glob '{shards_pattern}' --split train")
            
            # Update fold JSON path - try to find any fold file
            fold_dir = Path(f"artifacts/folds/{dataset_name}")
            if fold_dir.exists():
                fold_files = list(fold_dir.glob("*.json"))
                if fold_files:
                    # Use the first fold file found
                    fold_path = str(fold_files[0])
                    self.fold_json_var.set(fold_path)
                    self.log_message(f"✓ Found fold file: {fold_files[0].name}")
                else:
                    # Default to loso_fold_subject_1.json
                    fold_path = f"artifacts/folds/{dataset_name}/loso_fold_subject_1.json"
                    self.fold_json_var.set(fold_path)
                    self.log_message(f"⚠ No fold files found in {fold_dir}")
                    self.log_message(f"  Run: python scripts/generate_loso_folds.py --shards_glob '{shards_pattern}'")
            else:
                fold_path = f"artifacts/folds/{dataset_name}/loso_fold_subject_1.json"
                self.fold_json_var.set(fold_path)
                self.log_message(f"⚠ Fold directory not found: {fold_dir}")
                self.log_message(f"  Run: python scripts/generate_loso_folds.py --shards_glob '{shards_pattern}'")
            
            # Update class names based on dataset
            class_names_map = {
                "uci_har": "WALKING,WALKING_UPSTAIRS,WALKING_DOWNSTAIRS,SITTING,STANDING,LAYING",
                "pamap2": "lying,sitting,standing,walking,running,cycling,Nordic_walking,ascending_stairs,descending_stairs,vacuum_cleaning,ironing,rope_jumping",
                "mhealth": "0,1,2,3,4,5,6,7,8,9,10,11"
            }
            if dataset_name in class_names_map:
                self.class_names_var.set(class_names_map[dataset_name])
                self.log_message(f"✓ Auto-filled class names for {dataset_name}")
            
            # Update WandB project name based on dataset
            self.wandb_project_var.set(f"har-training-{dataset_name}")
            self.log_message(f"✓ Updated WandB project to: har-training-{dataset_name}")
                
    def browse_shards(self):
        """Browse for shards glob pattern"""
        filename = filedialog.askopenfilename(
            title="Select NPZ file (will use glob pattern)",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            # Convert to glob pattern
            path = Path(filename)
            glob_pattern = str(path.parent / "*.npz")
            self.shards_glob_var.set(glob_pattern)
            
    def browse_stats(self):
        """Browse for stats JSON file"""
        filename = filedialog.askopenfilename(
            title="Select Stats JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.stats_var.set(filename)
            
    def browse_folds(self):
        """Browse for fold JSON file"""
        filename = filedialog.askopenfilename(
            title="Select Fold JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.fold_json_var.set(filename)
            
    def generate_command(self):
        """Generate training command"""
        try:
            # Validate required fields
            required_fields = ["shards_glob", "stats", "model", "epochs", "batch_size", "lr"]
            missing_fields = [field for field in required_fields if not self.config.get(field)]
            
            if missing_fields:
                self.log_message(f"ERROR: Missing required fields: {missing_fields}")
                return
                
            # Build command
            cmd_parts = ["python", "scripts/train_baselines.py"]
            
            # Add arguments (only those supported by train_baselines.py)
            cmd_parts.extend(["--shards_glob", self.config["shards_glob"]])
            cmd_parts.extend(["--stats", self.config["stats"]])
            cmd_parts.extend(["--model", self.config["model"]])
            cmd_parts.extend(["--epochs", self.config["epochs"]])
            cmd_parts.extend(["--batch_size", self.config["batch_size"]])
            cmd_parts.extend(["--lr", self.config["lr"]])
            # Note: dropout and weight_decay are not supported by train_baselines.py
            
            # CV method
            cmd_parts.extend(["--cv", self.config["cv"]])
            if self.config["cv"] == "holdout":
                cmd_parts.extend(["--holdout_test_ratio", self.config["holdout_ratio"]])
                # Note: holdout_val_ratio is not configurable in GUI, uses default
            elif self.config["cv"] == "kfold":
                cmd_parts.extend(["--kfold_k", self.config["kfold_splits"]])
                # Note: kfold_idx is not configurable in GUI, uses default
            elif self.config["cv"] == "fold_json":
                cmd_parts.extend(["--fold_json", self.config["fold_json"]])
                
            # Training settings
            cmd_parts.extend(["--num_workers", self.config["num_workers"]])
            if self.config["amp"]:
                cmd_parts.append("--amp")
                
            # Logging settings
            if self.config["wandb"]:
                cmd_parts.append("--wandb")
                cmd_parts.extend(["--wandb_project", self.config["wandb_project"]])
                if self.config["wandb_run"]:
                    cmd_parts.extend(["--wandb_run", self.config["wandb_run"]])
                    
            cmd_parts.extend(["--plot_dir", self.config["plot_dir"]])
            
            # Class names
            if self.config["class_names"]:
                cmd_parts.extend(["--class_names", self.config["class_names"]])
                
            # Display command
            command = " ".join(cmd_parts)
            self.log_message("Generated Command:")
            self.log_message(command)
            
            # Store command for execution
            self.current_command = cmd_parts
            
        except Exception as e:
            self.log_message(f"ERROR generating command: {str(e)}")
            
    def run_experiment(self):
        """Run the training experiment"""
        if not hasattr(self, 'current_command'):
            self.log_message("ERROR: Please generate command first")
            return
            
        # Start experiment in separate thread
        thread = threading.Thread(target=self._run_experiment_thread)
        thread.daemon = True
        thread.start()
        
    def _run_experiment_thread(self):
        """Run experiment in separate thread"""
        try:
            self.log_message("Starting experiment...")
            
            if self.config["use_tmux"]:
                # Check if tmux is available
                if not self._check_tmux_available():
                    self.log_message("tmux not available, falling back to direct execution")
                    self._run_direct()
                    return
                    
                # Run in tmux session
                self._run_in_tmux()
            else:
                # Run directly
                self._run_direct()
                
        except Exception as e:
            self.log_message(f"ERROR running experiment: {str(e)}")
            
    def _check_tmux_available(self):
        """Check if tmux is available on the system"""
        try:
            result = subprocess.run(
                "tmux -V",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_message(f"tmux available: {result.stdout.strip()}")
                return True
            else:
                self.log_message("tmux not found in PATH")
                return False
                
        except Exception as e:
            self.log_message(f"Error checking tmux availability: {e}")
            return False
            
    def _quote_command_for_shell(self, command_parts):
        """Quote command arguments that contain glob patterns for shell execution"""
        quoted_command = []
        i = 0
        while i < len(command_parts):
            # Arguments that might contain glob patterns or paths with special characters
            if (command_parts[i] in ["--shards_glob", "--fold_json", "--stats", "--plot_dir", 
                                   "--class_names", "--wandb_project", "--wandb_run"] and 
                i + 1 < len(command_parts)):
                # Quote the argument value to prevent shell expansion
                quoted_command.append(command_parts[i])
                quoted_command.append(f'"{command_parts[i + 1]}"')
                i += 2
            else:
                quoted_command.append(command_parts[i])
                i += 1
        return quoted_command
            
    def _run_direct(self):
        """Run experiment directly"""
        process = subprocess.Popen(
            self.current_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            self.log_message(line.rstrip())
            
        process.wait()
        self.log_message(f"Experiment completed with exit code: {process.returncode}")
        
    def _run_in_tmux(self):
        """Run experiment in tmux session"""
        session_name = self.config['tmux_session']
        
        # Check if session already exists
        try:
            result = subprocess.run(
                f"tmux has-session -t {session_name}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_message(f"Session {session_name} already exists. Killing it first...")
                subprocess.run(f"tmux kill-session -t {session_name}", shell=True)
                time.sleep(1)  # Wait a moment
        except Exception as e:
            self.log_message(f"Error checking existing session: {e}")
        
        # Create log file
        log_file = Path(self.config["log_dir"]) / f"training_{int(time.time())}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create bash script for tmux with proper quoting
        quoted_command = self._quote_command_for_shell(self.current_command)
        
        script_content = f"""#!/bin/bash
cd {REPO_ROOT}
echo "Starting training experiment in tmux session: {session_name}"
echo "Command: {' '.join(quoted_command)}"
echo "Log file: {log_file}"
echo "----------------------------------------"
{' '.join(quoted_command)} 2>&1 | tee {log_file}
echo "----------------------------------------"
echo "Experiment completed. Session will remain open."
echo "Press Ctrl+C to exit or close this window."
while true; do
    sleep 10
done
"""
        
        script_path = Path("temp_training_script.sh")
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        # Run tmux command with better error handling
        try:
            # First create the session
            create_cmd = f"tmux new-session -d -s {session_name}"
            result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log_message(f"Error creating tmux session: {result.stderr}")
                return
                
            # Then send the command to the session
            send_cmd = f"tmux send-keys -t {session_name} 'bash {script_path}' Enter"
            result = subprocess.run(send_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log_message(f"Error sending command to tmux session: {result.stderr}")
                return
                
            # Track the session
            self.active_tmux_sessions.add(session_name)
            
            self.log_message(f"✓ Started tmux session: {session_name}")
            self.log_message(f"✓ Log file: {log_file}")
            self.log_message(f"✓ Use 'tmux attach -t {session_name}' to view session")
            self.log_message("✓ Session will continue running even if GUI is closed")
            
            # Verify session exists
            time.sleep(1)
            verify_result = subprocess.run(
                f"tmux has-session -t {session_name}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if verify_result.returncode == 0:
                self.log_message("✓ Session verified and running")
            else:
                self.log_message("⚠ Session verification failed")
                
        except Exception as e:
            self.log_message(f"Error running tmux command: {str(e)}")
            self.log_message("Make sure tmux is installed and available in PATH")
        
    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                        
                # Update GUI with loaded config
                for key, value in config.items():
                    if hasattr(self, f"{key}_var"):
                        getattr(self, f"{key}_var").set(value)
                        
                self.log_message(f"Loaded configuration from {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR loading config: {str(e)}")
                
    def save_config(self):
        """Save configuration to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        yaml.dump(self.config, f, default_flow_style=False)
                    else:
                        json.dump(self.config, f, indent=2)
                        
                self.log_message(f"Saved configuration to {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR saving config: {str(e)}")
                
    def log_message(self, message):
        """Add message to log output"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def on_closing(self):
        """Handle window closing - ask user about tmux sessions"""
        if self.active_tmux_sessions:
            sessions_list = ", ".join(self.active_tmux_sessions)
            result = messagebox.askyesnocancel(
                "Active Tmux Sessions",
                f"You have active tmux sessions: {sessions_list}\n\n"
                "What would you like to do?\n"
                "Yes: Kill all sessions and close GUI\n"
                "No: Keep sessions running and close GUI\n"
                "Cancel: Don't close GUI"
            )
            
            if result is True:  # Yes - kill sessions
                self.kill_all_tmux_sessions()
                self.root.destroy()
            elif result is False:  # No - keep sessions
                self.log_message("Keeping tmux sessions running...")
                self.root.destroy()
            # Cancel - do nothing, keep GUI open
        else:
            self.root.destroy()
            
    def kill_all_tmux_sessions(self):
        """Kill all tracked tmux sessions"""
        for session_name in self.active_tmux_sessions.copy():
            try:
                # Check if session exists
                result = subprocess.run(
                    f"tmux has-session -t {session_name}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:  # Session exists
                    subprocess.run(f"tmux kill-session -t {session_name}", shell=True)
                    self.log_message(f"Killed tmux session: {session_name}")
                    self.active_tmux_sessions.remove(session_name)
                else:
                    self.log_message(f"Tmux session {session_name} not found")
                    self.active_tmux_sessions.remove(session_name)
                    
            except Exception as e:
                self.log_message(f"Error killing session {session_name}: {str(e)}")
                
    def list_active_sessions(self):
        """List all active tmux sessions"""
        try:
            result = subprocess.run(
                "tmux list-sessions",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                sessions = result.stdout.strip().split('\n')
                if sessions and sessions[0]:  # Not empty
                    self.log_message("Active tmux sessions:")
                    for session in sessions:
                        self.log_message(f"  {session}")
                else:
                    self.log_message("No active tmux sessions")
            else:
                self.log_message("No tmux sessions found")
                
        except Exception as e:
            self.log_message(f"Error listing sessions: {str(e)}")
            
    def debug_tmux(self):
        """Debug tmux installation and configuration"""
        self.log_message("=== TMUX DEBUG INFORMATION ===")
        
        # Check tmux version
        try:
            result = subprocess.run(
                "tmux -V",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_message(f"✓ tmux version: {result.stdout.strip()}")
            else:
                self.log_message(f"✗ tmux not found: {result.stderr}")
                return
        except Exception as e:
            self.log_message(f"✗ Error checking tmux: {e}")
            return
            
        # Check tmux configuration
        try:
            result = subprocess.run(
                "tmux show-options -g",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_message("✓ tmux configuration loaded successfully")
            else:
                self.log_message(f"⚠ tmux configuration warning: {result.stderr}")
        except Exception as e:
            self.log_message(f"⚠ Error checking tmux config: {e}")
            
        # List all sessions
        try:
            result = subprocess.run(
                "tmux list-sessions",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                sessions = result.stdout.strip()
                if sessions:
                    self.log_message(f"✓ Current sessions:\n{sessions}")
                else:
                    self.log_message("✓ No active sessions")
            else:
                self.log_message(f"✗ Error listing sessions: {result.stderr}")
        except Exception as e:
            self.log_message(f"✗ Error listing sessions: {e}")
            
        # Check if we can create a test session
        test_session = "debug_test_session"
        try:
            # Kill test session if it exists
            subprocess.run(f"tmux kill-session -t {test_session}", shell=True, capture_output=True)
            
            # Create test session
            result = subprocess.run(
                f"tmux new-session -d -s {test_session}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_message(f"✓ Successfully created test session: {test_session}")
                
                # Send a test command
                subprocess.run(
                    f"tmux send-keys -t {test_session} 'echo Hello from tmux!' Enter",
                    shell=True,
                    capture_output=True
                )
                
                # Wait a moment and check if session is still alive
                time.sleep(1)
                check_result = subprocess.run(
                    f"tmux has-session -t {test_session}",
                    shell=True,
                    capture_output=True
                )
                
                if check_result.returncode == 0:
                    self.log_message("✓ Test session is running")
                else:
                    self.log_message("✗ Test session died immediately")
                    
                # Clean up test session
                subprocess.run(f"tmux kill-session -t {test_session}", shell=True, capture_output=True)
                self.log_message("✓ Test session cleaned up")
                
            else:
                self.log_message(f"✗ Failed to create test session: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"✗ Error testing tmux: {e}")
            
        self.log_message("=== END DEBUG INFORMATION ===")


def main():
    root = tk.Tk()
    app = TrainingConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
