"""
Sweep GUI for HAR Hyperparameter Tuning

This GUI provides a dedicated interface for configuring and running
Weights & Biases sweeps for hyperparameter optimization.
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


class SweepConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAR Sweep Configuration")
        self.root.geometry("900x1000")
        
        # Track active tmux sessions
        self.active_tmux_sessions = set()
        
        # Configuration storage
        self.sweep_config = {}
        self.base_config = {}
        self.reset_configs()
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create GUI
        self.create_widgets()
        
    def reset_configs(self):
        """Reset configurations to default values"""
        # Base experiment configuration
        self.base_config = {
            # Dataset settings
            "dataset": "",
            "shards_glob": "",
            "stats": "",
            "fold_json": "",
            "class_names": "",
            
            # Fixed model settings (not swept)
            "model": "cnn_tcn",
            "num_workers": "4",
            "amp": False,
            
            # CV settings
            "cv": "fold_json",
            "holdout_ratio": "0.2",
            "kfold_splits": "5",
            
            # Logging settings
            "wandb": True,
            "wandb_project": "har-sweeps",
            "plot_dir": "artifacts/plots",
            
            # Execution settings
            "use_tmux": False,
            "tmux_session": "har_sweep",
            "log_dir": "logs"
        }
        
        # Sweep configuration
        self.sweep_config = {
            # Sweep settings
            "sweep_id": "",
            "sweep_count": "20",
            "sweep_config_file": "",
            
            # Sweep parameters
            "method": "bayes",
            "metric": "val/accuracy",
            "goal": "maximize",
            
            # Parameter ranges
            "lr_start": "1e-5",
            "lr_end": "1e-1",
            "batch_sizes": "16,32,64,128",
            "epochs": "50,100,150,200",
            "dropout_start": "0.1",
            "dropout_end": "0.5",
            "weight_decay_start": "1e-6",
            "weight_decay_end": "1e-2",
            
            # Model selection
            "models": ["cnn_tcn", "cnn_bilstm"]
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
        
        # Base Experiment Configuration
        self.create_base_config_section(scrollable_frame)
        
        # Sweep Configuration
        self.create_sweep_config_section(scrollable_frame)
        
        # Sweep Parameters
        self.create_sweep_parameters_section(scrollable_frame)
        
        # Control Buttons
        self.create_control_buttons(scrollable_frame)
        
        # Log Output
        self.create_log_section(scrollable_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_base_config_section(self, parent):
        """Create base experiment configuration section"""
        base_frame = ttk.LabelFrame(parent, text="Base Experiment Configuration", padding=10)
        base_frame.pack(fill=tk.X, pady=5)
        
        # Dataset selection
        self.create_field(base_frame, "dataset", "Dataset", "", 0, 
                         combobox_values=["uci_har", "pamap2", "mhealth"])
        
        # Bind dataset change to auto-fill paths
        self.dataset_var.trace('w', lambda *args: self.on_dataset_change())
        
        # Shards glob pattern
        self.create_field(base_frame, "shards_glob", "Shards Glob Pattern", "", 1)
        ttk.Button(base_frame, text="Browse", 
                  command=self.browse_shards).grid(row=1, column=2, padx=5)
        
        # Stats JSON path
        self.create_field(base_frame, "stats", "Stats JSON Path", "", 2)
        ttk.Button(base_frame, text="Browse", 
                  command=self.browse_stats).grid(row=2, column=2, padx=5)
        
        # Fold JSON path
        self.create_field(base_frame, "fold_json", "Fold JSON Path", "", 3)
        ttk.Button(base_frame, text="Browse", 
                  command=self.browse_folds).grid(row=3, column=2, padx=5)
        
        # Class names
        self.create_field(base_frame, "class_names", "Class Names", "", 4)
        
        # Model (fixed, not swept)
        self.create_field(base_frame, "model", "Base Model", "cnn_tcn", 5,
                         combobox_values=["cnn_tcn", "cnn_bilstm"])
        
        # CV method
        self.create_field(base_frame, "cv", "CV Method", "fold_json", 6,
                         combobox_values=["fold_json", "holdout", "kfold", "loso"])
        
        # WandB project
        self.create_field(base_frame, "wandb_project", "WandB Project", "har-sweeps", 7)
        
        # Auto-scan button
        ttk.Button(base_frame, text="Auto-Scan Datasets", 
                  command=self.auto_scan_datasets).grid(row=8, column=0, columnspan=3, pady=5)
        
    def create_sweep_config_section(self, parent):
        """Create sweep configuration section"""
        sweep_frame = ttk.LabelFrame(parent, text="Sweep Configuration", padding=10)
        sweep_frame.pack(fill=tk.X, pady=5)
        
        # Sweep ID
        self.create_field(sweep_frame, "sweep_id", "Sweep ID (optional)", "", 0)
        
        # Sweep count
        self.create_field(sweep_frame, "sweep_count", "Sweep Count", "20", 1, "int")
        
        # Sweep config file
        self.create_field(sweep_frame, "sweep_config_file", "Sweep Config File", "", 2)
        ttk.Button(sweep_frame, text="Browse", 
                  command=self.browse_sweep_config).grid(row=2, column=2, padx=5)
        
        # Method selection
        self.create_field(sweep_frame, "method", "Method", "bayes", 3,
                         combobox_values=["bayes", "random", "grid"])
        
        # Metric selection
        self.create_field(sweep_frame, "metric", "Metric", "val/accuracy", 4,
                         combobox_values=["val/accuracy", "val/f1", "val/precision", "val/recall"])
        
        # Goal selection
        self.create_field(sweep_frame, "goal", "Goal", "maximize", 5,
                         combobox_values=["maximize", "minimize"])
        
    def create_sweep_parameters_section(self, parent):
        """Create sweep parameters section"""
        params_frame = ttk.LabelFrame(parent, text="Sweep Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        # Learning rate range
        ttk.Label(params_frame, text="Learning Rate Range:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "lr_start", "Min", "1e-5", 0, "float", column_offset=1)
        self.create_field(params_frame, "lr_end", "Max", "1e-1", 0, "float", column_offset=3)
        
        # Batch sizes
        self.create_field(params_frame, "batch_sizes", "Batch Sizes (comma-separated)", "16,32,64,128", 1)
        
        # Epochs
        self.create_field(params_frame, "epochs", "Epochs (comma-separated)", "50,100,150,200", 2)
        
        # Dropout range
        ttk.Label(params_frame, text="Dropout Range:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "dropout_start", "Min", "0.1", 3, "float", column_offset=1)
        self.create_field(params_frame, "dropout_end", "Max", "0.5", 3, "float", column_offset=3)
        
        # Weight decay range
        ttk.Label(params_frame, text="Weight Decay Range:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "weight_decay_start", "Min", "1e-6", 4, "float", column_offset=1)
        self.create_field(params_frame, "weight_decay_end", "Max", "1e-2", 4, "float", column_offset=3)
        
        # Model selection
        ttk.Label(params_frame, text="Models to Sweep:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        model_frame = ttk.Frame(params_frame)
        model_frame.grid(row=5, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        self.create_checkbox(model_frame, "model_cnn_tcn", "CNN-TCN", 0)
        self.create_checkbox(model_frame, "model_cnn_bilstm", "CNN-BiLSTM", 1)
        
    def create_control_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        ttk.Button(left_buttons, text="Reset Configs", command=self.reset_configs).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Load Base Config", command=self.load_base_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Load Sweep Config", command=self.load_sweep_config).pack(side=tk.LEFT, padx=5)
        
        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="Save Base Config", command=self.save_base_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Save Sweep Config", command=self.save_sweep_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Create Sweep", command=self.create_sweep).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Run Sweep", command=self.run_sweep).pack(side=tk.LEFT, padx=5)
        
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
        
        # Determine which config to update
        if key in self.base_config:
            self.base_config[key] = var.get()
            var.trace('w', lambda *args: self.update_base_config(key, var.get()))
        else:
            self.sweep_config[key] = var.get()
            var.trace('w', lambda *args: self.update_sweep_config(key, var.get()))
        
    def create_checkbox(self, parent, key, label, row):
        """Create a checkbox"""
        var = tk.BooleanVar(value=self.sweep_config.get(key, False))
        checkbox = ttk.Checkbutton(parent, text=label, variable=var)
        checkbox.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        
        setattr(self, f"{key}_var", var)
        var.trace('w', lambda *args: self.update_sweep_config(key, var.get()))
        
    def update_base_config(self, key, value):
        """Update base configuration when values change"""
        self.base_config[key] = value
        
    def update_sweep_config(self, key, value):
        """Update sweep configuration when values change"""
        self.sweep_config[key] = value
        
    def on_dataset_change(self):
        """Called when dataset selection changes - auto-fill paths"""
        dataset_name = self.dataset_var.get()
        if dataset_name:
            self.log_message(f"Dataset changed to: {dataset_name}")
            self.update_dataset_paths(dataset_name)
        
    def auto_scan_datasets(self):
        """Auto-scan for available datasets and auto-fill all paths"""
        self.log_message("Scanning for available datasets...")
        
        # Look for preprocessed datasets
        preprocessed_dir = Path("artifacts/preprocessed")
        if preprocessed_dir.exists():
            datasets = []
            for dataset_dir in preprocessed_dir.iterdir():
                if dataset_dir.is_dir():
                    datasets.append(dataset_dir.name)
            
            if datasets:
                # Auto-select first dataset
                selected_dataset = datasets[0]
                self.dataset_var.set(selected_dataset)
                self.update_dataset_paths(selected_dataset)
                self.log_message(f"Found datasets: {datasets}")
                self.log_message(f"Auto-selected: {selected_dataset}")
                self.log_message("All paths have been automatically filled!")
            else:
                self.log_message("No datasets found in artifacts/preprocessed/")
        else:
            self.log_message("Preprocessed directory not found")
            self.log_message("Please run preprocessing first: python scripts/preprocess.py --dataset <dataset_name>")
            
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
            self.wandb_project_var.set(f"har-sweeps-{dataset_name}")
            self.log_message(f"✓ Updated WandB project to: har-sweeps-{dataset_name}")
                
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
            
    def browse_sweep_config(self):
        """Browse for sweep config file"""
        filename = filedialog.askopenfilename(
            title="Select Sweep Config file",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.sweep_config_file_var.set(filename)
            
    def create_sweep(self):
        """Create a new sweep"""
        try:
            # Validate required fields
            required_fields = ["shards_glob", "stats", "method", "metric", "goal"]
            missing_fields = [field for field in required_fields if not self.base_config.get(field)]
            
            if missing_fields:
                self.log_message(f"ERROR: Missing required fields: {missing_fields}")
                return
                
            # Create sweep config
            sweep_config = self.create_sweep_config_dict()
            
            # Save sweep config
            sweep_config_file = Path("artifacts/sweep_configs") / f"sweep_{int(time.time())}.yaml"
            sweep_config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sweep_config_file, 'w') as f:
                yaml.dump(sweep_config, f, default_flow_style=False)
                
            self.sweep_config_file_var.set(str(sweep_config_file))
            self.log_message(f"Created sweep config: {sweep_config_file}")
            
            # Create base experiment config
            base_config_file = Path("artifacts/sweep_configs") / f"base_{int(time.time())}.yaml"
            with open(base_config_file, 'w') as f:
                yaml.dump(self.base_config, f, default_flow_style=False)
                
            self.log_message(f"Created base config: {base_config_file}")
            
            # Create sweep using run_sweep_new.py
            cmd = [
                "python", "scripts/run_sweep_new.py",
                "--action", "create",
                "--config", str(base_config_file),
                "--sweep_config", str(sweep_config_file),
                "--project", self.base_config["wandb_project"]
            ]
            
            self.log_message("Creating sweep...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_message("Sweep created successfully!")
                self.log_message(result.stdout)
                
                # Extract sweep ID from output
                for line in result.stdout.split('\n'):
                    if 'sweep_id' in line.lower() or 'sweep id' in line.lower():
                        sweep_id = line.split(':')[-1].strip()
                        self.sweep_id_var.set(sweep_id)
                        break
            else:
                self.log_message(f"ERROR creating sweep: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"ERROR creating sweep: {str(e)}")
            
    def run_sweep(self):
        """Run the sweep"""
        try:
            # Validate required fields
            if not self.sweep_config_file_var.get():
                self.log_message("ERROR: Please create or load a sweep config first")
                return
                
            # Create base experiment config if needed
            base_config_file = Path("artifacts/sweep_configs") / f"base_{int(time.time())}.yaml"
            with open(base_config_file, 'w') as f:
                yaml.dump(self.base_config, f, default_flow_style=False)
                
            # Run sweep
            cmd = [
                "python", "scripts/run_sweep.py",
                "--action", "run",
                "--config", str(base_config_file),
                "--project", self.base_config["wandb_project"],
                "--count", str(self.sweep_config["sweep_count"])
            ]
            
            # Add sweep config file
            if self.sweep_config_file_var.get():
                cmd.extend(["--sweep_config", self.sweep_config_file_var.get()])
            
            # Add sweep_id if provided
            sweep_id = self.sweep_id_var.get().strip()
            if sweep_id:
                cmd.extend(["--sweep_id", sweep_id])
            else:
                # If no sweep_id provided, we need to create a sweep first
                self.log_message("ERROR: --sweep_id is required for run action.")
                self.log_message("Please either:")
                self.log_message("1. Create a new sweep first using the 'Create Sweep' button, or")
                self.log_message("2. Enter an existing sweep ID in the 'Sweep ID' field")
                return
                
            self.log_message("Starting sweep...")
            self.log_message(f"Command: {' '.join(cmd)}")
            
            # Start sweep in separate thread
            thread = threading.Thread(target=self._run_sweep_thread, args=(cmd,))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.log_message(f"ERROR running sweep: {str(e)}")
            
    def _run_sweep_thread(self, cmd):
        """Run sweep in separate thread"""
        try:
            process = subprocess.Popen(
                cmd,
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
            self.log_message(f"Sweep completed with exit code: {process.returncode}")
            
            if process.returncode != 0:
                self.log_message("")
                self.log_message("TROUBLESHOOTING:")
                self.log_message("1. Make sure you're logged into W&B: wandb login")
                self.log_message("2. Check if the project exists in your W&B dashboard")
                self.log_message("3. Try creating the project manually in W&B first")
                self.log_message("4. Or use a different project name")
                self.log_message("5. Verify the sweep_id exists in the specified project")
            
        except Exception as e:
            self.log_message(f"ERROR in sweep thread: {str(e)}")
            
    def create_sweep_config_dict(self):
        """Create sweep configuration dictionary"""
        # Get selected models
        selected_models = []
        if self.model_cnn_tcn_var.get():
            selected_models.append("cnn_tcn")
        if self.model_cnn_bilstm_var.get():
            selected_models.append("cnn_bilstm")
            
        if not selected_models:
            selected_models = ["cnn_tcn"]  # Default
            
        # Create sweep config
        sweep_config = {
            "method": self.sweep_config["method"],
            "metric": {
                "name": self.sweep_config["metric"],
                "goal": self.sweep_config["goal"]
            },
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": float(self.sweep_config["lr_start"]),
                    "max": float(self.sweep_config["lr_end"])
                },
                "batch_size": {
                    "values": [int(x.strip()) for x in self.sweep_config["batch_sizes"].split(",")]
                },
                "epochs": {
                    "values": [int(x.strip()) for x in self.sweep_config["epochs"].split(",")]
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": float(self.sweep_config["dropout_start"]),
                    "max": float(self.sweep_config["dropout_end"])
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": float(self.sweep_config["weight_decay_start"]),
                    "max": float(self.sweep_config["weight_decay_end"])
                },
                "model": {
                    "values": selected_models
                }
            }
        }
        
        return sweep_config
        
    def load_base_config(self):
        """Load base configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Base Configuration",
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
                        
                self.log_message(f"Loaded base configuration from {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR loading base config: {str(e)}")
                
    def load_sweep_config(self):
        """Load sweep configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Sweep Configuration",
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
                
                # Handle special cases for sweep configuration
                if "sweep_id" in config:
                    self.sweep_id_var.set(config["sweep_id"])
                if "method" in config:
                    self.method_var.set(config["method"])
                if "metric" in config:
                    self.metric_var.set(config["metric"])
                if "goal" in config:
                    self.goal_var.set(config["goal"])
                
                # Set default values for missing required fields
                if not self.method_var.get():
                    self.method_var.set("bayes")
                if not self.metric_var.get():
                    self.metric_var.set("val/accuracy")
                if not self.goal_var.get():
                    self.goal_var.set("maximize")
                        
                self.sweep_config_file_var.set(filename)
                self.log_message(f"Loaded sweep configuration from {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR loading sweep config: {str(e)}")
                
    def save_base_config(self):
        """Save base configuration to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Base Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        yaml.dump(self.base_config, f, default_flow_style=False)
                    else:
                        json.dump(self.base_config, f, indent=2)
                        
                self.log_message(f"Saved base configuration to {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR saving base config: {str(e)}")
                
    def save_sweep_config(self):
        """Save sweep configuration to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Sweep Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                sweep_config = self.create_sweep_config_dict()
                with open(filename, 'w') as f:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        yaml.dump(sweep_config, f, default_flow_style=False)
                    else:
                        json.dump(sweep_config, f, indent=2)
                        
                self.log_message(f"Saved sweep configuration to {filename}")
                
            except Exception as e:
                self.log_message(f"ERROR saving sweep config: {str(e)}")
                
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
    app = SweepConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
