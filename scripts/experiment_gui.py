#!/usr/bin/env python3
"""
HAR Experiment Configuration GUI
Allows configuring experiments, exporting YAML configs, and running directly.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml
import subprocess
import sys
from pathlib import Path
import threading
import queue
import os
import glob
import time

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class ExperimentConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAR Experiment Configuration")
        self.root.geometry("900x800")
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scan for available datasets and files
        self.scan_available_datasets()
        
        # Create tabs
        self.create_config_tab()
        self.create_run_tab()
        self.create_log_tab()
        
        # Queue for thread communication
        self.log_queue = queue.Queue()
        self.running_process = None
        self.tmux_session = None
        
    def scan_available_datasets(self):
        """Scan for available datasets and files"""
        self.available_datasets = {}
        self.available_stats = {}
        self.available_folds = {}
        
        # Scan for preprocessed datasets
        preprocessed_dir = REPO_ROOT / "artifacts" / "preprocessed"
        if preprocessed_dir.exists():
            for dataset_dir in preprocessed_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    shard_files = list(dataset_dir.glob("*.npz"))
                    if shard_files:
                        self.available_datasets[dataset_name] = {
                            "shards_glob": str(dataset_dir / "*.npz"),
                            "count": len(shard_files)
                        }
        
        # Scan for stats files
        stats_dir = REPO_ROOT / "artifacts" / "norm_stats"
        if stats_dir.exists():
            for stats_file in stats_dir.glob("*.json"):
                dataset_name = stats_file.stem
                self.available_stats[dataset_name] = str(stats_file)
        
        # Scan for fold files
        folds_dir = REPO_ROOT / "artifacts" / "folds"
        if folds_dir.exists():
            for dataset_fold_dir in folds_dir.iterdir():
                if dataset_fold_dir.is_dir():
                    dataset_name = dataset_fold_dir.name
                    fold_files = list(dataset_fold_dir.glob("loso_fold_*.json"))
                    if fold_files:
                        self.available_folds[dataset_name] = {
                            "files": [str(f) for f in fold_files],
                            "count": len(fold_files)
                        }
        
        print(f"Found datasets: {list(self.available_datasets.keys())}")
        print(f"Found stats: {list(self.available_stats.keys())}")
        print(f"Found folds: {list(self.available_folds.keys())}")
        
    def create_config_tab(self):
        """Create the configuration tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configuration variables
        self.config_vars = {}
        
        # Dataset settings
        dataset_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Settings", padding=10)
        dataset_frame.pack(fill=tk.X, pady=5)
        
        # Dataset selection
        self.create_dataset_field(dataset_frame, "dataset", "Dataset", 0)
        self.create_field(dataset_frame, "shards_glob", "Shards Glob Pattern", 
                         "artifacts/preprocessed/uci_har/*.npz", 1)
        self.create_field(dataset_frame, "stats", "Stats JSON Path", 
                         "artifacts/norm_stats/uci_har.json", 2)
        
        # Add generate stats button
        stats_button_frame = ttk.Frame(dataset_frame)
        stats_button_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(stats_button_frame, text="Generate Stats", 
                  command=self.generate_stats).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(stats_button_frame, text="Check Stats", 
                  command=self.check_stats).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(stats_button_frame, text="Refresh Datasets", 
                  command=self.refresh_datasets).pack(side=tk.LEFT)
        
        # Basic settings
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Model Settings", padding=10)
        basic_frame.pack(fill=tk.X, pady=5)
        
        self.create_field(basic_frame, "model", "Model", 
                         "cnn_tcn", 0, combobox_values=["cnn_tcn", "cnn_bilstm"])
        self.create_field(basic_frame, "epochs", "Epochs", "10", 1, field_type="int")
        self.create_field(basic_frame, "batch_size", "Batch Size", "128", 2, field_type="int")
        self.create_field(basic_frame, "lr", "Learning Rate", "0.001", 3, field_type="float")
        
        # Cross-validation settings
        cv_frame = ttk.LabelFrame(scrollable_frame, text="Cross-Validation", padding=10)
        cv_frame.pack(fill=tk.X, pady=5)
        
        self.create_field(cv_frame, "cv", "CV Mode", "holdout", 0, 
                         combobox_values=["fold_json", "holdout", "kfold", "loso"])
        self.create_fold_field(cv_frame, "fold_json", "Fold JSON Path", "", 1)
        self.create_field(cv_frame, "holdout_test_ratio", "Holdout Test Ratio", "0.2", 2, field_type="float")
        self.create_field(cv_frame, "holdout_val_ratio", "Holdout Val Ratio", "0.1", 3, field_type="float")
        self.create_field(cv_frame, "kfold_k", "K-Fold K", "5", 4, field_type="int")
        self.create_field(cv_frame, "kfold_idx", "K-Fold Index", "0", 5, field_type="int")
        
        # Advanced settings
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Settings", padding=10)
        advanced_frame.pack(fill=tk.X, pady=5)
        
        self.create_field(advanced_frame, "num_workers", "Num Workers", "4", 0, field_type="int")
        self.create_field(advanced_frame, "plot_dir", "Plot Directory", "artifacts/plots", 1)
        self.create_field(advanced_frame, "class_names", "Class Names (comma-separated)", 
                         "Walking,Upstairs,Downstairs,Sitting,Standing,Laying", 2)
        
        # Checkboxes
        self.create_checkbox(advanced_frame, "calibrate", "Enable Calibration", 3)
        self.create_checkbox(advanced_frame, "amp", "Mixed Precision (AMP)", 4)
        self.create_checkbox(advanced_frame, "wandb", "Enable WandB Logging", 5)
        self.create_checkbox(advanced_frame, "use_tmux", "Run in Tmux Session", 6)
        
        # WandB settings
        wandb_frame = ttk.LabelFrame(scrollable_frame, text="WandB Settings", padding=10)
        wandb_frame.pack(fill=tk.X, pady=5)
        
        self.create_field(wandb_frame, "wandb_project", "WandB Project", "har-baselines", 0)
        self.create_field(wandb_frame, "wandb_run", "WandB Run Name", "", 1)
        
        # WandB Sweep settings
        sweep_frame = ttk.LabelFrame(scrollable_frame, text="WandB Sweep", padding=10)
        sweep_frame.pack(fill=tk.X, pady=5)
        
        self.create_checkbox(sweep_frame, "use_sweep", "Use Sweep", 0)
        self.create_field(sweep_frame, "sweep_id", "Sweep ID", "", 1)
        self.create_field(sweep_frame, "sweep_count", "Sweep Count", "10", 2, "int")
        self.create_field(sweep_frame, "sweep_config", "Sweep Config File", "", 3)
        
        # Sweep parameter configuration
        sweep_params_frame = ttk.LabelFrame(sweep_frame, text="Sweep Parameters", padding=5)
        sweep_params_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
        
        # Method selection
        ttk.Label(sweep_params_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(sweep_params_frame, "sweep_method", "Method", "bayes", 0, combobox_values=["bayes", "random", "grid"])
        
        # Metric configuration
        ttk.Label(sweep_params_frame, text="Metric:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(sweep_params_frame, "sweep_metric", "Metric", "val/accuracy", 1, combobox_values=["val/accuracy", "val/f1", "val/precision", "val/recall"])
        
        # Goal selection
        ttk.Label(sweep_params_frame, text="Goal:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(sweep_params_frame, "sweep_goal", "Goal", "maximize", 2, combobox_values=["maximize", "minimize"])
        
        # Parameter ranges
        params_frame = ttk.Frame(sweep_params_frame)
        params_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
        
        # Learning rate range
        ttk.Label(params_frame, text="lr:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "sweep_lr_start", "min", "1e-5", 0, "float", column_offset=1)
        self.create_field(params_frame, "sweep_lr_end", "max", "1e-1", 0, "float", column_offset=3)
        
        # Batch size values
        ttk.Label(params_frame, text="batch sizes:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "sweep_batch_sizes", "", "16,32,64,128", 1, column_offset=1)
        
        # Epochs values
        ttk.Label(params_frame, text="epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "sweep_epochs", "", "50,100,150,200", 2, column_offset=1)
        
        # Model selection
        ttk.Label(params_frame, text="model:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        model_frame = ttk.Frame(params_frame)
        model_frame.grid(row=3, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=2)
        
        self.create_checkbox(model_frame, "sweep_model_cnn_tcn", "CNN-TCN", 0)
        self.create_checkbox(model_frame, "sweep_model_cnn_bilstm", "CNN-BiLSTM", 1)
        
        # Dropout range
        ttk.Label(params_frame, text="dropout:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "sweep_dropout_start", "min", "0.1", 4, "float", column_offset=1)
        self.create_field(params_frame, "sweep_dropout_end", "max", "0.5", 4, "float", column_offset=3)
        
        # Weight decay range
        ttk.Label(params_frame, text="weight decay:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.create_field(params_frame, "sweep_wd_start", "min", "1e-6", 5, "float", column_offset=1)
        self.create_field(params_frame, "sweep_wd_end", "max", "1e-2", 5, "float", column_offset=3)
        
        # Sweep control buttons
        sweep_button_frame = ttk.Frame(sweep_frame)
        sweep_button_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
        
        ttk.Button(sweep_button_frame, text="Load Sweep Config", command=self.load_sweep_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(sweep_button_frame, text="Export Sweep Config", command=self.export_sweep_config).pack(side=tk.LEFT, padx=5)
        
        # Tmux settings
        tmux_frame = ttk.LabelFrame(scrollable_frame, text="Tmux Settings", padding=10)
        tmux_frame.pack(fill=tk.X, pady=5)
        
        self.create_field(tmux_frame, "tmux_session", "Tmux Session Name", "har-experiment", 0)
        self.create_field(tmux_frame, "tmux_log_dir", "Log Directory", "logs", 1)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Generate Command", command=self.generate_command).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_run_tab(self):
        """Create the run tab"""
        run_frame = ttk.Frame(self.notebook)
        self.notebook.add(run_frame, text="Run Experiment")
        
        # Command display
        ttk.Label(run_frame, text="Generated Command:").pack(anchor=tk.W, padx=5, pady=5)
        
        self.command_text = scrolledtext.ScrolledText(run_frame, height=8, wrap=tk.WORD)
        self.command_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Run buttons
        button_frame = ttk.Frame(run_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Generate Command", command=self.generate_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Experiment", command=self.run_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Experiment", command=self.stop_experiment).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(run_frame, textvariable=self.status_var).pack(anchor=tk.W, padx=5, pady=5)
        
    def create_log_tab(self):
        """Create the log tab"""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Logs")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log control buttons
        log_button_frame = ttk.Frame(log_frame)
        log_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_button_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_button_frame, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
    def create_field(self, parent, key, label, default, row, field_type="str", combobox_values=None, column_offset=0):
        """Create a labeled input field"""
        if label:  # Only create label if provided
            ttk.Label(parent, text=label).grid(row=row, column=0 + column_offset, sticky=tk.W, padx=5, pady=2)
        
        if combobox_values:
            var = tk.StringVar(value=default)
            widget = ttk.Combobox(parent, textvariable=var, values=combobox_values, width=50)
        elif field_type == "int":
            var = tk.StringVar(value=default)
            widget = ttk.Entry(parent, textvariable=var, width=50)
        elif field_type == "float":
            var = tk.StringVar(value=default)
            widget = ttk.Entry(parent, textvariable=var, width=50)
        else:
            var = tk.StringVar(value=default)
            widget = ttk.Entry(parent, textvariable=var, width=50)
            
        widget.grid(row=row, column=1 + column_offset, sticky=tk.W+tk.E, padx=5, pady=2)
        self.config_vars[key] = var
        
    def create_dataset_field(self, parent, key, label, row):
        """Create dataset selection field with dropdown"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        
        var = tk.StringVar()
        dataset_names = list(self.available_datasets.keys())
        if not dataset_names:
            dataset_names = ["No datasets found"]
            
        combobox = ttk.Combobox(parent, textvariable=var, values=dataset_names, width=47)
        combobox.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # Bind selection event to update paths
        def on_dataset_select(event):
            selected = var.get()
            if selected in self.available_datasets:
                # Update shards glob
                self.config_vars["shards_glob"].set(self.available_datasets[selected]["shards_glob"])
                
                # Update stats path - use available or generate default path
                if selected in self.available_stats:
                    self.config_vars["stats"].set(self.available_stats[selected])
                else:
                    # Generate default stats path
                    default_stats_path = f"artifacts/norm_stats/{selected}.json"
                    self.config_vars["stats"].set(default_stats_path)
                    self.log_message(f"Stats file not found for {selected}, will use: {default_stats_path}")
                
                # Update class names based on dataset
                if selected == "uci_har":
                    self.config_vars["class_names"].set("Walking,Upstairs,Downstairs,Sitting,Standing,Laying")
                elif selected == "pamap2":
                    self.config_vars["class_names"].set("Lying,Sitting,Standing,Walking,Running,Cycling,Nordic_walking,Ascending_stairs,Descending_stairs,Vacuum_cleaning,Ironing,Rope_jumping")
                elif selected == "mhealth":
                    self.config_vars["class_names"].set("Standing still,Sitting and relaxing,Lying down,Walking,Climbing stairs,Waist bends forward,Frontal elevation of arms,Knees bending,Cycling,Jumping,Jogging,Brushing teeth")
        
        combobox.bind("<<ComboboxSelected>>", on_dataset_select)
        self.config_vars[key] = var
        
    def create_fold_field(self, parent, key, label, default_value, row):
        """Create fold selection field with dropdown"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        
        var = tk.StringVar(value=default_value)
        fold_files = []
        for dataset, fold_info in self.available_folds.items():
            for fold_file in fold_info["files"]:
                fold_name = Path(fold_file).stem
                fold_files.append(f"{dataset}: {fold_name}")
        
        if not fold_files:
            fold_files = ["No fold files found"]
            
        combobox = ttk.Combobox(parent, textvariable=var, values=fold_files, width=47)
        combobox.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # Bind selection event to update fold path
        def on_fold_select(event):
            selected = var.get()
            if ": " in selected:
                dataset, fold_name = selected.split(": ", 1)
                if dataset in self.available_folds:
                    fold_file = next((f for f in self.available_folds[dataset]["files"] 
                                    if Path(f).stem == fold_name), "")
                    if fold_file:
                        var.set(fold_file)
        
        combobox.bind("<<ComboboxSelected>>", on_fold_select)
        self.config_vars[key] = var
        
    def check_stats(self):
        """Check if stats file exists and is valid"""
        stats_path = self.config_vars["stats"].get()
        if not stats_path:
            self.log_message("No stats path specified")
            return
            
        stats_file = Path(stats_path)
        if stats_file.exists():
            try:
                import json
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                self.log_message(f"✓ Stats file exists: {stats_path}")
                self.log_message(f"  - Contains {len(stats_data.get('splits', {}))} splits")
                self.log_message(f"  - Channels: {len(stats_data.get('mean', []))}")
            except Exception as e:
                self.log_message(f"✗ Stats file corrupted: {e}")
        else:
            self.log_message(f"✗ Stats file not found: {stats_path}")
            self.log_message("  Click 'Generate Stats' to create it")
            
    def generate_stats(self):
        """Generate stats file for the current dataset"""
        config = self.get_config()
        shards_glob = config.get("shards_glob", "")
        stats_path = config.get("stats", "")
        
        if not shards_glob or not stats_path:
            messagebox.showwarning("Warning", "Please select a dataset first!")
            return
            
        # Check if shards exist
        import glob
        shard_files = glob.glob(shards_glob)
        if not shard_files:
            messagebox.showerror("Error", f"No shard files found matching: {shards_glob}")
            return
            
        self.log_message(f"Generating stats for {len(shard_files)} shard files...")
        self.log_message(f"Output: {stats_path}")
        
        # Run stats generation in a separate thread
        thread = threading.Thread(target=self._generate_stats_thread, args=(shards_glob, stats_path))
        thread.daemon = True
        thread.start()
        
    def _generate_stats_thread(self, shards_glob, stats_path):
        """Generate stats in a separate thread"""
        try:
            # Change to repo directory
            os.chdir(REPO_ROOT)
            
            # Create stats directory if it doesn't exist
            stats_file = Path(stats_path)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build command (escape glob patterns for shell)
            escaped_shards = shards_glob.replace('*', '\\*')
            cmd = [
                "python", "scripts/compute_norm_stats.py",
                "--shards_glob", escaped_shards,
                "--output", stats_path,
                "--split", "train"
            ]
            
            self.log_message(f"Running: {' '.join(cmd)}")
            
            # Run command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                self.log_queue.put(line.rstrip())
                
            process.wait()
            
            if process.returncode == 0:
                self.log_queue.put("✓ Stats generation completed successfully!")
                # Rescan available stats
                self.log_queue.put("Rescanning available datasets...")
                self.scan_available_datasets()
                self.log_queue.put("Dataset scan completed")
            else:
                self.log_queue.put(f"✗ Stats generation failed with return code {process.returncode}")
                
        except Exception as e:
            self.log_queue.put(f"Error generating stats: {e}")
            
    def refresh_datasets(self):
        """Refresh the available datasets list"""
        self.log_message("Refreshing available datasets...")
        self.scan_available_datasets()
        self.log_message("Dataset refresh completed")
        
        # Update dataset dropdown if it exists
        if "dataset" in self.config_vars:
            current_selection = self.config_vars["dataset"].get()
            dataset_names = list(self.available_datasets.keys())
            if not dataset_names:
                dataset_names = ["No datasets found"]
            
            # Find the dataset combobox and update its values
            for widget in self.root.winfo_children():
                self._update_combobox_values(widget, "dataset", dataset_names)
                
    def _update_combobox_values(self, widget, key, values):
        """Recursively update combobox values"""
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                if isinstance(child, ttk.Combobox) and hasattr(child, 'cget'):
                    try:
                        if child.cget('textvariable') == str(self.config_vars[key]):
                            child['values'] = values
                            return
                    except:
                        pass
                self._update_combobox_values(child, key, values)
            
    def create_checkbox(self, parent, key, label, row):
        """Create a checkbox field"""
        var = tk.BooleanVar()
        widget = ttk.Checkbutton(parent, text=label, variable=var)
        widget.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.config_vars[key] = var
        
    def get_config(self):
        """Get current configuration as dictionary"""
        config = {}
        for key, var in self.config_vars.items():
            if isinstance(var, tk.BooleanVar):
                config[key] = var.get()
            else:
                value = var.get()
                # Try to convert to appropriate type
                if key in ["epochs", "batch_size", "num_workers", "kfold_k", "kfold_idx"]:
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = value
                elif key in ["lr", "holdout_test_ratio", "holdout_val_ratio"]:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
                else:
                    config[key] = value
        return config
        
    def set_config(self, config):
        """Set configuration from dictionary"""
        for key, value in config.items():
            if key in self.config_vars:
                if isinstance(self.config_vars[key], tk.BooleanVar):
                    self.config_vars[key].set(bool(value))
                else:
                    self.config_vars[key].set(str(value))
                    
    def load_config(self):
        """Load configuration from YAML file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = yaml.safe_load(f)
                self.set_config(config)
                self.log_message(f"Loaded configuration from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
                
    def save_config(self):
        """Save configuration to YAML file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = self.get_config()
                with open(filename, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                self.log_message(f"Saved configuration to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                
    def reset_config(self):
        """Reset configuration to defaults"""
        default_config = {
            "dataset": "uci_har",
            "shards_glob": "artifacts/preprocessed/uci_har/*.npz",
            "stats": "artifacts/norm_stats/uci_har.json",
            "model": "cnn_tcn",
            "epochs": 10,
            "batch_size": 128,
            "lr": 0.001,
            "cv": "holdout",
            "fold_json": "",
            "holdout_test_ratio": 0.2,
            "holdout_val_ratio": 0.1,
            "kfold_k": 5,
            "kfold_idx": 0,
            "num_workers": 4,
            "plot_dir": "artifacts/plots",
            "class_names": "Walking,Upstairs,Downstairs,Sitting,Standing,Laying",
            "calibrate": True,
            "amp": False,
            "wandb": False,
            "use_tmux": False,
            "wandb_project": "har-baselines",
            "wandb_run": "",
            "use_sweep": False,
            "sweep_id": "",
            "sweep_count": "10",
            "sweep_config": "",
            "sweep_method": "bayes",
            "sweep_metric": "val/accuracy",
            "sweep_goal": "maximize",
            "sweep_lr_start": "1e-5",
            "sweep_lr_end": "1e-1",
            "sweep_batch_sizes": "16,32,64,128",
            "sweep_epochs": "50,100,150,200",
            "sweep_model_cnn_tcn": True,
            "sweep_model_cnn_bilstm": True,
            "sweep_dropout_start": "0.1",
            "sweep_dropout_end": "0.5",
            "sweep_wd_start": "1e-6",
            "sweep_wd_end": "1e-2",
            "tmux_session": "har-experiment",
            "tmux_log_dir": "logs"
        }
        self.set_config(default_config)
        self.log_message("Reset to default configuration")
        
    def generate_command(self):
        """Generate command line from current configuration"""
        config = self.get_config()
        
        # Build command
        if config.get("use_tmux", False):
            # Generate tmux command
            tmux_session = config.get("tmux_session", "har-experiment")
            tmux_log_dir = config.get("tmux_log_dir", "logs")
            log_file = f"{tmux_log_dir}/experiment_{tmux_session}_{int(time.time())}.log"
            
            # Create the actual command
            cmd_parts = ["python", "scripts/train_baselines.py"]
        else:
            cmd_parts = ["python", "scripts/train_baselines.py"]
        
        # Add required arguments (escape glob patterns for shell)
        escaped_shards = config["shards_glob"].replace('*', '\\*')
        cmd_parts.extend(["--shards_glob", escaped_shards])
        cmd_parts.extend(["--stats", config["stats"]])
        cmd_parts.extend(["--model", config["model"]])
        cmd_parts.extend(["--epochs", str(config["epochs"])])
        cmd_parts.extend(["--batch_size", str(config["batch_size"])])
        cmd_parts.extend(["--lr", str(config["lr"])])
        
        # Add CV arguments
        cmd_parts.extend(["--cv", config["cv"]])
        if config["cv"] == "fold_json" and config["fold_json"]:
            cmd_parts.extend(["--fold_json", config["fold_json"]])
        elif config["cv"] == "holdout":
            cmd_parts.extend(["--holdout_test_ratio", str(config["holdout_test_ratio"])])
            cmd_parts.extend(["--holdout_val_ratio", str(config["holdout_val_ratio"])])
        elif config["cv"] == "kfold":
            cmd_parts.extend(["--kfold_k", str(config["kfold_k"])])
            cmd_parts.extend(["--kfold_idx", str(config["kfold_idx"])])
            
        # Add optional arguments
        cmd_parts.extend(["--num_workers", str(config["num_workers"])])
        cmd_parts.extend(["--plot_dir", config["plot_dir"]])
        if config["class_names"]:
            cmd_parts.extend(["--class_names", config["class_names"]])
            
        # Add flags
        if config["calibrate"]:
            cmd_parts.append("--calibrate")
        if config["amp"]:
            cmd_parts.append("--amp")
        if config["wandb"]:
            cmd_parts.append("--wandb")
            cmd_parts.extend(["--wandb_project", config["wandb_project"]])
            if config["wandb_run"]:
                cmd_parts.extend(["--wandb_run", config["wandb_run"]])
        
        # Handle sweep functionality
        if config.get("use_sweep", False):
            # For sweeps, we need to use run_sweep.py instead of train_baselines.py
            # Create sweep config if needed
            sweep_config_file = config.get("sweep_config", "")
            if not sweep_config_file:
                sweep_config_file = self.create_sweep_config_from_gui()
                if not sweep_config_file:
                    self.log_message("ERROR: Failed to create sweep configuration")
                    return
            
            # Create base experiment config for the sweep
            base_config_file = self.create_base_experiment_config()
            if not base_config_file:
                self.log_message("ERROR: Failed to create base experiment configuration")
                return
            
            # Replace the command with run_sweep.py
            cmd_parts = ["python", "scripts/run_sweep.py"]
            cmd_parts.extend(["--action", "run"])
            cmd_parts.extend(["--config", f'"{base_config_file}"'])
            cmd_parts.extend(["--sweep_config", f'"{sweep_config_file}"'])
            cmd_parts.extend(["--project", config.get("wandb_project", "har-baselines")])
            cmd_parts.extend(["--count", str(config.get("sweep_count", "10"))])
            
            # If sweep ID exists, add it
            sweep_id = config.get("sweep_id", "")
            if sweep_id:
                cmd_parts.extend(["--sweep_id", sweep_id])
        
        # Finalize command
        if config.get("use_tmux", False):
            # Wrap in tmux command with proper escaping
            base_command = " ".join(cmd_parts)
            tmux_session = config.get("tmux_session", "har-experiment")
            tmux_log_dir = config.get("tmux_log_dir", "logs")
            log_file = f"{tmux_log_dir}/experiment_{tmux_session}_{int(time.time())}.log"
            
            # Create tmux command that runs the experiment and keeps session open
            tmux_command = f'tmux new -s "{tmux_session}" -d "bash -lc \'{base_command} 2>&1 | tee -a {log_file}; ec=$?; echo; echo \\"==> Experiment finished with exit code: $ec\\"; echo \\"Log: {log_file}\\"; echo \\"Keeping session open. Press Ctrl-C to exit, or detach with Ctrl-b d.\\"; exec bash\'"'
            command = tmux_command
        else:
            command = " ".join(cmd_parts)
            
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(1.0, command)
        
    def run_experiment(self):
        """Run the experiment in a separate thread"""
        if self.running_process:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
            
        command = self.command_text.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("Warning", "Please generate a command first!")
            return
            
        config = self.get_config()
        if config.get("use_tmux", False):
            self.status_var.set("Starting tmux session...")
            self.log_message(f"Starting tmux experiment: {command}")
            self.log_message("Note: Experiment will run in detached tmux session")
            self.log_message(f"Attach with: tmux attach -t {config.get('tmux_session', 'har-experiment')}")
        else:
            self.status_var.set("Running...")
            self.log_message(f"Starting experiment: {command}")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_experiment_thread, args=(command, config.get("use_tmux", False)))
        thread.daemon = True
        thread.start()
        
    def _run_experiment_thread(self, command, use_tmux=False):
        """Run experiment in separate thread"""
        try:
            # Change to repo directory
            os.chdir(REPO_ROOT)
            
            if use_tmux:
                # For tmux, just execute the command (it will create the session)
                self.running_process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # For tmux, we just wait for the session to be created
                self.running_process.wait()
                
                if self.running_process.returncode == 0:
                    self.log_queue.put("Tmux session created successfully!")
                    self.log_queue.put("Experiment is running in detached session")
                    self.status_var.set("Running in tmux")
                else:
                    self.log_queue.put(f"Failed to create tmux session: {self.running_process.returncode}")
                    self.status_var.set("Failed")
            else:
                # Regular execution
                self.running_process = subprocess.Popen(
                    command.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Read output line by line
                for line in iter(self.running_process.stdout.readline, ''):
                    self.log_queue.put(line.rstrip())
                    
                self.running_process.wait()
                
                if self.running_process.returncode == 0:
                    self.log_queue.put("Experiment completed successfully!")
                    self.status_var.set("Completed")
                else:
                    self.log_queue.put(f"Experiment failed with return code {self.running_process.returncode}")
                    self.status_var.set("Failed")
                
        except Exception as e:
            self.log_queue.put(f"Error running experiment: {e}")
            self.status_var.set("Error")
        finally:
            self.running_process = None
            
    def stop_experiment(self):
        """Stop the running experiment"""
        if self.running_process:
            self.running_process.terminate()
            self.log_message("Experiment stopped by user")
            self.status_var.set("Stopped")
            self.running_process = None
        else:
            messagebox.showinfo("Info", "No experiment is currently running")
    
    def create_sweep(self):
        """Create a new wandb sweep"""
        try:
            import wandb
        except ImportError:
            self.log_message("ERROR: wandb not installed. Please install with: pip install wandb")
            return
        
        config = self.get_config()
        sweep_config_file = config.get("sweep_config", "")
        
        if not sweep_config_file:
            # Create sweep config from GUI parameters
            sweep_config_file = self.create_sweep_config_from_gui()
            if not sweep_config_file:
                return
        
        if not Path(sweep_config_file).exists():
            self.log_message(f"ERROR: Sweep config file not found: {sweep_config_file}")
            return
        
        try:
            # Initialize wandb
            wandb.login()
            
            # Create sweep
            sweep_id = wandb.sweep(sweep_config_file, project=config["wandb_project"])
            
            # Update GUI with sweep ID
            self.config_vars["sweep_id"].set(sweep_id)
            
            self.log_message(f"Created sweep with ID: {sweep_id}")
            self.log_message(f"Sweep config: {sweep_config_file}")
            self.log_message(f"Project: {config['wandb_project']}")
            
        except Exception as e:
            self.log_message(f"ERROR creating sweep: {e}")
    
    
    def _get_selected_models(self, config):
        """Get list of selected models from checkboxes"""
        models = []
        if config.get("sweep_model_cnn_tcn", False):
            models.append("cnn_tcn")
        if config.get("sweep_model_cnn_bilstm", False):
            models.append("cnn_bilstm")
        return models if models else ["cnn_tcn", "cnn_bilstm"]  # Default to both if none selected
    
    def create_base_experiment_config(self):
        """Create a base experiment configuration file for the sweep"""
        config = self.get_config()
        dataset = config.get("dataset", "uci_har")
        
        # Create base experiment configuration
        base_config = {
            "shards_glob": config.get("shards_glob", ""),
            "stats": config.get("stats", ""),
            "model": config.get("model", "cnn_tcn"),
            "epochs": int(config.get("epochs", "100")),
            "batch_size": int(config.get("batch_size", "32")),
            "lr": float(config.get("lr", "1e-3")),
            "cv": config.get("cv", "fold_json"),
            "fold_json": config.get("fold_json", ""),
            "plot_dir": config.get("plot_dir", "artifacts/plots"),
            "class_names": config.get("class_names", ""),
            "calibrate": config.get("calibrate", True),
            "amp": config.get("amp", False),
            "wandb": config.get("wandb", False),
            "wandb_project": config.get("wandb_project", "har-baselines"),
            "wandb_run": config.get("wandb_run", "")
        }
        
        # Save to file
        base_config_path = f"artifacts/experiments/{dataset}_base_config.yaml"
        Path(base_config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(base_config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        self.log_message(f"Created base experiment config: {base_config_path}")
        return base_config_path
    
    def create_sweep_config_from_gui(self):
        """Create a sweep configuration file from GUI parameters"""
        config = self.get_config()
        dataset = config.get("dataset", "uci_har")
        
        # Parse batch sizes and epochs from comma-separated strings
        try:
            batch_sizes = [int(x.strip()) for x in config.get("sweep_batch_sizes", "16,32,64,128").split(",")]
            epochs = [int(x.strip()) for x in config.get("sweep_epochs", "50,100,150,200").split(",")]
        except ValueError as e:
            self.log_message(f"ERROR parsing batch sizes or epochs: {e}")
            return None
        
        # Create sweep configuration from GUI parameters
        sweep_config = {
            "method": config.get("sweep_method", "bayes"),
            "metric": {
                "name": config.get("sweep_metric", "val/accuracy"),
                "goal": config.get("sweep_goal", "maximize")
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10,
                "eta": 2
            },
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": float(config.get("sweep_lr_start", "1e-5")),
                    "max": float(config.get("sweep_lr_end", "1e-1"))
                },
                "batch_size": {
                    "values": batch_sizes
                },
                "epochs": {
                    "values": epochs
                },
                "model": {
                    "values": self._get_selected_models(config)
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": float(config.get("sweep_dropout_start", "0.1")),
                    "max": float(config.get("sweep_dropout_end", "0.5"))
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": float(config.get("sweep_wd_start", "1e-6")),
                    "max": float(config.get("sweep_wd_end", "1e-2"))
                }
            }
        }
        
        # Save to file
        sweep_config_path = f"artifacts/sweep_configs/{dataset}_sweep_gui.yaml"
        Path(sweep_config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(sweep_config_path, 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)
        
        # Update GUI
        self.config_vars["sweep_config"].set(sweep_config_path)
        
        self.log_message(f"Created sweep config from GUI: {sweep_config_path}")
        return sweep_config_path
    
    def load_sweep_config(self):
        """Load an existing sweep configuration file into the GUI"""
        # Ask user for file location
        file_path = filedialog.askopenfilename(
            title="Load Sweep Configuration",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir="artifacts/sweep_configs"
        )
        
        if not file_path:
            return
        
        try:
            # Load the sweep configuration
            with open(file_path, 'r') as f:
                sweep_config = yaml.safe_load(f)
            
            # Update GUI fields with loaded values
            if "method" in sweep_config:
                self.config_vars["sweep_method"].set(sweep_config["method"])
            
            if "metric" in sweep_config:
                metric = sweep_config["metric"]
                if "name" in metric:
                    self.config_vars["sweep_metric"].set(metric["name"])
                if "goal" in metric:
                    self.config_vars["sweep_goal"].set(metric["goal"])
            
            if "parameters" in sweep_config:
                params = sweep_config["parameters"]
                
                # Learning rate
                if "lr" in params and "min" in params["lr"] and "max" in params["lr"]:
                    self.config_vars["sweep_lr_start"].set(str(params["lr"]["min"]))
                    self.config_vars["sweep_lr_end"].set(str(params["lr"]["max"]))
                
                # Batch sizes
                if "batch_size" in params and "values" in params["batch_size"]:
                    batch_sizes = ",".join(map(str, params["batch_size"]["values"]))
                    self.config_vars["sweep_batch_sizes"].set(batch_sizes)
                
                # Epochs
                if "epochs" in params and "values" in params["epochs"]:
                    epochs = ",".join(map(str, params["epochs"]["values"]))
                    self.config_vars["sweep_epochs"].set(epochs)
                
                # Models
                if "model" in params and "values" in params["model"]:
                    model_values = params["model"]["values"]
                    self.config_vars["sweep_model_cnn_tcn"].set("cnn_tcn" in model_values)
                    self.config_vars["sweep_model_cnn_bilstm"].set("cnn_bilstm" in model_values)
                
                # Dropout
                if "dropout" in params and "min" in params["dropout"] and "max" in params["dropout"]:
                    self.config_vars["sweep_dropout_start"].set(str(params["dropout"]["min"]))
                    self.config_vars["sweep_dropout_end"].set(str(params["dropout"]["max"]))
                
                # Weight decay
                if "weight_decay" in params and "min" in params["weight_decay"] and "max" in params["weight_decay"]:
                    self.config_vars["sweep_wd_start"].set(str(params["weight_decay"]["min"]))
                    self.config_vars["sweep_wd_end"].set(str(params["weight_decay"]["max"]))
            
            # Update the sweep config file path
            self.config_vars["sweep_config"].set(file_path)
            
            self.log_message(f"Loaded sweep configuration from: {file_path}")
            messagebox.showinfo("Success", f"Sweep configuration loaded from:\n{file_path}")
            
        except Exception as e:
            self.log_message(f"ERROR loading sweep config: {e}")
            messagebox.showerror("Error", f"Failed to load sweep configuration:\n{e}")
    
    def export_sweep_config(self):
        """Export the current sweep configuration to a file"""
        config = self.get_config()
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Sweep Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialfile=f"{config.get('dataset', 'uci_har')}_sweep_config.yaml"
        )
        
        if not file_path:
            return
        
        # Create sweep config from GUI parameters
        sweep_config = self.create_sweep_config_dict()
        if not sweep_config:
            return
        
        # Save to file
        with open(file_path, 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)
        
        self.log_message(f"Exported sweep configuration to: {file_path}")
        messagebox.showinfo("Success", f"Sweep configuration exported to:\n{file_path}")
    
    def create_sweep_config_dict(self):
        """Create a sweep configuration dictionary from GUI parameters"""
        config = self.get_config()
        
        # Parse batch sizes and epochs from comma-separated strings
        try:
            batch_sizes = [int(x.strip()) for x in config.get("sweep_batch_sizes", "16,32,64,128").split(",")]
            epochs = [int(x.strip()) for x in config.get("sweep_epochs", "50,100,150,200").split(",")]
        except ValueError as e:
            self.log_message(f"ERROR parsing batch sizes or epochs: {e}")
            return None
        
        # Create sweep configuration from GUI parameters
        sweep_config = {
            "method": config.get("sweep_method", "bayes"),
            "metric": {
                "name": config.get("sweep_metric", "val/accuracy"),
                "goal": config.get("sweep_goal", "maximize")
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10,
                "eta": 2
            },
            "parameters": {
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": float(config.get("sweep_lr_start", "1e-5")),
                    "max": float(config.get("sweep_lr_end", "1e-1"))
                },
                "batch_size": {
                    "values": batch_sizes
                },
                "epochs": {
                    "values": epochs
                },
                "model": {
                    "values": self._get_selected_models(config)
                },
                "dropout": {
                    "distribution": "uniform",
                    "min": float(config.get("sweep_dropout_start", "0.1")),
                    "max": float(config.get("sweep_dropout_end", "0.5"))
                },
                "weight_decay": {
                    "distribution": "log_uniform_values",
                    "min": float(config.get("sweep_wd_start", "1e-6")),
                    "max": float(config.get("sweep_wd_end", "1e-2"))
                }
            }
        }
        
        return sweep_config
    
    def create_default_sweep_config(self):
        """Create a default sweep configuration file"""
        config = self.get_config()
        dataset = config.get("dataset", "uci_har")
        
        # Default sweep configuration
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "val/accuracy",
                "goal": "maximize"
            },
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
                    "values": self._get_selected_models(config)
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
        
        # Save to file
        sweep_config_path = f"artifacts/sweep_configs/{dataset}_sweep.yaml"
        Path(sweep_config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(sweep_config_path, 'w') as f:
            yaml.dump(sweep_config, f, default_flow_style=False)
        
        # Update GUI
        self.config_vars["sweep_config"].set(sweep_config_path)
        
        self.log_message(f"Created default sweep config: {sweep_config_path}")
        return sweep_config_path
            
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
    def clear_logs(self):
        """Clear log text"""
        self.log_text.delete(1.0, tk.END)
        
    def save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")
                
    def check_log_queue(self):
        """Check for new log messages from thread"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_message(message)
        except queue.Empty:
            pass
        # Schedule next check
        self.root.after(100, self.check_log_queue)

def main():
    root = tk.Tk()
    app = ExperimentConfigGUI(root)
    
    # Start checking log queue
    app.check_log_queue()
    
    # Load default config
    app.reset_config()
    app.generate_command()
    
    root.mainloop()

if __name__ == "__main__":
    main()
