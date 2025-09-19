#!/usr/bin/env python3
"""
Config-based Experiment GUI
Simple GUI for running experiments from YAML configuration files.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import sys
from pathlib import Path
import threading
import queue
import os

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ConfigExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Config-based Experiment Runner")
        self.root.geometry("800x600")
        
        # Queue for thread communication
        self.log_queue = queue.Queue()
        self.running_process = None
        
        # Create GUI
        self.create_widgets()
        
        # Start checking log queue
        self.check_log_queue()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Config-based Experiment Runner", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Configuration file selection
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=15)
        config_frame.pack(fill=tk.X, pady=5)
        
        # Config file selection
        ttk.Label(config_frame, text="Experiment Config:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.config_var = tk.StringVar()
        config_entry = ttk.Entry(config_frame, textvariable=self.config_var, width=60)
        config_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        ttk.Button(config_frame, text="Browse", 
                  command=self.browse_config).grid(row=0, column=2, padx=5, pady=5)
        
        # Sweep configuration
        sweep_frame = ttk.LabelFrame(main_frame, text="Sweep Configuration (Optional)", padding=15)
        sweep_frame.pack(fill=tk.X, pady=5)
        
        # Sweep checkbox
        self.use_sweep_var = tk.BooleanVar()
        ttk.Checkbutton(sweep_frame, text="Use Weights & Biases Sweep", 
                       variable=self.use_sweep_var,
                       command=self.toggle_sweep_config).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Sweep config file selection
        ttk.Label(sweep_frame, text="Sweep Config:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sweep_config_var = tk.StringVar()
        self.sweep_config_entry = ttk.Entry(sweep_frame, textvariable=self.sweep_config_var, width=60)
        self.sweep_config_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.sweep_config_entry.config(state='disabled')
        
        ttk.Button(sweep_frame, text="Browse", 
                  command=self.browse_sweep_config).grid(row=1, column=2, padx=5, pady=5)
        
        # Example configs frame
        examples_frame = ttk.LabelFrame(main_frame, text="Example Configurations", padding=15)
        examples_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(examples_frame, text="Quick start with example configurations:").pack(anchor=tk.W, pady=(0, 10))
        
        examples_buttons = ttk.Frame(examples_frame)
        examples_buttons.pack(fill=tk.X)
        
        ttk.Button(examples_buttons, text="HAR Experiment", 
                  command=self.load_har_example).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(examples_buttons, text="HAR Sweep", 
                  command=self.load_har_sweep_example).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(examples_buttons, text="Color Coding HAR", 
                  command=self.load_color_coding_example).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(examples_buttons, text="View Configs", 
                  command=self.view_configs).pack(side=tk.LEFT)
        
        # Command display
        command_frame = ttk.LabelFrame(main_frame, text="Generated Command", padding=15)
        command_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.command_text = scrolledtext.ScrolledText(command_frame, height=6, wrap=tk.WORD)
        self.command_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Generate Command", 
                  command=self.generate_command).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Experiment", 
                  command=self.run_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Experiment", 
                  command=self.stop_experiment).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).pack(anchor=tk.W, pady=5)
        
        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=15)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log control buttons
        log_button_frame = ttk.Frame(log_frame)
        log_button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_button_frame, text="Clear Logs", 
                  command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_button_frame, text="Save Logs", 
                  command=self.save_logs).pack(side=tk.LEFT, padx=5)
    
    def toggle_sweep_config(self):
        """Enable/disable sweep configuration based on checkbox"""
        if self.use_sweep_var.get():
            self.sweep_config_entry.config(state='normal')
        else:
            self.sweep_config_entry.config(state='disabled')
            self.sweep_config_var.set("")
    
    def browse_config(self):
        """Browse for experiment configuration file"""
        filename = filedialog.askopenfilename(
            title="Select Experiment Configuration",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "configs")
        )
        if filename:
            self.config_var.set(filename)
            self.generate_command()
    
    def browse_sweep_config(self):
        """Browse for sweep configuration file"""
        filename = filedialog.askopenfilename(
            title="Select Sweep Configuration",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "configs")
        )
        if filename:
            self.sweep_config_var.set(filename)
            self.generate_command()
    
    def load_har_example(self):
        """Load HAR experiment example configuration"""
        example_config = REPO_ROOT / "configs" / "har_experiment_example.yaml"
        if example_config.exists():
            self.config_var.set(str(example_config))
            self.generate_command()
            self.log_message(f"Loaded HAR experiment example: {example_config}")
        else:
            messagebox.showerror("Error", f"Example config not found: {example_config}")
    
    def load_har_sweep_example(self):
        """Load HAR sweep example configuration"""
        example_config = REPO_ROOT / "configs" / "har_experiment_example.yaml"
        sweep_config = REPO_ROOT / "configs" / "har_sweep_example.yaml"
        
        if example_config.exists() and sweep_config.exists():
            self.config_var.set(str(example_config))
            self.sweep_config_var.set(str(sweep_config))
            self.use_sweep_var.set(True)
            self.toggle_sweep_config()
            self.generate_command()
            self.log_message(f"Loaded HAR sweep example: {example_config} + {sweep_config}")
        else:
            messagebox.showerror("Error", f"Example configs not found: {example_config} or {sweep_config}")
    
    def load_color_coding_example(self):
        """Load color-coding HAR example configuration"""
        example_config = REPO_ROOT / "configs" / "color_coding_har_experiment.yaml"
        
        if example_config.exists():
            self.config_var.set(str(example_config))
            self.use_sweep_var.set(False)
            self.toggle_sweep_config()
            self.generate_command()
            self.log_message(f"Loaded color-coding HAR example: {example_config}")
        else:
            messagebox.showerror("Error", f"Color-coding config not found: {example_config}")
    
    def view_configs(self):
        """Open configs directory in file explorer"""
        try:
            import platform
            system = platform.system()
            
            configs_dir = REPO_ROOT / "configs"
            
            if system == "Windows":
                subprocess.Popen(["explorer", str(configs_dir)])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", str(configs_dir)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(configs_dir)])
                
            self.log_message("Opened configs directory")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open configs directory: {str(e)}")
    
    def generate_command(self):
        """Generate command line from current configuration"""
        config_file = self.config_var.get()
        if not config_file:
            self.command_text.delete(1.0, tk.END)
            self.command_text.insert(1.0, "Please select a configuration file first.")
            return
        
        # Build command
        cmd_parts = ["python", "scripts/run_experiment_from_config.py", config_file]
        
        if self.use_sweep_var.get():
            sweep_config = self.sweep_config_var.get()
            if not sweep_config:
                self.command_text.delete(1.0, tk.END)
                self.command_text.insert(1.0, "Please select a sweep configuration file when using sweeps.")
                return
            cmd_parts.extend(["--sweep", "--sweep-config", sweep_config])
        
        command = " ".join(cmd_parts)
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(1.0, command)
    
    def run_experiment(self):
        """Run the experiment in a separate thread"""
        if self.running_process:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
        
        command = self.command_text.get(1.0, tk.END).strip()
        if not command or "Please select" in command:
            messagebox.showwarning("Warning", "Please generate a valid command first!")
            return
        
        self.status_var.set("Running...")
        self.log_message(f"Starting experiment: {command}")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_experiment_thread, args=(command,))
        thread.daemon = True
        thread.start()
    
    def _run_experiment_thread(self, command):
        """Run experiment in separate thread"""
        try:
            # Change to repo directory
            os.chdir(REPO_ROOT)
            
            # Run the command
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
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ConfigExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
