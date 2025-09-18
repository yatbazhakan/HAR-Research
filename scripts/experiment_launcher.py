"""
HAR Experiment Launcher GUI

This GUI provides a simple launcher for different HAR experiment tools:
- Training GUI: For single experiment runs
- Sweep GUI: For hyperparameter tuning
- Quick Tests: For running predefined test suites
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ExperimentLauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HAR Experiment Launcher")
        self.root.geometry("600x400")
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="HAR Experiment Tools", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Choose the tool you want to use for your HAR experiments:",
                              font=("Arial", 10))
        desc_label.pack(pady=(0, 20))
        
        # Tool selection frame
        tool_frame = ttk.Frame(main_frame)
        tool_frame.pack(fill=tk.X, pady=10)
        
        # Training GUI button
        training_frame = ttk.LabelFrame(tool_frame, text="Single Experiment Training", padding=15)
        training_frame.pack(fill=tk.X, pady=5)
        
        training_desc = ttk.Label(training_frame, 
                                 text="Configure and run individual HAR experiments with custom hyperparameters")
        training_desc.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(training_frame, text="Open Training GUI", 
                  command=self.open_training_gui,
                  style="Accent.TButton").pack(anchor=tk.W)
        
        # Sweep GUI button
        sweep_frame = ttk.LabelFrame(tool_frame, text="Hyperparameter Sweep", padding=15)
        sweep_frame.pack(fill=tk.X, pady=5)
        
        sweep_desc = ttk.Label(sweep_frame, 
                              text="Run Weights & Biases sweeps for automated hyperparameter optimization")
        sweep_desc.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(sweep_frame, text="Open Sweep GUI", 
                  command=self.open_sweep_gui,
                  style="Accent.TButton").pack(anchor=tk.W)
        
        # Model Builder button
        builder_frame = ttk.LabelFrame(tool_frame, text="Neural Network Builder", padding=15)
        builder_frame.pack(fill=tk.X, pady=5)
        
        builder_desc = ttk.Label(builder_frame, 
                                text="Build neural network architectures with an intuitive interface")
        builder_desc.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(builder_frame, text="Open Model Builder", 
                  command=self.open_model_builder,
                  style="Accent.TButton").pack(anchor=tk.W)
        
        # Quick Tests button
        tests_frame = ttk.LabelFrame(tool_frame, text="Quick Tests", padding=15)
        tests_frame.pack(fill=tk.X, pady=5)
        
        tests_desc = ttk.Label(tests_frame, 
                              text="Run predefined test suites for UCI-HAR, PAMAP2, and MHEALTH datasets")
        tests_desc.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(tests_frame, text="Run Quick Tests", 
                  command=self.run_quick_tests,
                  style="Accent.TButton").pack(anchor=tk.W)
        
        # Config-based Experiments button
        config_frame = ttk.LabelFrame(tool_frame, text="Config-based Experiments", padding=15)
        config_frame.pack(fill=tk.X, pady=5)
        
        config_desc = ttk.Label(config_frame, 
                               text="Run experiments directly from YAML configuration files")
        config_desc.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(config_frame, text="Run from Config", 
                  command=self.run_from_config,
                  style="Accent.TButton").pack(anchor=tk.W)
        
        # Command Line Tools section
        cli_frame = ttk.LabelFrame(tool_frame, text="Command Line Tools", padding=15)
        cli_frame.pack(fill=tk.X, pady=5)
        
        cli_desc = ttk.Label(cli_frame, 
                            text="Access command line tools for advanced users")
        cli_desc.pack(anchor=tk.W, pady=(0, 10))
        
        cli_buttons = ttk.Frame(cli_frame)
        cli_buttons.pack(anchor=tk.W)
        
        ttk.Button(cli_buttons, text="Open Terminal", 
                  command=self.open_terminal).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cli_buttons, text="View Scripts", 
                  command=self.view_scripts).pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                     font=("Arial", 9), foreground="green")
        self.status_label.pack(anchor=tk.W)
        
    def open_training_gui(self):
        """Open the training GUI"""
        try:
            self.update_status("Opening Training GUI...")
            
            # Check if training_gui.py exists
            training_script = REPO_ROOT / "scripts" / "training_gui.py"
            if not training_script.exists():
                messagebox.showerror("Error", f"Training GUI script not found: {training_script}")
                return
                
            # Launch training GUI
            subprocess.Popen([sys.executable, str(training_script)], 
                           cwd=str(REPO_ROOT))
            
            self.update_status("Training GUI opened successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Training GUI: {str(e)}")
            self.update_status("Error opening Training GUI")
            
    def open_sweep_gui(self):
        """Open the sweep GUI"""
        try:
            self.update_status("Opening Sweep GUI...")
            
            # Check if sweep_gui.py exists
            sweep_script = REPO_ROOT / "scripts" / "sweep_gui.py"
            if not sweep_script.exists():
                messagebox.showerror("Error", f"Sweep GUI script not found: {sweep_script}")
                return
                
            # Launch sweep GUI
            subprocess.Popen([sys.executable, str(sweep_script)], 
                           cwd=str(REPO_ROOT))
            
            self.update_status("Sweep GUI opened successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Sweep GUI: {str(e)}")
            self.update_status("Error opening Sweep GUI")
            
    def open_model_builder(self):
        """Open the neural network model builder GUI"""
        try:
            self.update_status("Opening Model Builder...")
            
            # Check if simple_model_builder_gui.py exists
            builder_script = REPO_ROOT / "scripts" / "simple_model_builder_gui.py"
            if not builder_script.exists():
                messagebox.showerror("Error", f"Model Builder script not found: {builder_script}")
                return
                
            # Launch model builder GUI
            subprocess.Popen([sys.executable, str(builder_script)], 
                           cwd=str(REPO_ROOT))
            
            self.update_status("Model Builder opened successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Model Builder: {str(e)}")
            self.update_status("Error opening Model Builder")
            
            
    def run_quick_tests(self):
        """Run quick tests"""
        try:
            self.update_status("Running quick tests...")
            
            # Check if run_quick_tests.sh exists
            test_script = REPO_ROOT / "scripts" / "run_quick_tests.sh"
            if not test_script.exists():
                messagebox.showerror("Error", f"Quick tests script not found: {test_script}")
                return
                
            # Ask for confirmation
            result = messagebox.askyesno("Confirm", 
                                       "This will run comprehensive tests on UCI-HAR dataset. "
                                       "This may take a while. Continue?")
            if not result:
                self.update_status("Quick tests cancelled")
                return
                
            # Launch quick tests in tmux
            tmux_cmd = f"tmux new-session -d -s har_quick_tests 'bash {test_script}'"
            subprocess.run(tmux_cmd, shell=True, cwd=str(REPO_ROOT))
            
            messagebox.showinfo("Info", 
                              "Quick tests started in tmux session 'har_quick_tests'.\n"
                              "Use 'tmux attach -t har_quick_tests' to view progress.")
            
            self.update_status("Quick tests started in tmux session")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run quick tests: {str(e)}")
            self.update_status("Error running quick tests")
            
    def open_terminal(self):
        """Open terminal in project directory"""
        try:
            self.update_status("Opening terminal...")
            
            import platform
            system = platform.system()
            
            if system == "Windows":
                subprocess.Popen(["cmd"], cwd=str(REPO_ROOT))
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Terminal", str(REPO_ROOT)])
            else:  # Linux
                subprocess.Popen(["gnome-terminal", "--working-directory", str(REPO_ROOT)])
                
            self.update_status("Terminal opened")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open terminal: {str(e)}")
            self.update_status("Error opening terminal")
            
    def run_from_config(self):
        """Run experiment from configuration file"""
        try:
            self.update_status("Opening config-based experiment runner...")
            
            # Check if config_experiment_gui.py exists
            config_gui_script = REPO_ROOT / "scripts" / "config_experiment_gui.py"
            if not config_gui_script.exists():
                messagebox.showerror("Error", f"Config GUI script not found: {config_gui_script}")
                return
                
            # Launch config GUI
            subprocess.Popen([sys.executable, str(config_gui_script)], 
                           cwd=str(REPO_ROOT))
            
            self.update_status("Config-based experiment runner opened")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open config runner: {str(e)}")
            self.update_status("Error opening config runner")
            
    def view_scripts(self):
        """Open scripts directory in file explorer"""
        try:
            self.update_status("Opening scripts directory...")
            
            import platform
            system = platform.system()
            
            scripts_dir = REPO_ROOT / "scripts"
            
            if system == "Windows":
                subprocess.Popen(["explorer", str(scripts_dir)])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", str(scripts_dir)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(scripts_dir)])
                
            self.update_status("Scripts directory opened")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open scripts directory: {str(e)}")
            self.update_status("Error opening scripts directory")
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure accent button style
    style.configure("Accent.TButton", 
                   font=("Arial", 10, "bold"),
                   padding=(10, 5))
    
    app = ExperimentLauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
