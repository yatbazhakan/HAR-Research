#!/usr/bin/env python3
"""
CRITICAL: Neural Network Model Builder GUI

This is a comprehensive GUI tool for building and configuring neural network models
for Human Activity Recognition (HAR) tasks. It provides an intuitive interface for:

Key Features:
- Visual model architecture design
- Layer-by-layer parameter configuration
- Real-time model statistics and validation
- PyTorch code generation
- Model testing and export capabilities
- Support for custom HAR modules and PyTorch layers

Architecture:
- Left Panel: Layer management (add, edit, remove, reorder layers)
- Right Panel: Generated PyTorch code and model statistics
- Bottom: Control buttons for testing, saving, and exporting

CRITICAL: This GUI is essential for:
1. Rapid prototyping of HAR models
2. Educational purposes (understanding model architectures)
3. Experimentation with different layer combinations
4. Generating production-ready PyTorch code

Usage:
    python scripts/simple_model_builder_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import torch
import torch.nn as nn
import sys
import os

# CRITICAL: Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import the model configuration builder
from har.models.model_config_builder import ModelConfigBuilder


class SimpleModelBuilderGUI:
    """
    CRITICAL: Main GUI class for the Neural Network Model Builder
    
    This class manages the entire GUI application including:
    - Model configuration state management
    - Layer management (add, edit, remove, reorder)
    - Real-time code generation and validation
    - Model statistics calculation
    - User interaction handling
    
    The GUI maintains a model_config dictionary that represents the current
    model architecture. This configuration is used to:
    1. Generate PyTorch code
    2. Calculate model statistics
    3. Test model functionality
    4. Export model configurations
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Builder - Professional Interface")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # CRITICAL: Initialize model configuration builder
        # This handles the actual model creation and validation
        self.builder = ModelConfigBuilder()
        
        # CRITICAL: Model configuration state
        # This dictionary represents the current model architecture
        self.model_config = {
            "name": "custom_model",                    # Model name
            "input_shape": [1, 9, 128],               # [batch, features, sequence_length]
            "num_classes": 6,                         # Number of output classes
            "layers": []                              # List of layer configurations
        }
        
        # CRITICAL: Layer counter for unique IDs
        # Ensures each layer has a unique identifier
        self.layer_counter = 0
        
        # Configure modern styling
        self.setup_style()
        
        # Create GUI
        self.create_widgets()
        self.update_code_output()
        
    def setup_style(self):
        """Setup modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors and fonts
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 9), foreground='#7f8c8d')
        
        # Configure buttons
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TButton', font=('Arial', 9))
        style.configure('Danger.TButton', font=('Arial', 9))
        
    def enable_row_resizing(self):
        """Enable row resizing in the treeview"""
        # Bind mouse events for row resizing
        self.layer_tree.bind("<Button-1>", self.on_tree_click)
        self.layer_tree.bind("<B1-Motion>", self.on_tree_drag)
        self.layer_tree.bind("<ButtonRelease-1>", self.on_tree_release)
        
        # Variables for tracking resize operation
        self.resize_start_y = None
        self.resize_item = None
        self.resize_original_height = None
        
    def on_tree_click(self, event):
        """Handle mouse click on treeview"""
        # Check if click is near the bottom edge of an item
        item = self.layer_tree.identify_row(event.y)
        if item:
            # Get item bounds
            bbox = self.layer_tree.bbox(item)
            if bbox:
                item_bottom = bbox[1] + bbox[3]
                # If click is within 5 pixels of bottom edge, start resize
                if abs(event.y - item_bottom) < 5:
                    self.resize_start_y = event.y
                    self.resize_item = item
                    self.resize_original_height = bbox[3]
                    self.layer_tree.configure(cursor="sb_v_double_arrow")
                    return
        
        # Reset resize state
        self.resize_start_y = None
        self.resize_item = None
        self.resize_original_height = None
        self.layer_tree.configure(cursor="")
        
    def on_tree_drag(self, event):
        """Handle mouse drag for resizing"""
        if self.resize_start_y is not None and self.resize_item is not None:
            # Calculate height change
            height_change = event.y - self.resize_start_y
            new_height = max(20, self.resize_original_height + height_change)
            
            # Update treeview height
            current_height = self.layer_tree.cget('height')
            height_ratio = new_height / self.resize_original_height
            new_tree_height = int(current_height * height_ratio)
            self.layer_tree.configure(height=max(5, new_tree_height))
            
    def on_tree_release(self, event):
        """Handle mouse release after resize"""
        self.resize_start_y = None
        self.resize_item = None
        self.resize_original_height = None
        self.layer_tree.configure(cursor="")
        
    def on_layer_double_click(self, event):
        """Handle double-click on layer to edit parameters"""
        self.edit_layer()
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Create main canvas and scrollbar for scrolling
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Store canvas reference for cleanup
        self.canvas = canvas
        
        # Cleanup function
        def cleanup():
            try:
                canvas.unbind_all("<MouseWheel>")
            except:
                pass
            self.root.destroy()
        self.root.protocol("WM_DELETE_WINDOW", cleanup)
        
        # Main container
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Model info
        self.create_model_info_section(main_frame)
        
        # Two-column layout for the rest
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left column - Layer management
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right column - Code output
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create sections in their respective frames
        self.create_layer_management_section(left_frame)
        self.create_code_section(right_frame)
        
        # Status bar
        self.create_status_bar(scrollable_frame)
        
    def create_model_info_section(self, parent):
        """Create model information section"""
        info_frame = ttk.LabelFrame(parent, text="Model Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model name
        ttk.Label(info_frame, text="Model Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_name_var = tk.StringVar(value="custom_model")
        ttk.Entry(info_frame, textvariable=self.model_name_var, width=20).grid(row=0, column=1, padx=5, pady=5)
        
        # Input shape
        ttk.Label(info_frame, text="Input Shape:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.input_shape_var = tk.StringVar(value="[1, 9, 128]")
        ttk.Entry(info_frame, textvariable=self.input_shape_var, width=15).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(info_frame, text="[batch, features, sequence_length]", 
                 font=("Arial", 8), foreground="gray").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        
        # Number of classes
        ttk.Label(info_frame, text="Classes:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_classes_var = tk.StringVar(value="6")
        ttk.Entry(info_frame, textvariable=self.num_classes_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Update button
        ttk.Button(info_frame, text="Update Model Info", 
                  command=self.update_model_info, style='Primary.TButton').grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Model statistics
        self.create_model_stats_section(info_frame)
        
    def create_model_stats_section(self, parent):
        """Create model statistics section"""
        stats_frame = ttk.LabelFrame(parent, text="Model Statistics", padding=10)
        stats_frame.grid(row=2, column=0, columnspan=5, sticky=tk.W+tk.E, pady=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=60, font=('Consolas', 9), 
                                 state=tk.DISABLED, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self, parent):
        """Create status bar with instructions"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_text = "Layer Management: Select a layer and use buttons below to Edit, Remove, Move Up ↑, or Move Down ↓"
        ttk.Label(status_frame, text=status_text, font=('Arial', 9), 
                 foreground='#7f8c8d').pack(anchor=tk.W)
        
    def create_layer_management_section(self, parent):
        """Create layer management section"""
        # Add layer section (top row)
        add_frame = ttk.LabelFrame(parent, text="Add New Layer")
        add_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(add_frame, text="Add New Layer:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Layer type selection
        ttk.Label(add_frame, text="Layer Type:").pack(anchor=tk.W)
        self.layer_type_var = tk.StringVar()
        self.layer_type_combo = ttk.Combobox(add_frame, textvariable=self.layer_type_var, width=25)
        self.layer_type_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Populate layer types (but don't call on_layer_type_change yet)
        self.populate_layer_types()
        
        # Layer parameters frame
        self.params_frame = ttk.Frame(add_frame)
        self.params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Bind layer type change
        self.layer_type_combo.bind('<<ComboboxSelected>>', self.on_layer_type_change)
        
        # Initialize parameters for the first layer type
        self.on_layer_type_change()
        
        # Add layer button
        ttk.Button(add_frame, text="Add Layer", command=self.add_layer, 
                  style='Success.TButton').pack(fill=tk.X, pady=5)
        
        # Layer list section (bottom row)
        list_frame = ttk.LabelFrame(parent, text="Model Layers")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Layer list with treeview for better display
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for layers
        columns = ('Index', 'Type', 'Parameters', 'Output Shape')
        self.layer_tree = ttk.Treeview(list_container, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.layer_tree.heading('Index', text='#')
        self.layer_tree.heading('Type', text='Layer Type')
        self.layer_tree.heading('Parameters', text='Key Parameters')
        self.layer_tree.heading('Output Shape', text='Output Shape')
        
        self.layer_tree.column('Index', width=40, minwidth=40)
        self.layer_tree.column('Type', width=150, minwidth=100)
        self.layer_tree.column('Parameters', width=200, minwidth=150)
        self.layer_tree.column('Output Shape', width=100, minwidth=80)
        
        # Make row heights adjustable
        self.layer_tree.configure(height=8)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.layer_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_container, orient=tk.HORIZONTAL, command=self.layer_tree.xview)
        self.layer_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.layer_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Enable row resizing
        self.enable_row_resizing()
        
        # Enable double-click to edit layers
        self.layer_tree.bind('<Double-1>', self.on_layer_double_click)
        
        # Layer control buttons
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Make buttons more prominent
        ttk.Button(button_frame, text="Edit Layer", command=self.edit_layer, 
                  style='Primary.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Remove Layer", command=self.remove_layer, 
                  style='Danger.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Move Up ↑", command=self.move_layer_up, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Move Down ↓", command=self.move_layer_down, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 5))
        
    def create_code_section(self, parent):
        """Create code output and control section"""
        code_frame = ttk.LabelFrame(parent, text="Generated PyTorch Code (Editable)")
        code_frame.pack(fill=tk.BOTH, expand=True)
        
        # Code text area - make it editable
        self.code_text = tk.Text(code_frame, height=15, wrap=tk.WORD, font=("Consolas", 9), 
                                bg='#f8f9fa', fg='#2c3e50')
        code_scrollbar = ttk.Scrollbar(code_frame, orient=tk.VERTICAL, command=self.code_text.yview)
        self.code_text.configure(yscrollcommand=code_scrollbar.set)
        
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        code_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(control_frame, text="Test Model", command=self.test_model, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Clear All", command=self.clear_all, 
                  style='Danger.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Copy Code", command=self.copy_code).pack(side=tk.LEFT)
        
    def populate_layer_types(self):
        """Populate the layer type combobox"""
        layer_types = []
        
        # PyTorch core layers
        pytorch_layers = [
            "Linear", "Conv1d", "Conv2d", "Conv3d", "LSTM", "GRU", "RNN",
            "BatchNorm1d", "BatchNorm2d", "LayerNorm", "Dropout", "ReLU", 
            "Sigmoid", "Tanh", "Softmax", "Flatten", "MaxPool1d", "AvgPool1d"
        ]
        
        # Custom HAR modules
        har_layers = [
            "ConvBlock1D", "ResidualBlock1D", "DepthwiseSeparableConv1D",
            "SEBlock1D", "InceptionBlock1D", "DilatedConvBlock", "TemporalConvNet",
            "BiDirectionalLSTM", "LayerNormLSTM", "PeepholeLSTM", "IndRNN",
            "ConvLSTM1D", "StackedLSTM", "GRUCell", "QRNN",
            "PositionalEncoding", "MultiHeadAttention", "FeedForward", 
            "TransformerBlock", "ConvTransformerBlock", "TimeSeriesTransformer",
            "SelfAttention", "CrossAttention", "SpatialAttention", "ChannelAttention",
            "TemporalAttention", "CBAM", "ECA", "CoordinateAttention"
        ]
        
        layer_types.extend([f"PyTorch: {layer}" for layer in pytorch_layers])
        layer_types.extend([f"HAR: {layer}" for layer in har_layers])
        
        self.layer_type_combo['values'] = layer_types
        if layer_types:
            self.layer_type_combo.set(layer_types[0])
            # Don't call on_layer_type_change during initialization
            # It will be called when the user actually selects something
    
    def on_layer_type_change(self, event=None):
        """Handle layer type selection change"""
        # Check if params_frame exists (might not during initialization)
        if not hasattr(self, 'params_frame'):
            return
            
        # Clear existing parameter widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        layer_type = self.layer_type_var.get()
        if not layer_type:
            return
            
        # Extract actual layer name
        if ":" in layer_type:
            actual_layer_name = layer_type.split(":", 1)[1].strip()
        else:
            actual_layer_name = layer_type
            
        # Get layer parameters
        params = self.get_layer_parameters(actual_layer_name)
        
        # Create parameter widgets
        self.param_vars = {}
        for i, (param_name, param_info) in enumerate(params.items()):
            ttk.Label(self.params_frame, text=f"{param_name}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            
            if param_info['type'] == 'int':
                var = tk.StringVar(value=str(param_info['default']))
                entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky=tk.W, pady=2)
                self.param_vars[param_name] = var
                
            elif param_info['type'] == 'float':
                var = tk.StringVar(value=str(param_info['default']))
                entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky=tk.W, pady=2)
                self.param_vars[param_name] = var
                
            elif param_info['type'] == 'bool':
                var = tk.BooleanVar(value=param_info['default'])
                checkbox = ttk.Checkbutton(self.params_frame, variable=var)
                checkbox.grid(row=i, column=1, sticky=tk.W, pady=2)
                self.param_vars[param_name] = var
                
            elif param_info['type'] == 'choice':
                var = tk.StringVar(value=param_info['default'])
                combo = ttk.Combobox(self.params_frame, textvariable=var, width=12)
                combo['values'] = param_info['choices']
                combo.grid(row=i, column=1, sticky=tk.W, pady=2)
                self.param_vars[param_name] = var
                
            elif param_info['type'] == 'list':
                var = tk.StringVar(value=str(param_info['default']))
                entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
                entry.grid(row=i, column=1, sticky=tk.W, pady=2)
                ttk.Label(self.params_frame, text="(e.g., [1, 2, 3])", 
                         font=("Arial", 8), foreground="gray").grid(row=i, column=2, sticky=tk.W, padx=(5, 0), pady=2)
                self.param_vars[param_name] = var
    
    def get_layer_parameters(self, layer_name):
        """Get parameters for a specific layer type"""
        # Common parameters for different layer types
        params = {
            'Linear': {
                'in_features': {'type': 'int', 'default': 128, 'description': 'Input features'},
                'out_features': {'type': 'int', 'default': 64, 'description': 'Output features'},
                'bias': {'type': 'bool', 'default': True, 'description': 'Use bias'}
            },
            'Conv1d': {
                'in_channels': {'type': 'int', 'default': 9, 'description': 'Input channels'},
                'out_channels': {'type': 'int', 'default': 32, 'description': 'Output channels'},
                'kernel_size': {'type': 'int', 'default': 3, 'description': 'Kernel size'},
                'stride': {'type': 'int', 'default': 1, 'description': 'Stride'},
                'padding': {'type': 'int', 'default': 1, 'description': 'Padding'},
                'bias': {'type': 'bool', 'default': True, 'description': 'Use bias'}
            },
            'LSTM': {
                'input_size': {'type': 'int', 'default': 128, 'description': 'Input size'},
                'hidden_size': {'type': 'int', 'default': 64, 'description': 'Hidden size'},
                'num_layers': {'type': 'int', 'default': 1, 'description': 'Number of layers'},
                'bias': {'type': 'bool', 'default': True, 'description': 'Use bias'},
                'batch_first': {'type': 'bool', 'default': True, 'description': 'Batch first'},
                'dropout': {'type': 'float', 'default': 0.0, 'description': 'Dropout rate'}
            },
            'ConvBlock1D': {
                'in_channels': {'type': 'int', 'default': 9, 'description': 'Input channels'},
                'out_channels': {'type': 'int', 'default': 32, 'description': 'Output channels'},
                'kernel_size': {'type': 'int', 'default': 3, 'description': 'Kernel size'},
                'stride': {'type': 'int', 'default': 1, 'description': 'Stride'},
                'padding': {'type': 'int', 'default': 1, 'description': 'Padding'},
                'activation': {'type': 'choice', 'default': 'ReLU', 'choices': ['ReLU', 'Sigmoid', 'Tanh', 'None'], 'description': 'Activation'},
                'batch_norm': {'type': 'bool', 'default': True, 'description': 'Use batch norm'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout rate'}
            },
            'BiDirectionalLSTM': {
                'input_size': {'type': 'int', 'default': 128, 'description': 'Input size'},
                'hidden_size': {'type': 'int', 'default': 32, 'description': 'Hidden size'},
                'num_layers': {'type': 'int', 'default': 1, 'description': 'Number of layers'},
                'bias': {'type': 'bool', 'default': True, 'description': 'Use bias'},
                'batch_first': {'type': 'bool', 'default': True, 'description': 'Batch first'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout rate'},
                'lstm_type': {'type': 'choice', 'default': 'standard', 'choices': ['standard', 'layer_norm', 'peephole'], 'description': 'LSTM type'}
            },
            'Dropout': {
                'p': {'type': 'float', 'default': 0.5, 'description': 'Dropout probability'}
            },
            'ReLU': {},
            'Sigmoid': {},
            'Tanh': {},
            'Flatten': {
                'start_dim': {'type': 'int', 'default': 1, 'description': 'Start dimension'},
                'end_dim': {'type': 'int', 'default': -1, 'description': 'End dimension'}
            }
        }
        
        return params.get(layer_name, {})
    
    def add_layer(self):
        """
        CRITICAL: Add a new layer to the current model configuration
        
        This method handles the complete process of adding a layer:
        1. Validates that a layer type is selected
        2. Extracts the actual layer name from the selection
        3. Collects and validates all layer parameters
        4. Creates a layer configuration dictionary
        5. Adds the layer to the model configuration
        6. Updates the GUI display and statistics
        
        The layer configuration follows this structure:
        {
            'id': unique_integer_id,
            'layer_type': 'LayerName',
            'config': {param_name: param_value, ...}
        }
        """
        # CRITICAL: Validate layer type selection
        layer_type = self.layer_type_var.get()
        if not layer_type:
            messagebox.showerror("Error", "Please select a layer type")
            return
            
        # CRITICAL: Extract actual layer name from formatted string
        # Format: "Category: LayerName" -> "LayerName"
        if ":" in layer_type:
            actual_layer_name = layer_type.split(":", 1)[1].strip()
        else:
            actual_layer_name = layer_type
            
        # CRITICAL: Validate that parameters have been configured
        if not hasattr(self, 'param_vars'):
            messagebox.showerror("Error", "Please select a layer type first to configure parameters")
            return
            
        # CRITICAL: Collect and validate layer parameters
        layer_config = {}
        for param_name, var in self.param_vars.items():
            try:
                if isinstance(var, tk.BooleanVar):
                    # Boolean parameters
                    layer_config[param_name] = var.get()
                else:
                    value = var.get()
                    # CRITICAL: Type conversion for numeric parameters
                    if value.replace('.', '').replace('-', '').isdigit():
                        if '.' in value:
                            layer_config[param_name] = float(value)
                        else:
                            layer_config[param_name] = int(value)
                    elif value.startswith('[') and value.endswith(']'):
                        # CRITICAL: Parse list parameters (e.g., kernel_size=[3,3])
                        layer_config[param_name] = eval(value)
                    else:
                        # String parameters
                        layer_config[param_name] = value
            except:
                # Fallback: use raw string value
                layer_config[param_name] = var.get()
        
        # CRITICAL: Create layer configuration dictionary
        layer_info = {
            'id': self.layer_counter,           # Unique identifier
            'layer_type': actual_layer_name,    # Layer class name
            'config': layer_config              # Layer parameters
        }
        
        # CRITICAL: Add layer to model configuration
        self.model_config['layers'].append(layer_info)
        self.layer_counter += 1
        
        # Update layer list
        self.update_layer_list()
        self.update_code_output()
        self.update_model_stats()
        
        messagebox.showinfo("Success", f"Added {actual_layer_name} layer")
    
    def update_layer_list(self):
        """Update the layer treeview"""
        # Clear existing items
        for item in self.layer_tree.get_children():
            self.layer_tree.delete(item)
        
        # Add layers
        for i, layer in enumerate(self.model_config['layers']):
            # Format parameters for display
            key_params = []
            for key, value in layer['config'].items():
                if key in ['out_features', 'out_channels', 'hidden_size', 'kernel_size', 'stride', 'padding']:
                    key_params.append(f"{key}={value}")
            
            params_str = ', '.join(key_params) if key_params else 'Default'
            
            # Calculate output shape (simplified)
            output_shape = self.calculate_output_shape(i)
            
            self.layer_tree.insert('', 'end', values=(
                i + 1,
                layer['layer_type'],
                params_str,
                output_shape
            ))
    
    def calculate_output_shape(self, layer_index):
        """Calculate output shape for a layer (simplified)"""
        if layer_index >= len(self.model_config['layers']):
            return "N/A"
            
        layer = self.model_config['layers'][layer_index]
        layer_type = layer['layer_type']
        config = layer['config']
        
        # This is a simplified calculation
        if layer_type == 'Linear':
            return f"[batch, {config.get('out_features', '?')}]"
        elif layer_type in ['Conv1d', 'ConvBlock1D']:
            return f"[batch, {config.get('out_channels', '?')}, seq_len]"
        elif layer_type in ['LSTM', 'BiDirectionalLSTM']:
            hidden_size = config.get('hidden_size', '?')
            if layer_type == 'BiDirectionalLSTM':
                return f"[batch, seq_len, {hidden_size}*2]"
            else:
                return f"[batch, seq_len, {hidden_size}]"
        elif layer_type == 'Flatten':
            return "[batch, features]"
        else:
            return "[batch, ...]"
    
    def update_model_stats(self):
        """Update model statistics display"""
        try:
            # Build model to get statistics
            model = self.builder.build_model_from_config(self.model_config)
            
            # Calculate statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            # Update stats text
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            
            stats_text = f"""Model: {self.model_config['name']}
Input Shape: {self.model_config['input_shape']}
Output Classes: {self.model_config['num_classes']}
Total Layers: {len(self.model_config['layers'])}

Parameters:
  Total: {total_params:,}
  Trainable: {trainable_params:,}
  Non-trainable: {total_params - trainable_params:,}

Model Size: {model_size_mb:.2f} MB

Layer Breakdown:"""
            
            for i, layer in enumerate(self.model_config['layers']):
                # Try to get layer parameters - the model structure may vary
                try:
                    if hasattr(model, 'layers') and i < len(model.layers):
                        layer_params = sum(p.numel() for p in model.layers[i].parameters())
                    else:
                        # Fallback: estimate based on layer type
                        layer_type = layer['layer_type']
                        config = layer['config']
                        if layer_type == 'Linear':
                            in_features = config.get('in_features', 128)
                            out_features = config.get('out_features', 64)
                            layer_params = in_features * out_features + (out_features if config.get('bias', True) else 0)
                        elif layer_type in ['Conv1d', 'ConvBlock1D']:
                            in_channels = config.get('in_channels', 9)
                            out_channels = config.get('out_channels', 32)
                            kernel_size = config.get('kernel_size', 3)
                            layer_params = in_channels * out_channels * kernel_size + (out_channels if config.get('bias', True) else 0)
                        else:
                            layer_params = 0
                except:
                    layer_params = 0
                    
                stats_text += f"\n  {i+1}. {layer['layer_type']}: {layer_params:,} params"
            
            self.stats_text.insert(1.0, stats_text)
            self.stats_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"Error calculating statistics:\n{str(e)}")
            self.stats_text.config(state=tk.DISABLED)
    
    def edit_layer(self):
        """Edit the selected layer"""
        selection = self.layer_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a layer to edit")
            return
            
        # Get selected layer index
        item = self.layer_tree.item(selection[0])
        index = int(item['values'][0]) - 1
        layer = self.model_config['layers'][index]
        
        # Open parameter editing dialog
        self.open_layer_edit_dialog(index, layer)
    
    def open_layer_edit_dialog(self, layer_index, layer):
        """Open a dialog to edit layer parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {layer['layer_type']} Layer")
        dialog.geometry("500x400")
        dialog.resizable(True, True)
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Layer type (read-only)
        ttk.Label(main_frame, text=f"Layer Type: {layer['layer_type']}", 
                 style='Subtitle.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Get layer parameters
        params = self.get_layer_parameters(layer['layer_type'])
        param_vars = {}
        
        # Create parameter widgets
        for i, (param_name, param_info) in enumerate(params.items()):
            param_frame = ttk.Frame(params_frame)
            param_frame.pack(fill=tk.X, pady=2)
            
            # Parameter name and description
            ttk.Label(param_frame, text=f"{param_name}:", 
                     style='Subtitle.TLabel').pack(side=tk.LEFT, padx=(0, 10))
            
            if 'description' in param_info:
                ttk.Label(param_frame, text=f"({param_info['description']})", 
                         style='Info.TLabel').pack(side=tk.LEFT)
            
            # Input widget
            input_frame = ttk.Frame(param_frame)
            input_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
            
            # Get current value
            current_value = layer['config'].get(param_name, param_info['default'])
            
            if param_info['type'] == 'int':
                var = tk.StringVar(value=str(current_value))
                entry = ttk.Entry(input_frame, textvariable=var, width=15)
                entry.pack(side=tk.LEFT)
                param_vars[param_name] = var
                
            elif param_info['type'] == 'float':
                var = tk.StringVar(value=str(current_value))
                entry = ttk.Entry(input_frame, textvariable=var, width=15)
                entry.pack(side=tk.LEFT)
                param_vars[param_name] = var
                
            elif param_info['type'] == 'bool':
                var = tk.BooleanVar(value=bool(current_value))
                checkbox = ttk.Checkbutton(input_frame, variable=var)
                checkbox.pack(side=tk.LEFT)
                param_vars[param_name] = var
                
            elif param_info['type'] == 'choice':
                var = tk.StringVar(value=str(current_value))
                combo = ttk.Combobox(input_frame, textvariable=var, width=12)
                combo['values'] = param_info['choices']
                combo.pack(side=tk.LEFT)
                param_vars[param_name] = var
                
            elif param_info['type'] == 'list':
                var = tk.StringVar(value=str(current_value))
                entry = ttk.Entry(input_frame, textvariable=var, width=20)
                entry.pack(side=tk.LEFT)
                ttk.Label(input_frame, text="(e.g., [1, 2, 3])", 
                         style='Info.TLabel').pack(side=tk.LEFT, padx=(5, 0))
                param_vars[param_name] = var
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_changes():
            try:
                # Validate and get parameters
                new_config = {}
                for param_name, var in param_vars.items():
                    if isinstance(var, tk.BooleanVar):
                        new_config[param_name] = var.get()
                    else:
                        value = var.get().strip()
                        if not value:
                            continue
                            
                        # Try to convert to appropriate type
                        if value.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                            if '.' in value or 'e' in value.lower():
                                new_config[param_name] = float(value)
                            else:
                                new_config[param_name] = int(value)
                        elif value.startswith('[') and value.endswith(']'):
                            new_config[param_name] = eval(value)
                        elif value.lower() in ['true', 'false']:
                            new_config[param_name] = value.lower() == 'true'
                        else:
                            new_config[param_name] = value
                
                # Update the layer
                self.model_config['layers'][layer_index]['config'] = new_config
                
                # Update displays
                self.update_layer_list()
                self.update_code_output()
                self.update_model_stats()
                
                dialog.destroy()
                messagebox.showinfo("Success", f"Updated {layer['layer_type']} layer parameters")
                
            except Exception as e:
                messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
        
        def cancel_changes():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save Changes", command=save_changes, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=cancel_changes).pack(side=tk.LEFT)
    
    def remove_layer(self):
        """Remove the selected layer"""
        selection = self.layer_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a layer to remove")
            return
            
        # Get selected layer info
        item = self.layer_tree.item(selection[0])
        index = int(item['values'][0]) - 1
        layer_name = item['values'][1]
        
        if messagebox.askyesno("Confirm", f"Remove {layer_name} layer?"):
            self.model_config['layers'].pop(index)
            self.update_layer_list()
            self.update_code_output()
            self.update_model_stats()
    
    def move_layer_up(self):
        """Move selected layer up"""
        selection = self.layer_tree.selection()
        if not selection:
            return
            
        item = self.layer_tree.item(selection[0])
        index = int(item['values'][0]) - 1
        
        if index == 0:
            return
            
        # Swap layers
        layer = self.model_config['layers'].pop(index)
        self.model_config['layers'].insert(index - 1, layer)
        
        self.update_layer_list()
        self.update_code_output()
        self.update_model_stats()
        
        # Reselect the moved layer
        self.layer_tree.selection_set(self.layer_tree.get_children()[index - 1])
    
    def move_layer_down(self):
        """Move selected layer down"""
        selection = self.layer_tree.selection()
        if not selection:
            return
            
        item = self.layer_tree.item(selection[0])
        index = int(item['values'][0]) - 1
        
        if index == len(self.model_config['layers']) - 1:
            return
            
        # Swap layers
        layer = self.model_config['layers'].pop(index)
        self.model_config['layers'].insert(index + 1, layer)
        
        self.update_layer_list()
        self.update_code_output()
        self.update_model_stats()
        
        # Reselect the moved layer
        self.layer_tree.selection_set(self.layer_tree.get_children()[index + 1])
    
    def update_model_info(self):
        """Update model information"""
        try:
            # Parse input shape
            input_shape_str = self.input_shape_var.get().strip()
            if input_shape_str.startswith('[') and input_shape_str.endswith(']'):
                self.model_config['input_shape'] = eval(input_shape_str)
            else:
                messagebox.showerror("Error", "Input shape must be a list like [1, 9, 128]")
                return
                
            self.model_config['name'] = self.model_name_var.get()
            self.model_config['num_classes'] = int(self.num_classes_var.get())
            
            self.update_code_output()
            self.update_model_stats()
            messagebox.showinfo("Success", "Model information updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def update_code_output(self):
        """Update the generated code display"""
        try:
            # Generate PyTorch code
            code = self.generate_pytorch_code()
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, code)
        except Exception as e:
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, f"Error generating code: {str(e)}")
    
    def generate_pytorch_code(self):
        """Generate PyTorch model code"""
        code = f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class {self.model_config['name']}(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input shape: {self.model_config['input_shape']}
        # Number of classes: {self.model_config['num_classes']}
        
'''
        
        # Add layers
        for i, layer in enumerate(self.model_config['layers']):
            layer_type = layer['layer_type']
            config = layer['config']
            
            # Generate layer initialization
            if config:
                config_str = ', '.join([f'{k}={repr(v)}' for k, v in config.items()])
                code += f"        self.layer_{i} = nn.{layer_type}({config_str})\n"
            else:
                code += f"        self.layer_{i} = nn.{layer_type}()\n"
        
        # Generate forward method
        code += '''
    def forward(self, x):
        # Input shape: ''' + str(self.model_config['input_shape']) + '''
'''
        
        for i, layer in enumerate(self.model_config['layers']):
            layer_type = layer['layer_type']
            
            # Handle special cases
            if layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'Softmax']:
                code += f"        x = F.{layer_type.lower()}(x)\n"
            elif any(rnn_type in layer_type for rnn_type in ['LSTM', 'GRU', 'RNN', 'BiDirectionalLSTM']):
                code += f"        x, _ = self.layer_{i}(x)  # Unpack RNN output\n"
            else:
                code += f"        x = self.layer_{i}(x)\n"
        
        code += f"        return x  # Output shape: [batch_size, {self.model_config['num_classes']}]\n"
        
        # Add model creation example
        code += f'''
# Create model instance
model = {self.model_config['name']}()

# Test with dummy input
dummy_input = torch.randn(2, {self.model_config['input_shape'][1]}, {self.model_config['input_shape'][2]})
output = model(dummy_input)
print(f"Input shape: {{dummy_input.shape}}")
print(f"Output shape: {{output.shape}}")
'''
        
        return code
    
    def test_model(self):
        """Test the generated model"""
        try:
            # Build model using the builder
            model = self.builder.build_model_from_config(self.model_config)
            
            # Test with dummy input
            input_shape = self.model_config['input_shape']
            batch_size = 2
            dummy_input = torch.randn(batch_size, input_shape[1], input_shape[2])
            
            with torch.no_grad():
                output = model(dummy_input)
            
            messagebox.showinfo("Test Results", 
                              f"Model test successful!\n\n"
                              f"Input shape: {dummy_input.shape}\n"
                              f"Output shape: {output.shape}\n"
                              f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            messagebox.showerror("Test Failed", f"Model test failed:\n{str(e)}")
    
    def save_config(self):
        """Save model configuration to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Model Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.model_config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def load_config(self):
        """Load model configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Model Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.model_config = json.load(f)
                
                # Update GUI
                self.model_name_var.set(self.model_config['name'])
                self.input_shape_var.set(str(self.model_config['input_shape']))
                self.num_classes_var.set(str(self.model_config['num_classes']))
                
                # Update layer counter
                if self.model_config['layers']:
                    self.layer_counter = max(layer['id'] for layer in self.model_config['layers']) + 1
                else:
                    self.layer_counter = 0
                
                self.update_layer_list()
                self.update_code_output()
                self.update_model_stats()
                
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {str(e)}")
    
    def clear_all(self):
        """Clear all layers"""
        if messagebox.askyesno("Confirm", "Clear all layers?"):
            self.model_config['layers'] = []
            self.layer_counter = 0
            self.update_layer_list()
            self.update_code_output()
            self.update_model_stats()
    
    def copy_code(self):
        """Copy generated code to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.code_text.get(1.0, tk.END))
        messagebox.showinfo("Success", "Code copied to clipboard!")


def main():
    root = tk.Tk()
    app = SimpleModelBuilderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
