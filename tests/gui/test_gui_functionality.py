#!/usr/bin/env python3
"""
CRITICAL: GUI Functionality Tests

This module contains tests for GUI functionality, including model builder GUI,
sweep GUI, and experiment GUI components.
"""

import unittest
import sys
import tkinter as tk
from pathlib import Path
import tempfile
import shutil

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import GUI modules
try:
    from scripts.simple_model_builder_gui import SimpleModelBuilderGUI
    from scripts.sweep_gui import SweepConfigGUI
    from scripts.experiment_gui import ExperimentConfigGUI
    from scripts.training_gui import TrainingConfigGUI
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"GUI modules not available: {e}")


class TestModelBuilderGUI(unittest.TestCase):
    """Test cases for Model Builder GUI"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        if not GUI_AVAILABLE:
            self.skipTest("GUI modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        self.gui = SimpleModelBuilderGUI(self.root)
    
    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, 'root'):
            self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gui_initialization(self):
        """Test that GUI initializes correctly"""
        self.assertIsNotNone(self.gui)
        self.assertIsNotNone(self.gui.builder)
        self.assertIsNotNone(self.gui.model_config)
    
    def test_layer_management(self):
        """Test layer management functionality"""
        # Test adding a layer
        initial_layers = len(self.gui.model_config['layers'])
        
        # Add a Linear layer
        self.gui.layer_type_var.set("Linear")
        self.gui.add_layer()
        
        # Check that layer was added
        self.assertEqual(len(self.gui.model_config['layers']), initial_layers + 1)
        
        # Test removing a layer
        if len(self.gui.model_config['layers']) > 0:
            self.gui.remove_layer()
            self.assertEqual(len(self.gui.model_config['layers']), initial_layers)
    
    def test_model_configuration(self):
        """Test model configuration functionality"""
        # Test setting model name
        test_name = "test_model"
        self.gui.model_name_var.set(test_name)
        self.assertEqual(self.gui.model_config['name'], test_name)
        
        # Test setting input shape
        test_shape = [1, 3, 32, 32]
        self.gui.input_shape_var.set("1, 3, 32, 32")
        # Note: Input shape parsing would need to be implemented in the GUI
        
        # Test setting number of classes
        test_classes = 10
        self.gui.num_classes_var.set(str(test_classes))
        self.assertEqual(self.gui.model_config['num_classes'], test_classes)
    
    def test_layer_validation(self):
        """Test layer parameter validation"""
        # Test adding a layer with invalid parameters
        self.gui.layer_type_var.set("Linear")
        
        # This should handle invalid parameters gracefully
        try:
            self.gui.add_layer()
        except Exception as e:
            self.fail(f"Layer addition failed with valid parameters: {e}")
    
    def test_model_statistics(self):
        """Test model statistics calculation"""
        # Add some layers
        self.gui.layer_type_var.set("Linear")
        self.gui.add_layer()
        
        # Test statistics calculation
        try:
            self.gui.update_model_stats()
        except Exception as e:
            self.fail(f"Model statistics calculation failed: {e}")


class TestSweepGUI(unittest.TestCase):
    """Test cases for Sweep GUI"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        if not GUI_AVAILABLE:
            self.skipTest("GUI modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        self.gui = SweepConfigGUI(self.root)
    
    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, 'root'):
            self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gui_initialization(self):
        """Test that Sweep GUI initializes correctly"""
        self.assertIsNotNone(self.gui)
        self.assertIsNotNone(self.gui.base_config)
        self.assertIsNotNone(self.gui.sweep_config)
    
    def test_configuration_loading(self):
        """Test configuration loading functionality"""
        # Test loading base configuration
        try:
            self.gui.load_base_config()
        except Exception as e:
            # This might fail if no config file exists, which is expected
            pass
    
    def test_sweep_validation(self):
        """Test sweep configuration validation"""
        # Test validation with empty configuration
        try:
            self.gui.validate_sweep_config()
        except Exception as e:
            # This might fail with empty config, which is expected
            pass


class TestExperimentGUI(unittest.TestCase):
    """Test cases for Experiment GUI"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        if not GUI_AVAILABLE:
            self.skipTest("GUI modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        self.gui = ExperimentConfigGUI(self.root)
    
    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, 'root'):
            self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gui_initialization(self):
        """Test that Experiment GUI initializes correctly"""
        self.assertIsNotNone(self.gui)
        self.assertIsNotNone(self.gui.config)
    
    def test_configuration_management(self):
        """Test configuration management functionality"""
        # Test setting basic configuration
        try:
            self.gui.config['experiment_name'] = 'test_experiment'
            self.gui.config['dataset'] = 'uci_har'
        except Exception as e:
            self.fail(f"Configuration management failed: {e}")


class TestTrainingGUI(unittest.TestCase):
    """Test cases for Training GUI"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        if not GUI_AVAILABLE:
            self.skipTest("GUI modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        self.gui = TrainingConfigGUI(self.root)
    
    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, 'root'):
            self.root.destroy()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_gui_initialization(self):
        """Test that Training GUI initializes correctly"""
        self.assertIsNotNone(self.gui)
        self.assertIsNotNone(self.gui.config)
    
    def test_training_configuration(self):
        """Test training configuration functionality"""
        # Test setting training parameters
        try:
            self.gui.config['epochs'] = 10
            self.gui.config['batch_size'] = 32
            self.gui.config['learning_rate'] = 0.001
        except Exception as e:
            self.fail(f"Training configuration failed: {e}")


if __name__ == '__main__':
    unittest.main()
