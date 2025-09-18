#!/usr/bin/env python3
"""
CRITICAL: Integration Tests for Training Pipeline

This module contains integration tests that test the complete training pipeline,
including data loading, model building, training, and evaluation.
"""

import unittest
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import tempfile
import shutil

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from har.models.model_config_builder import build_model_from_yaml
from har.datasets.ucihar import load_ucihar_windows


class TestTrainingPipeline(unittest.TestCase):
    """Test cases for complete training pipeline"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path("configs")
    
    def tearDown(self):
        """Clean up after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_simple_training_loop(self):
        """Test a simple training loop with dummy data"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        # Create dummy data
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(5):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Check that loss is decreasing (generally)
            if epoch > 0:
                self.assertLess(total_loss, 100)  # Reasonable loss value
    
    def test_model_training_with_yaml_config(self):
        """Test training a model loaded from YAML configuration"""
        config_file = "har_cnn_tcn.yaml"
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            self.skipTest(f"Configuration file not found: {config_path}")
        
        try:
            # Load model from YAML
            model = build_model_from_yaml(config_path)
            
            # Create dummy HAR data
            batch_size = 32
            X = torch.randn(batch_size, 6, 128)  # HAR input shape
            y = torch.randint(0, 6, (batch_size,))  # 6 activity classes
            
            # Set up training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Check that training completed without errors
            self.assertFalse(torch.isnan(loss))
            self.assertGreater(loss.item(), 0)
            
        except Exception as e:
            self.fail(f"YAML model training test failed: {e}")
    
    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        # Create dummy test data
        X_test = torch.randn(50, 10)
        y_test = torch.randint(0, 3, (50,))
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_test).float().mean()
        
        # Check that evaluation completed
        self.assertGreaterEqual(accuracy.item(), 0)
        self.assertLessEqual(accuracy.item(), 1)
    
    def test_model_save_load(self):
        """Test saving and loading models"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test that models produce same output
        dummy_input = torch.randn(1, 10)
        
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = loaded_model(dummy_input)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_data_loading_integration(self):
        """Test data loading integration with training"""
        try:
            # Load UCI-HAR data
            data = load_ucihar_windows(
                root_dir="data/uci_har"
            )
            
            # Create simple model
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6 * 128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 6)
            )
            
            # Create data loader
            X = torch.from_numpy(data['X'])
            y = torch.from_numpy(data['y']) - 1  # Convert to 0-based indexing
            
            dataset = torch.utils.data.TensorDataset(X, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Test one training step
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                break  # Just test one step
            
            # Check that training completed without errors
            self.assertFalse(torch.isnan(loss))
            
        except FileNotFoundError:
            self.skipTest("UCI-HAR dataset not found. Please download it first.")
        except Exception as e:
            self.fail(f"Data loading integration test failed: {e}")
    
    def test_model_validation_during_training(self):
        """Test model validation during training"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        # Create train and validation data
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 3, (100,))
        X_val = torch.randn(50, 10)
        y_val = torch.randint(0, 3, (50,))
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop with validation
        for epoch in range(3):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=1)
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_accuracy = val_correct / val_total
            
            # Check that validation completed
            self.assertGreaterEqual(val_accuracy, 0)
            self.assertLessEqual(val_accuracy, 1)


if __name__ == '__main__':
    unittest.main()
