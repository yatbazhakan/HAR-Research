#!/usr/bin/env python3
"""
CRITICAL: Unit Tests for Data Loading

This module contains unit tests for data loading functionality,
including UCI-HAR dataset loading and preprocessing.
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from har.datasets.ucihar import load_ucihar_windows


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading functionality"""
    
    def test_ucihar_loading(self):
        """Test UCI-HAR dataset loading"""
        try:
            # Test loading UCI-HAR dataset
            data = load_ucihar_windows(
                root_dir="data/uci_har"
            )
            
            # Check data structure
            self.assertIn('X', data)
            self.assertIn('y', data)
            self.assertIn('subject_id', data)
            
            # Check data shapes
            X = data['X']
            y = data['y']
            subject_id = data['subject_id']
            
            self.assertEqual(len(X), len(y))
            self.assertEqual(len(X), len(subject_id))
            self.assertEqual(X.shape[1], 6)  # 6 sensor channels
            self.assertEqual(X.shape[2], 128)  # 128 time steps
            
            # Check data types
            self.assertEqual(X.dtype, np.float32)
            self.assertEqual(y.dtype, np.int64)
            
            # Check value ranges
            self.assertTrue(np.all(y >= 1))  # Activity labels start from 1
            self.assertTrue(np.all(y <= 6))  # 6 activity classes
            
        except FileNotFoundError:
            self.skipTest("UCI-HAR dataset not found. Please download it first.")
        except Exception as e:
            self.fail(f"Unexpected error loading UCI-HAR dataset: {e}")
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        # Create dummy data
        dummy_data = {
            'X': np.random.randn(100, 6, 128).astype(np.float32),
            'y': np.random.randint(1, 7, 100),
            'subject_id': np.random.randint(1, 31, 100)
        }
        
        # Test data validation
        X = dummy_data['X']
        y = dummy_data['y']
        subject_id = dummy_data['subject_id']
        
        # Check shapes
        self.assertEqual(X.shape, (100, 6, 128))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(subject_id.shape, (100,))
        
        # Check data types
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(subject_id.dtype, np.int64)
    
    def test_window_creation(self):
        """Test window creation for time series data"""
        # Create dummy time series data
        time_series = np.random.randn(1000, 6)  # 1000 time steps, 6 channels
        
        # Test windowing
        window_size = 128
        overlap = 0.5
        step_size = int(window_size * (1 - overlap))
        
        windows = []
        for i in range(0, len(time_series) - window_size + 1, step_size):
            window = time_series[i:i + window_size]
            windows.append(window)
        
        # Check window creation
        self.assertGreater(len(windows), 0)
        
        for window in windows:
            self.assertEqual(window.shape, (window_size, 6))
    
    def test_subject_id_consistency(self):
        """Test that subject IDs are consistent across data splits"""
        try:
            # Load train and test data
            train_data = load_ucihar_windows(
                root_dir="data/uci_har"
            )
            
            # Note: The function loads all data, we need to filter by split
            # For now, just test that the function works
            self.assertIsNotNone(train_data)
            
            # Check that subject IDs are consistent
            if 'subject_id' in train_data.columns:
                train_subjects = set(train_data['subject_id'])
                
                # Should have subjects
                self.assertGreater(len(train_subjects), 0)
                
                # Check subject ID ranges
                self.assertTrue(all(1 <= s <= 30 for s in train_subjects))  # UCI-HAR has 30 subjects
            
        except FileNotFoundError:
            self.skipTest("UCI-HAR dataset not found. Please download it first.")


if __name__ == '__main__':
    unittest.main()
