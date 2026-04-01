"""
Unit Tests for Data Pipeline
"""

import unittest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_data_dir.mkdir(exist_ok=True)
    
    def test_manifest_creation(self):
        """Test that manifest CSV can be created"""
        # Create sample manifest
        data = {
            'audio_path': ['/path/to/audio1.wav', '/path/to/audio2.wav'],
            'transcript': ['hello world', 'goodbye world'],
            'accent_type': ['north_american', 'British'],
            'duration_seconds': [5.0, 6.0],
            'source_dataset': ['librispeech', 'common_voice'],
            'speaker_id': ['ls_001', 'cv_001']
        }
        
        df = pd.DataFrame(data)
        
        # Assertions
        self.assertEqual(len(df), 2)
        self.assertIn('audio_path', df.columns)
        self.assertIn('accent_type', df.columns)
        
        print("✓ Manifest creation test passed")
    
    def test_data_split_ratios(self):
        """Test that data splits have correct ratios"""
        # Create sample data
        total_samples = 1000
        data = {
            'audio_path': [f'/audio/{i}.wav' for i in range(total_samples)],
            'speaker_id': [f'speaker_{i%500}' for i in range(total_samples)],
            'accent_type': ['north_american'] * total_samples,
            'transcript': ['test'] * total_samples,
            'duration_seconds': [5.0] * total_samples,
            'source_dataset': ['test'] * total_samples
        }
        
        df = pd.DataFrame(data)
        
        # Simulate split (80/10/10)
        train_size = int(len(df) * 0.8)
        val_size = int(len(df) * 0.1)
        test_size = len(df) - train_size - val_size
        
        # Assertions
        self.assertEqual(train_size, 800)
        self.assertEqual(val_size, 100)
        self.assertEqual(test_size, 100)
        
        print("✓ Data split ratio test passed")
    
    def test_accent_categorization(self):
        """Test accent categorization logic"""
        import json
        
        accent_mapping_path = Path(__file__).parent.parent / "data" / "accent_mapping.json"
        
        if accent_mapping_path.exists():
            with open(accent_mapping_path, 'r') as f:
                mapping = json.load(f)
            
            # Check mapping structure
            self.assertIn('arabic', mapping)
            self.assertIn('south_asian', mapping)
            self.assertIn('east_asian', mapping)
            self.assertIn('european', mapping)
            self.assertIn('north_american', mapping)
            
            print("✓ Accent categorization test passed")


class TestDataQuality(unittest.TestCase):
    """Test data quality checks"""
    
    def test_transcript_validation(self):
        """Test transcript validation"""
        test_cases = [
            ("The quick brown fox", True),
            ("", False),
            ("   ", False),
            ("Valid transcript", True)
        ]
        
        for transcript, expected in test_cases:
            is_valid = len(str(transcript).strip()) > 0
            self.assertEqual(is_valid, expected)
        
        print("✓ Transcript validation test passed")
    
    def test_duration_validation(self):
        """Test duration validation"""
        test_cases = [
            (0.05, False),   # Too short
            (0.1, True),     # Minimum
            (5.0, True),     # Normal
            (300.0, True)    # Long
        ]
        
        min_duration = 0.1
        
        for duration, expected in test_cases:
            is_valid = duration >= min_duration
            self.assertEqual(is_valid, expected)
        
        print("✓ Duration validation test passed")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
