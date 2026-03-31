"""
Data Splitting Script - Phase 3.3
Creates balanced train/validation/test splits with speaker independence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Creates balanced train/validation/test splits"""
    
    def __init__(self, manifest_path: str = "data/manifests/master.csv", 
                 output_dir: str = "data/manifests"):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Master manifest not found: {self.manifest_path}")
        
        self.df = pd.read_csv(self.manifest_path)
        self.min_per_accent = 50  # Minimum samples per accent in test set
    
    def validate_data(self) -> bool:
        """Validate manifest before splitting"""
        print("\n" + "="*60)
        print("VALIDATING DATA")
        print("="*60)
        
        print(f"Total samples: {len(self.df)}")
        print(f"Unique speakers: {self.df['speaker_id'].nunique()}")
        print(f"Unique accents: {self.df['accent_type'].nunique()}")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print(f"\n⚠ Missing values detected:\n{missing[missing > 0]}")
        
        # Check accent distribution
        print("\nAccent distribution:")
        accent_counts = self.df['accent_type'].value_counts()
        for accent, count in accent_counts.items():
            print(f"  {accent}: {count} samples")
        
        # Warn if any accent has < min_per_accent samples
        if (accent_counts < self.min_per_accent).any():
            print(f"\n⚠ Warning: Some accents have < {self.min_per_accent} samples")
            print("  Consider reducing min_per_accent or adding more data")
        
        return True
    
    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits with speaker independence"""
        print("\n" + "="*60)
        print("CREATING SPLITS")
        print("="*60)
        
        print("Splitting data (80% train, 10% val, 10% test)...")
        print("Ensuring speaker independence...")
        
        # First split: train (80%) vs temp (20%)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, temp_idx = next(splitter.split(self.df, groups=self.df['speaker_id']))
        
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        temp_df = self.df.iloc[temp_idx].reset_index(drop=True)
        
        # Second split: validation (50%) vs test (50%)
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_df['speaker_id']))
        
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> bool:
        """Validate that splits meet requirements"""
        print("\n" + "="*60)
        print("VALIDATING SPLITS")
        print("="*60)
        
        # Check speaker independence
        train_speakers = set(train_df['speaker_id'])
        val_speakers = set(val_df['speaker_id'])
        test_speakers = set(test_df['speaker_id'])
        
        train_val_overlap = train_speakers & val_speakers
        train_test_overlap = train_speakers & test_speakers
        val_test_overlap = val_speakers & test_speakers
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("✗ Speaker overlap detected!")
            return False
        
        print("✓ No speaker overlap between splits")
        
        # Check accent distribution in test set
        print("\nAccent distribution in test set:")
        test_accents = test_df['accent_type'].value_counts()
        for accent, count in test_accents.items():
            status = "✓" if count >= self.min_per_accent else "✗"
            print(f"  {status} {accent}: {count} samples")
        
        if (test_accents < self.min_per_accent).any():
            print(f"\n⚠ Some accents have < {self.min_per_accent} samples in test set")
        
        # Print split statistics
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")
        
        print(f"\nSpeaker distribution:")
        print(f"  Train speakers: {train_df['speaker_id'].nunique()}")
        print(f"  Val speakers: {val_df['speaker_id'].nunique()}")
        print(f"  Test speakers: {test_df['speaker_id'].nunique()}")
        
        return True
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> None:
        """Save split manifests to CSV"""
        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "validation.csv"
        test_path = self.output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n✓ Splits saved:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print(f"  {test_path}")
    
    def run(self) -> None:
        """Run data splitting pipeline"""
        print("\n" + "DATA SPLITTING - PHASE 3.3".center(60))
        print("="*60)
        
        # Validate
        self.validate_data()
        
        # Create splits
        train_df, val_df, test_df = self.create_splits()
        
        # Validate splits
        if self.validate_splits(train_df, val_df, test_df):
            self.save_splits(train_df, val_df, test_df)
            print("\n✓ Data splitting complete!")
        else:
            print("\n✗ Split validation failed. Please review the data.")


if __name__ == "__main__":
    from typing import Tuple
    
    splitter = DataSplitter()
    splitter.run()
