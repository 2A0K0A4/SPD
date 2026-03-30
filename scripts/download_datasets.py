"""
Dataset Download Manager
Downloads all required datasets for accent recognition project
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class DatasetDownloadManager:
    """Manages downloading and extracting all datasets"""
    
    def __init__(self, data_raw_dir: str = "data/raw"):
        self.data_raw_dir = Path(data_raw_dir)
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        
    def download_common_voice(self) -> None:
        """Download Mozilla Common Voice dataset"""
        print("\n" + "="*60)
        print("COMMON VOICE DATASET")
        print("="*60)
        print("""
        Mozilla Common Voice is a large multilingual speech corpus.
        
        MANUAL DOWNLOAD REQUIRED:
        1. Go to: https://commonvoice.mozilla.org/en/datasets
        2. Create a free account if needed
        3. Download English dataset (validate clips)
        4. Extract to: data/raw/common_voice/
        
        File structure should be:
        data/raw/common_voice/
            ├── validated.tsv
            ├── clips/
            └── [audio files]
        
        Note: Download size ~70GB for full dataset
              Recommended: Start with 10GB subset
        """)
        
    def download_librispeech(self) -> None:
        """Download LibriSpeech dataset"""
        print("\n" + "="*60)
        print("LIBRISPEECH DATASET")
        print("="*60)
        print("""
        LibriSpeech is a corpus of 1000+ hours of read English speech.
        
        SEMI-AUTOMATIC DOWNLOAD:
        Two recommended subsets can be downloaded via commands.
        
        Command 1: train-clean-100 (100 hours clean training data)
        Command 2: test-clean (test subset)
        """)
        
        choice = input("Download LibriSpeech? (y/n): ").strip().lower()
        if choice != 'y':
            print("Skipping LibriSpeech.")
            return
            
        librispeech_dir = self.data_raw_dir / "librispeech"
        librispeech_dir.mkdir(exist_ok=True)
        
        try:
            print("Downloading train-clean-100 (100 hours)...")
            url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
            subprocess.run([
                "curl", "-L", url, "-o", 
                str(librispeech_dir / "train-clean-100.tar.gz")
            ], check=True)
            
            print("Extracting...")
            subprocess.run([
                "tar", "-xzf", str(librispeech_dir / "train-clean-100.tar.gz"),
                "-C", str(librispeech_dir)
            ], check=True)
            
            print("✓ LibriSpeech downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading LibriSpeech: {e}")
            print("Please download manually from: https://www.openslr.org/12")
    
    def download_speech_accent_archive(self) -> None:
        """Download Speech Accent Archive"""
        print("\n" + "="*60)
        print("SPEECH ACCENT ARCHIVE")
        print("="*60)
        print("""
        Speech Accent Archive contains 2000+ speakers from 170+ countries
        reading the same English passage.
        
        MANUAL DOWNLOAD REQUIRED:
        1. Go to: https://accent.gmu.edu/
        2. Download available recordings (8 MB zip files)
        3. Extract to: data/raw/speech_accent_archive/
        
        File structure should be:
        data/raw/speech_accent_archive/
            ├── [country folders]
            ├── [audio files]
            └── metadata.txt (if available)
        
        Note: ~2000 audio files total (~2GB)
        """)
    
    def download_voxpopuli(self) -> None:
        """Download VoxPopuli dataset"""
        print("\n" + "="*60)
        print("VOXPOPULI DATASET")
        print("="*60)
        print("""
        VoxPopuli contains MEP (Member of European Parliament) recordings
        with various European accents.
        
        MANUAL DOWNLOAD REQUIRED:
        1. Go to: https://github.com/facebookresearch/voxpopuli
        2. Follow instructions to download English portion
        3. Extract to: data/raw/voxpopuli/
        
        Alternatively, use provided download script:
        python scripts/download_voxpopuli.py
        
        File structure should be:
        data/raw/voxpopuli/
            ├── asr/
            ├── multilingual/
            └── speech_16khz/
        
        Note: Large download (~100GB) - English subset recommended
        """)
    
    def verify_downloads(self) -> Dict[str, bool]:
        """Verify that all datasets are downloaded"""
        print("\n" + "="*60)
        print("VERIFYING DOWNLOADS")
        print("="*60)
        
        datasets = {
            'common_voice': self.data_raw_dir / 'common_voice',
            'librispeech': self.data_raw_dir / 'librispeech',
            'speech_accent_archive': self.data_raw_dir / 'speech_accent_archive',
            'voxpopuli': self.data_raw_dir / 'voxpopuli'
        }
        
        verification = {}
        for name, path in datasets.items():
            exists = path.exists()
            has_files = len(list(path.glob('*'))) > 0 if exists else False
            status = "✓" if has_files else "✗"
            verification[name] = has_files
            print(f"{status} {name}: {path}")
        
        return verification
    
    def run(self) -> None:
        """Run dataset download manager"""
        print("\n" + "🎵 ACCENT RECOGNITION - DATASET DOWNLOAD MANAGER 🎵".center(60))
        print("="*60)
        
        print("""
        This script will help you download all required datasets.
        Note: Some datasets require manual download due to licensing.
        
        DATASETS REQUIRED:
        1. Mozilla Common Voice (English) - ~70GB
        2. LibriSpeech - ~100GB
        3. Speech Accent Archive - ~2GB
        4. VoxPopuli (English) - ~100GB
        
        Total space needed: ~200-300GB
        
        INSTRUCTION SUMMARY:
        - Phase 2.1: Common Voice (MANUAL)
        - Phase 2.2: LibriSpeech (SEMI-AUTOMATIC)
        - Phase 2.3: Speech Accent Archive (MANUAL)
        - Phase 2.4: VoxPopuli (MANUAL)
        """)
        
        input("Press Enter to continue...")
        
        # Show download instructions
        self.download_common_voice()
        input("Press Enter when done downloading Common Voice...")
        
        self.download_librispeech()
        
        self.download_speech_accent_archive()
        input("Press Enter when done downloading Speech Accent Archive...")
        
        self.download_voxpopuli()
        input("Press Enter when done downloading VoxPopuli...")
        
        # Verify
        verification = self.verify_downloads()
        
        completed = sum(1 for v in verification.values() if v)
        total = len(verification)
        
        print(f"\n✓ Downloaded: {completed}/{total} datasets")
        
        if completed > 0:
            print("\n✓ Ready for Phase 3: Dataset Organization!")
        else:
            print("\n⚠ Please download at least one dataset before proceeding.")


if __name__ == "__main__":
    manager = DatasetDownloadManager()
    manager.run()
