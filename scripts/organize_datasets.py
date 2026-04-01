"""
Dataset Organization Script - Phase 3.1 & 3.2
Creates master manifest and categorizes by accent type
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organizes raw datasets into unified manifest with accent categorization"""
    
    def __init__(self, data_raw_dir: str = "data/raw", data_manifests_dir: str = "data/manifests"):
        self.data_raw_dir = Path(data_raw_dir)
        self.manifests_dir = Path(data_manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        # Load accent mapping
        with open("data/accent_mapping.json", "r") as f:
            self.accent_mapping = json.load(f)
        
        self.master_manifest = []
    
    def process_common_voice(self) -> None:
        """Extract metadata from Common Voice dataset"""
        print("\n" + "="*60)
        print("PROCESSING COMMON VOICE")
        print("="*60)
        
        cv_dir = self.data_raw_dir / "common_voice"
        if not cv_dir.exists():
            print("✗ Common Voice directory not found")
            return
        
        # Common Voice has validated.tsv
        validated_file = cv_dir / "validated.tsv"
        if not validated_file.exists():
            print("✗ validated.tsv not found")
            return
        
        print("Reading validated.tsv...")
        try:
            df = pd.read_csv(validated_file, sep='\t')
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Common Voice"):
                audio_path = cv_dir / "clips" / row['path']
                
                if not audio_path.exists():
                    continue
                
                # Get accent from row
                accent = row.get('accent', '').lower() if 'accent' in row else "english"
                
                # Categorize accent
                accent_type = self._categorize_accent(accent)
                
                # Get duration
                try:
                    audio, sr = librosa.load(str(audio_path), sr=16000)
                    duration = len(audio) / sr
                except:
                    continue
                
                # Extract speaker_id (use participant_id)
                speaker_id = f"cv_{row.get('client_id', 'unknown')}"
                
                self.master_manifest.append({
                    'audio_path': str(audio_path),
                    'transcript': row.get('sentence', ''),
                    'accent_type': accent_type,
                    'duration_seconds': duration,
                    'source_dataset': 'common_voice',
                    'speaker_id': speaker_id,
                    'accent_label': accent
                })
            
            print(f"✓ Processed {len([m for m in self.master_manifest if m['source_dataset'] == 'common_voice'])} Common Voice samples")
        
        except Exception as e:
            logger.error(f"Error processing Common Voice: {e}")
    
    def process_librispeech(self) -> None:
        """Extract metadata from LibriSpeech dataset"""
        print("\n" + "="*60)
        print("PROCESSING LIBRISPEECH")
        print("="*60)
        
        ls_dir = self.data_raw_dir / "librispeech"
        if not ls_dir.exists():
            print("✗ LibriSpeech directory not found")
            return
        
        # Find all .flac files and corresponding .trans.txt
        speaker_count = 0
        
        for flac_file in tqdm(ls_dir.rglob("*.flac"), desc="Processing LibriSpeech"):
            # Find corresponding transcript
            trans_file = flac_file.parent / f"{flac_file.stem}.trans.txt"
            
            if not trans_file.exists():
                continue
            
            # Read transcript
            with open(trans_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                
                file_id, transcript = parts
                # Construct matching flac path
                matching_flac = [f for f in ls_dir.rglob(f"{file_id}.flac")]
                if not matching_flac:
                    continue
                
                audio_path = matching_flac[0]
                
                try:
                    audio, sr = librosa.load(str(audio_path), sr=16000)
                    duration = len(audio) / sr
                except:
                    continue
                
                # LibriSpeech is North American English
                accent_type = "north_american"
                speaker_id = f"ls_{file_id.split('-')[0]}"
                
                self.master_manifest.append({
                    'audio_path': str(audio_path),
                    'transcript': transcript.strip(),
                    'accent_type': accent_type,
                    'duration_seconds': duration,
                    'source_dataset': 'librispeech',
                    'speaker_id': speaker_id,
                    'accent_label': 'north_american'
                })
                
                speaker_count += 1
        
        print(f"✓ Processed {speaker_count} LibriSpeech samples")
    
    def process_speech_accent_archive(self) -> None:
        """Extract metadata from Speech Accent Archive"""
        print("\n" + "="*60)
        print("PROCESSING SPEECH ACCENT ARCHIVE")
        print("="*60)
        
        saa_dir = self.data_raw_dir / "speech_accent_archive"
        if not saa_dir.exists():
            print("✗ Speech Accent Archive directory not found")
            return
        
        # All speakers read the same passage
        passage = "please call stella ask her to bring these things with her from the store"
        
        processed = 0
        for audio_file in tqdm(list(saa_dir.glob("*.wav")) + list(saa_dir.glob("*.mp3")), 
                              desc="Processing Speech Accent Archive"):
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                duration = len(audio) / sr
            except:
                continue
            
            # Extract accent info from filename or metadata
            filename = audio_file.stem.lower()
            
            # Try to extract accent from filename
            # Typical format: [number]_[country]_[gender]_[age]
            parts = filename.split('_')
            
            if len(parts) >= 2:
                country_or_accent = parts[1]
            else:
                country_or_accent = "english"
            
            accent_type = self._categorize_accent(country_or_accent)
            speaker_id = f"saa_{audio_file.stem}"
            
            self.master_manifest.append({
                'audio_path': str(audio_file),
                'transcript': passage,
                'accent_type': accent_type,
                'duration_seconds': duration,
                'source_dataset': 'speech_accent_archive',
                'speaker_id': speaker_id,
                'accent_label': country_or_accent
            })
            
            processed += 1
        
        print(f"✓ Processed {processed} Speech Accent Archive samples")
    
    def process_voxpopuli(self) -> None:
        """Extract metadata from VoxPopuli dataset"""
        print("\n" + "="*60)
        print("PROCESSING VOXPOPULI")
        print("="*60)
        
        vp_dir = self.data_raw_dir / "voxpopuli"
        if not vp_dir.exists():
            print("✗ VoxPopuli directory not found")
            return
        
        # VoxPopuli structure: speech_16khz/[language]/[year]/[id].wav
        processed = 0
        
        for audio_file in tqdm(list(vp_dir.rglob("*.wav")), desc="Processing VoxPopuli"):
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                duration = len(audio) / sr
            except:
                continue
            
            # VoxPopuli contains European speakers
            accent_type = "european"
            speaker_id = f"vp_{audio_file.stem}"
            
            # Try to extract transcript from manifest
            transcript = ""
            
            self.master_manifest.append({
                'audio_path': str(audio_file),
                'transcript': transcript,
                'accent_type': accent_type,
                'duration_seconds': duration,
                'source_dataset': 'voxpopuli',
                'speaker_id': speaker_id,
                'accent_label': 'european'
            })
            
            processed += 1
        
        print(f"✓ Processed {processed} VoxPopuli samples")
    
    def _categorize_accent(self, accent_label: str) -> str:
        """Categorize accent into one of 5 main types"""
        accent_label = accent_label.lower().strip()
        
        for category, accents in self.accent_mapping.items():
            if any(acc in accent_label for acc in accents):
                return category
        
        return "unknown"
    
    def save_master_manifest(self) -> None:
        """Save master manifest CSV"""
        if not self.master_manifest:
            print("✗ No data to save")
            return
        
        df = pd.DataFrame(self.master_manifest)
        output_path = self.manifests_dir / "master.csv"
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Master manifest saved: {output_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Accent distribution:\n{df['accent_type'].value_counts()}")
    
    def run(self) -> None:
        """Run dataset organization"""
        print("\n" + "🎵 DATASET ORGANIZATION - PHASE 3.1 & 3.2 🎵".center(60))
        print("="*60)
        
        print("Processing datasets...")
        self.process_common_voice()
        self.process_librispeech()
        self.process_speech_accent_archive()
        self.process_voxpopuli()
        
        self.save_master_manifest()


if __name__ == "__main__":
    organizer = DatasetOrganizer()
    organizer.run()
