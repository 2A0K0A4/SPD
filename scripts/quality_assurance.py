"""
Data Quality Assurance Script - Phase 4
Validates data quality and identifies issues
"""

import os
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityAssurance:
    """Comprehensive data quality validation"""
    
    def __init__(self, manifest_path: str = "data/manifests/master.csv",
                 output_dir: str = "docs"):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        self.df = pd.read_csv(self.manifest_path)
        self.issues = []
        self.stats = {
            'total_validated': 0,
            'missing_files': 0,
            'corrupted_files': 0,
            'duration_mismatches': 0,
            'too_short': 0,
            'empty_transcripts': 0,
            'quality_issues': 0
        }
    
    def validate_file_existence(self) -> None:
        """Check if all files exist"""
        print("\n" + "="*60)
        print("VALIDATING FILE EXISTENCE")
        print("="*60)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking files"):
            path = row['audio_path']
            
            if not os.path.exists(path):
                self.issues.append(f"[MISSING] {path}")
                self.stats['missing_files'] += 1
    
    def validate_file_integrity(self) -> None:
        """Check if audio files can be loaded"""
        print("\n" + "="*60)
        print("VALIDATING FILE INTEGRITY")
        print("="*60)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading files"):
            path = row['audio_path']
            
            if not os.path.exists(path):
                continue
            
            try:
                audio, sr = librosa.load(path, sr=16000)
                self.stats['total_validated'] += 1
            except Exception as e:
                self.issues.append(f"[CORRUPTED] {path} - {str(e)[:50]}")
                self.stats['corrupted_files'] += 1
    
    def validate_duration(self) -> None:
        """Check if duration matches metadata"""
        print("\n" + "="*60)
        print("VALIDATING DURATION")
        print("="*60)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking duration"):
            path = row['audio_path']
            expected_duration = row['duration_seconds']
            
            if not os.path.exists(path):
                continue
            
            try:
                audio, sr = librosa.load(path, sr=16000)
                actual_duration = len(audio) / sr
                
                # Allow 1 second tolerance
                if abs(actual_duration - expected_duration) > 1.0:
                    self.issues.append(
                        f"[DURATION_MISMATCH] {path} - "
                        f"Expected: {expected_duration:.2f}s, Actual: {actual_duration:.2f}s"
                    )
                    self.stats['duration_mismatches'] += 1
            except:
                pass
    
    def validate_minimum_duration(self, min_duration: float = 0.1) -> None:
        """Check for very short files"""
        print("\n" + "="*60)
        print("VALIDATING MINIMUM DURATION")
        print("="*60)
        print(f"Minimum acceptable duration: {min_duration} seconds")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking min duration"):
            path = row['audio_path']
            duration = row['duration_seconds']
            
            if duration < min_duration:
                self.issues.append(f"[TOO_SHORT] {path} - {duration:.3f}s")
                self.stats['too_short'] += 1
    
    def validate_transcripts(self) -> None:
        """Check for empty transcripts"""
        print("\n" + "="*60)
        print("VALIDATING TRANSCRIPTS")
        print("="*60)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking transcripts"):
            transcript = row.get('transcript', '')
            path = row['audio_path']
            
            if pd.isna(transcript) or len(str(transcript).strip()) == 0:
                self.issues.append(f"[EMPTY_TRANSCRIPT] {path}")
                self.stats['empty_transcripts'] += 1
    
    def analyze_audio_quality(self, sample_rate: int = 16000) -> None:
        """Analyze audio quality metrics"""
        print("\n" + "="*60)
        print("ANALYZING AUDIO QUALITY")
        print("="*60)
        
        for idx, row in tqdm(self.df.head(1000).iterrows(), total=min(1000, len(self.df)), 
                            desc="Analyzing quality"):
            path = row['audio_path']
            
            if not os.path.exists(path):
                continue
            
            try:
                audio, sr = librosa.load(path, sr=sample_rate)
                
                # Check for clipping (values at max)
                max_val = np.abs(audio).max()
                if max_val > 0.95:  # Close to max value
                    self.issues.append(f"[CLIPPING] {path} - Peak: {max_val:.3f}")
                    self.stats['quality_issues'] += 1
                
                # Check for silence
                energy = np.sum(audio ** 2) / len(audio)
                if energy < 1e-6:
                    self.issues.append(f"[SILENT] {path}")
                    self.stats['quality_issues'] += 1
            
            except:
                pass
    
    def generate_quality_report(self) -> str:
        """Generate quality report"""
        report = """# Data Quality Assurance Report

## Summary

"""
        report += f"- **Total Validated:** {self.stats['total_validated']:,}\n"
        report += f"- **Missing Files:** {self.stats['missing_files']}\n"
        report += f"- **Corrupted Files:** {self.stats['corrupted_files']}\n"
        report += f"- **Duration Mismatches:** {self.stats['duration_mismatches']}\n"
        report += f"- **Too Short (<0.1s):** {self.stats['too_short']}\n"
        report += f"- **Empty Transcripts:** {self.stats['empty_transcripts']}\n"
        report += f"- **Quality Issues:** {self.stats['quality_issues']}\n"
        
        total_issues = sum(
            v for k, v in self.stats.items() if k != 'total_validated'
        )
        
        report += f"\n**Total Issues Found:** {total_issues}\n"
        
        if total_issues == 0:
            report += "\n✓ **No issues found!** Dataset quality is excellent.\n"
        elif total_issues < len(self.df) * 0.01:
            report += f"\n✓ **Good quality!** Less than 1% of files have issues.\n"
        else:
            percentage = total_issues / len(self.df) * 100
            report += f"\n⚠ **{percentage:.1f}% of files have issues.** Review recommended.\n"
        
        if self.issues:
            report += "\n## Detailed Issues\n\n"
            report += "```\n"
            for issue in self.issues[:100]:  # First 100 issues
                report += f"{issue}\n"
            
            if len(self.issues) > 100:
                report += f"\n... and {len(self.issues) - 100} more issues\n"
            
            report += "```\n"
        
        report += "\n## Recommendations\n\n"
        
        if self.stats['missing_files'] > 0:
            report += f"- Remove {self.stats['missing_files']} files with missing paths\n"
        if self.stats['corrupted_files'] > 0:
            report += f"- Re-download or replace {self.stats['corrupted_files']} corrupted files\n"
        if self.stats['empty_transcripts'] > 0:
            report += f"- Review or add {self.stats['empty_transcripts']} empty transcripts\n"
        if total_issues == 0:
            report += "- Dataset is ready for next phase!\n"
        
        return report
    
    def run(self) -> None:
        """Run complete QA pipeline"""
        print("\n" + "🎵 DATA QUALITY ASSURANCE - PHASE 4 🎵".center(60))
        print("="*60)
        
        self.validate_file_existence()
        self.validate_file_integrity()
        self.validate_duration()
        self.validate_minimum_duration()
        self.validate_transcripts()
        self.analyze_audio_quality()
        
        # Generate report
        report = self.generate_quality_report()
        report_path = self.output_dir / "data_quality_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Quality report saved: {report_path}")
        print(f"\nIssues found: {len(self.issues)}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(report.split("## Detailed Issues")[0])  # Print first part


if __name__ == "__main__":
    qa = DataQualityAssurance()
    qa.run()
