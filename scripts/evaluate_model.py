"""
Model Evaluation and Testing Framework - Phase 5
Tests trained model performance on test samples
"""

import os
import json
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Tuple, List
from jiwer import wer, cer
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates model performance on test samples"""
    
    def __init__(self, test_samples_dir: str = "tests/samples",
                 results_dir: str = "tests/results"):
        self.test_samples_dir = Path(test_samples_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.accents = ['arabic', 'south_asian', 'east_asian', 'european', 'north_american']
        self.results = []
    
    def transcribe_audio(self, audio_path: str) -> Tuple[str, float]:
        """
        Transcribe audio file using the trained model.
        This is a placeholder - replace with actual model inference.
        """
        # TODO: Integrate with actual trained model
        print(f"[MOCK] Transcribing: {audio_path}")
        return "", 0.0  # (transcript, processing_time)
    
    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict:
        """Calculate WER and CER metrics"""
        try:
            word_error = wer(reference, hypothesis)
            char_error = cer(reference, hypothesis)
            
            return {
                'wer': word_error,
                'cer': char_error,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'wer': None,
                'cer': None,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_accent(self, accent: str, num_samples: int = 10) -> None:
        """Evaluate model on samples of specific accent"""
        print(f"\nEvaluating {accent}...")
        
        accent_dir = self.test_samples_dir / accent
        if not accent_dir.exists():
            print(f"✗ No samples found for {accent}")
            return
        
        audio_files = list(accent_dir.glob("*.wav")) + list(accent_dir.glob("*.mp3"))
        audio_files = audio_files[:num_samples]
        
        if not audio_files:
            print(f"✗ No audio files found in {accent_dir}")
            return
        
        accent_results = []
        
        for audio_file in audio_files:
            # Load reference transcript
            transcript_file = audio_file.with_suffix('.txt')
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    reference = f.read().strip()
            else:
                reference = ""
            
            # Transcribe
            hypothesis, proc_time = self.transcribe_audio(str(audio_file))
            
            # Calculate metrics
            metrics = self.calculate_metrics(reference, hypothesis)
            
            result = {
                'accent': accent,
                'audio_file': audio_file.name,
                'reference': reference,
                'hypothesis': hypothesis,
                'processing_time': proc_time,
                'wer': metrics.get('wer'),
                'cer': metrics.get('cer'),
                'timestamp': datetime.now().isoformat()
            }
            
            accent_results.append(result)
            self.results.append(result)
        
        print(f"✓ Evaluated {len(accent_results)} samples for {accent}")
    
    def generate_evaluation_report(self) -> str:
        """Generate evaluation report"""
        if not self.results:
            return "No evaluation results found."
        
        df = pd.DataFrame(self.results)
        
        report = """# Model Evaluation Report

## Executive Summary

"""
        
        # Overall statistics
        total_samples = len(df)
        avg_wer = df['wer'].mean()
        avg_cer = df['cer'].mean()
        
        report += f"- **Total Test Samples:** {total_samples}\n"
        report += f"- **Average WER:** {avg_wer:.2%}\n"
        report += f"- **Average CER:** {avg_cer:.2%}\n"
        
        # By accent results
        report += "\n## Results by Accent Type\n\n"
        report += "| Accent | Samples | Avg WER | Avg CER | Avg Processing Time |\n"
        report += "|---|---|---|---|---|\n"
        
        for accent in self.accents:
            accent_df = df[df['accent'] == accent]
            if len(accent_df) > 0:
                samples = len(accent_df)
                avg_wer = accent_df['wer'].mean()
                avg_cer = accent_df['cer'].mean()
                avg_time = accent_df['processing_time'].mean()
                
                report += f"| {accent} | {samples} | {avg_wer:.2%} | {avg_cer:.2%} | {avg_time:.3f}s |\n"
        
        # Overall
        report += f"| **Overall** | **{len(df)}** | **{df['wer'].mean():.2%}** | **{df['cer'].mean():.2%}** | **{df['processing_time'].mean():.3f}s** |\n"
        
        # Detailed results
        report += "\n## Detailed Results\n\n"
        
        for accent in self.accents:
            accent_df = df[df['accent'] == accent]
            if len(accent_df) == 0:
                continue
            
            report += f"\n### {accent.upper()}\n\n"
            
            for idx, row in accent_df.iterrows():
                report += f"**Sample {idx+1}: {row['audio_file']}**\n"
                report += f"- Reference: {row['reference'][:100]}{'...' if len(row['reference']) > 100 else ''}\n"
                report += f"- Hypothesis: {row['hypothesis'][:100]}{'...' if len(row['hypothesis']) > 100 else ''}\n"
                report += f"- WER: {row['wer']:.2%} | CER: {row['cer']:.2%} | Time: {row['processing_time']:.3f}s\n\n"
        
        return report
    
    def run(self) -> None:
        """Run evaluation pipeline"""
        print("\n" + "MODEL EVALUATION - PHASE 5".center(60))
        print("="*60)
        
        print(f"Test samples directory: {self.test_samples_dir}")
        print(f"Results directory: {self.results_dir}\n")
        
        # Evaluate each accent
        for accent in self.accents:
            self.evaluate_accent(accent, num_samples=10)
        
        # Generate report
        report = self.generate_evaluation_report()
        report_path = self.results_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Evaluation report saved: {report_path}")
        
        # Save detailed results as JSON
        results_json = self.results_dir / "evaluation_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✓ Detailed results saved: {results_json}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
