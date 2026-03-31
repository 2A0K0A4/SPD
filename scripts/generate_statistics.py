"""
Dataset Statistics Report Generator - Phase 3.4
Generates comprehensive analysis of dataset with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DatasetStatistics:
    """Generate comprehensive dataset statistics report"""
    
    def __init__(self, manifest_path: str = "data/manifests/master.csv",
                 output_dir: str = "docs"):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        self.df = pd.read_csv(self.manifest_path)
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'total_files': len(self.df),
            'total_hours': self.df['duration_seconds'].sum() / 3600,
            'avg_duration': self.df['duration_seconds'].mean(),
            'min_duration': self.df['duration_seconds'].min(),
            'max_duration': self.df['duration_seconds'].max(),
            'unique_speakers': self.df['speaker_id'].nunique(),
            'unique_accents': self.df['accent_type'].nunique(),
            'unique_sources': self.df['source_dataset'].nunique(),
        }
        
        # By accent type
        stats['by_accent'] = self.df.groupby('accent_type').agg({
            'audio_path': 'count',
            'duration_seconds': ['sum', 'mean']
        }).round(2)
        
        # By dataset source
        stats['by_source'] = self.df.groupby('source_dataset').agg({
            'audio_path': 'count',
            'duration_seconds': 'sum'
        }).round(2)
        
        return stats
    
    def generate_visualizations(self) -> None:
        """Generate charts and visualizations"""
        print("Generating visualizations...")
        
        # 1. Hours per accent
        fig, ax = plt.subplots(figsize=(10, 6))
        accent_hours = self.df.groupby('accent_type')['duration_seconds'].sum() / 3600
        accent_hours.sort_values(ascending=False).plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Audio Hours per Accent Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Accent Type', fontsize=12)
        ax.set_ylabel('Hours', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'accent_distribution_hours.png', dpi=150)
        plt.close()
        
        # 2. Sample count per accent
        fig, ax = plt.subplots(figsize=(10, 6))
        accent_counts = self.df['accent_type'].value_counts()
        accent_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Sample Count per Accent Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Accent Type', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'accent_distribution_samples.png', dpi=150)
        plt.close()
        
        # 3. Dataset source composition
        fig, ax = plt.subplots(figsize=(10, 6))
        source_counts = self.df['source_dataset'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
        ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Dataset Source Composition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'dataset_source_composition.png', dpi=150)
        plt.close()
        
        # 4. Duration distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df['duration_seconds'].hist(bins=50, ax=ax, color='green', alpha=0.7)
        ax.set_title('Duration Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Duration (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'duration_distribution.png', dpi=150)
        plt.close()
        
        # 5. Speakers per accent
        fig, ax = plt.subplots(figsize=(10, 6))
        speakers_per_accent = self.df.groupby('accent_type')['speaker_id'].nunique()
        speakers_per_accent.sort_values(ascending=False).plot(kind='bar', ax=ax, color='mediumpurple')
        ax.set_title('Unique Speakers per Accent Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Accent Type', fontsize=12)
        ax.set_ylabel('Number of Speakers', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'images' / 'speakers_per_accent.png', dpi=150)
        plt.close()
        
        print("✓ Visualizations saved to docs/images/")
    
    def generate_report(self, stats: Dict) -> str:
        """Generate markdown report"""
        report = """# Dataset Statistics Report

## Executive Summary

This report provides a comprehensive analysis of the accent recognition dataset compiled from multiple sources.

## Dataset Overview

"""
        
        report += f"""### Total Statistics
- **Total Audio Files:** {stats['total_files']:,}
- **Total Duration:** {stats['total_hours']:.1f} hours
- **Average Duration:** {stats['avg_duration']:.2f} seconds
- **Duration Range:** {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s
- **Unique Speakers:** {stats['unique_speakers']:,}
- **Unique Accent Types:** {stats['unique_accents']}
- **Data Sources:** {stats['unique_sources']}

"""
        
        # Breakdown by accent
        report += """### Breakdown by Accent Type

| Accent Type | Samples | Total Hours | Avg Duration |
|---|---|---|---|
"""
        
        for accent in stats['by_accent'].index:
            samples = len(self.df[self.df['accent_type'] == accent])
            hours = self.df[self.df['accent_type'] == accent]['duration_seconds'].sum() / 3600
            avg_dur = self.df[self.df['accent_type'] == accent]['duration_seconds'].mean()
            report += f"| {accent} | {samples} | {hours:.1f} | {avg_dur:.2f}s |\n"
        
        # Breakdown by source
        report += """
### Breakdown by Dataset Source

| Source | Samples | Total Hours |
|---|---|---|
"""
        
        for source in stats['by_source'].index:
            samples = len(self.df[self.df['source_dataset'] == source])
            hours = self.df[self.df['source_dataset'] == source]['duration_seconds'].sum() / 3600
            report += f"| {source} | {samples} | {hours:.1f} |\n"
        
        # Visualizations section
        report += """

## Visualizations

![Accent Distribution (Hours)](images/accent_distribution_hours.png)
*Figure 1: Total audio hours per accent type*

![Accent Distribution (Samples)](images/accent_distribution_samples.png)
*Figure 2: Sample count per accent type*

![Dataset Source](images/dataset_source_composition.png)
*Figure 3: Dataset source composition*

![Duration Distribution](images/duration_distribution.png)
*Figure 4: Duration distribution of audio samples*

![Speakers](images/speakers_per_accent.png)
*Figure 5: Unique speakers per accent type*

## Data Quality Notes

- All statistics calculated from master manifest
- Duration values computed from actual audio loading
- Missing values excluded from analysis
- Speaker independence: Each speaker appears in only one split (handled in split phase)

## Recommendations

1. **Minimum samples**: Ensure at least 50 samples per accent in test set
2. **Speaker balance**: Try to balance speakers across train/val/test
3. **Duration**: Consider stratified sampling by duration ranges
4. **Data augmentation**: Consider augmentation for underrepresented accents

## Next Steps

- Phase 3.3: Create balanced train/validation/test splits
- Phase 4: Quality assurance and data validation
- Phase 5: Begin model training and evaluation

---

*Report generated automatically*
"""
        
        return report
    
    def run(self) -> None:
        """Generate complete statistics report"""
        print("\n" + "DATASET STATISTICS - PHASE 3.4".center(60))
        print("="*60)
        
        print("Calculating statistics...")
        stats = self.calculate_statistics()
        
        print("\nDataset Overview:")
        print(f"  Total files: {stats['total_files']:,}")
        print(f"  Total duration: {stats['total_hours']:.1f} hours")
        print(f"  Unique speakers: {stats['unique_speakers']:,}")
        print(f"  Accents: {stats['unique_accents']}")
        
        self.generate_visualizations()
        
        report = self.generate_report(stats)
        report_path = self.output_dir / "dataset_statistics.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved: {report_path}")


if __name__ == "__main__":
    stats = DatasetStatistics()
    stats.run()
