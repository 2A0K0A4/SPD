# SPD: Speech Processing & Dialect Recognition

A comprehensive system for processing multilingual speech data, extracting acoustic features, and recognizing speaker dialects/accents using machine learning.

## Overview

SPD (Speech Processing & Dialect) is designed to:
- Process multi-source speech datasets (Common Voice, LibriSpeech, Speech Accent Archive, VoxPopuli)
- Extract comprehensive acoustic and prosodic features using librosa and scipy
- Organize and validate audio data with quality assurance checks
- Build balanced train/validation/test splits for machine learning
- Train and evaluate accent recognition models
- Generate statistical analysis and visualizations

## Project Structure

```
SPD/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── common_voice/
│   │   ├── librispeech/
│   │   ├── speech_accent_archive/
│   │   └── voxpopuli/
│   ├── manifests/                    # CSV metadata files
│   │   ├── master.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── features/                     # Extracted features
│   └── accent_mapping.json           # Accent categorization
├── scripts/
│   ├── generate_test_data.py        # Create synthetic datasets
│   ├── organize_datasets.py         # Process raw data → master.csv
│   ├── split_data.py                # Create train/val/test splits
│   ├── generate_statistics.py       # Dataset analysis & visualizations
│   ├── quality_assurance.py         # Validate dataset quality
│   ├── evaluate_model.py            # Evaluate model performance
│   └── feature_extraction.py        # Extract acoustic features
├── src/
│   ├── models/                      # Model architectures
│   ├── preprocessing/               # Audio preprocessing
│   └── utils/                       # Utility functions
├── tests/
│   ├── samples/                     # Test audio samples by accent
│   ├── results/                     # Test evaluation results
│   └── test_pipeline.py             # Integration tests
├── docs/
│   ├── images/                      # Visualizations & charts
│   ├── README.md                    # This file
│   ├── sdp2_final_report.md        # Technical report
│   ├── user_testing_report.md      # User feedback & bugs
│   └── project_poster.txt          # Poster outline
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── coordinator.py                  # CLI interface
└── main.py                          # Entry point
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git version control

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aalaei/SPD.git
   cd SPD
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv accent-env
   
   # On Windows:
   accent-env\Scripts\activate
   
   # On Linux/macOS:
   source accent-env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import librosa, pandas, torch; print('✓ Core dependencies installed')"
   ```

## Usage

### Quick Start

```bash
# 1. Generate test datasets (if data not available)
python scripts/generate_test_data.py

# 2. Organize datasets into master manifest
python scripts/organize_datasets.py

# 3. Create balanced train/validation/test splits
python scripts/split_data.py

# 4. Generate dataset statistics and visualizations
python scripts/generate_statistics.py

# 5. Run quality assurance validation
python scripts/quality_assurance.py

# 6. Evaluate model performance
python scripts/evaluate_model.py
```

### CLI Interface

```bash
# Interactive menu-driven interface
python coordinator.py

# Main entry point
python main.py
```

## Dataset Structure

### Accent Categories

The system recognizes 5 major accent categories:

1. **Arabic** - Egyptian, Levantine, Gulf, Saudi, Moroccan, Tunisian
2. **South Asian** - Indian, Pakistani, Bangladeshi, Nepali, Sri Lankan
3. **East Asian** - Chinese, Mandarin, Japanese, Korean, Vietnamese, Thai
4. **European** - French, German, Spanish, Italian, Russian, Polish, Dutch
5. **North American** - American (baseline), Canadian

### Master Manifest Format

```csv
audio_path,transcript,accent_type,duration_seconds,source_dataset,speaker_id
data/raw/common_voice/cv-valid-test/sample_001.mp3,hello world,european,2.45,common_voice,speaker_cv_001
data/raw/librispeech/dev-clean/910/143699/910-143699-0011.flac,the presentation,north_american,3.12,librispeech,910
...
```

## Features

### Data Processing
- **Multi-source ingestion**: Unified processing for 4 different dataset formats
- **Accent categorization**: Automatic mapping of 50+ accent variants to 5 main categories
- **Speaker independence**: Train/val/test splits ensure no speaker leakage
- **Duration analysis**: Identifies and handles variable-length speech

### Quality Assurance
- File existence validation
- Audio integrity checking (corrupt files detection)
- Duration mismatch identification
- Minimum duration enforcement (0.1s threshold)
- Transcript validation
- Audio quality analysis (SNR, clipping detection)

### Feature Extraction
- **Acoustic features**: MFCCs, spectral centroid, spectral rolloff, zero-crossing rate
- **Prosodic features**: Pitch, energy, formants
- **Time-frequency representations**: Spectrograms, mel-spectrograms
- **Statistical aggregations**: Mean, std, min, max over time

### Evaluation
- Word Error Rate (WER) calculation using jiwer
- Character Error Rate (CER) computation
- Per-accent performance metrics
- Confusion matrices by accent type
- F1-score and accuracy reporting

## Configuration

### Accent Mapping

Edit `data/accent_mapping.json` to customize accent categories:

```json
{
  "arabic": ["egyptian", "levantine", "gulf", "saudi"],
  "south_asian": ["indian", "pakistani", "bangladeshi"],
  "east_asian": ["chinese", "mandarin", "japanese", "korean"],
  "european": ["french", "german", "spanish", "italian"],
  "north_american": ["american", "canadian"]
}
```

### Dataset Paths

Configure in scripts:
- Common Voice: `data/raw/common_voice/validated/`
- LibriSpeech: `data/raw/librispeech/`
- Speech Accent Archive: `data/raw/speech_accent_archive/`
- VoxPopuli: `data/raw/voxpopuli/`

## Outputs

### Generated Files
- **manifests/master.csv**: Unified dataset metadata
- **manifests/train.csv**: 80% training samples
- **manifests/validation.csv**: 10% validation samples
- **manifests/test.csv**: 10% test samples
- **dataset_statistics.md**: Statistical summary with visualizations
- **data_quality_report.md**: Quality validation results
- **evaluation_report.md**: Model performance metrics
- **images/**: PNG visualizations (5+ charts)

## Pipeline

```
Raw Datasets
    ↓
generate_test_data.py (synthetic data generation)
    ↓
organize_datasets.py (master.csv creation)
    ↓
split_data.py (train/val/test splits)
    ↓
generate_statistics.py (analysis & visualizations)
    ↓
quality_assurance.py (validation & QC)
    ↓
feature_extraction.py (acoustic features)
    ↓
model training & evaluation
```

## Performance Metrics

### Dataset Statistics (Test Run)
- **Total files**: 25 samples
- **Unique speakers**: 25
- **Accent distribution**: 5 categories represented
- **Duration range**: 0.1-3.0 seconds
- **Source datasets**: 4 sources (CV, LibriSpeech, SAA, VoxPopuli)

### Quality Metrics
- **File integrity**: 100% (0 corrupted files)
- **Transcript completeness**: 60% (synthetic data limitation)
- **Duration validation**: 100% (all files > 0.1s)

## Dependencies

### Core Dependencies
- **librosa** (0.10.0): Audio processing and feature extraction
- **pandas** (2.1.3): Data manipulation and CSV handling
- **numpy** (1.24.3): Numerical computing
- **scipy**: Signal processing and statistics
- **soundfile**: Audio I/O (reading/writing WAV, FLAC)
- **torch**: Deep learning framework
- **transformers**: Pre-trained model access

### Development Dependencies
- **pytest**: Unit testing framework
- **matplotlib**: Visualization
- **seaborn**: Statistical plotting
- **scikit-learn**: Machine learning utilities

## Troubleshooting

### Common Issues

**Q: "UnicodeEncodeError" when generating reports?**
- A: Ensure UTF-8 encoding in file I/O. Use `encoding='utf-8'` in open() calls.

**Q: "No audio files found" in evaluation?**
- A: Create test samples in `tests/samples/<accent>/` directories.

**Q: Out of memory with large datasets?**
- A: Process datasets in batches using the batch_size parameter (future enhancement).

**Q: Module not found errors?**
- A: Activate virtual environment and reinstall: `pip install -r requirements.txt`

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit with clear messages: `git commit -m "feat: description"`
4. Push to branch: `git push origin feature/your-feature`
5. Create Pull Request with detailed description

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py::test_master_manifest -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Notes

- **Dataset processing**: ~1-2 minutes for 1000 files
- **Statistics generation**: <1 minute for 1000 files
- **Feature extraction**: ~5-10 seconds per audio file
- **Model evaluation**: Depends on model architecture and GPU availability

## Future Enhancements

- GPU acceleration for feature extraction
- Batch processing pipeline
- Real-time accent prediction API
- Web interface for dataset management
- Multi-language support
- Automated hyperparameter tuning

## License

Project License: MIT

## Contact & Support

**Author**: Alaa Alaei  
**Email**: alaa@example.com  
**GitHub**: [@aalaei](https://github.com/aalaei)

For issues, feature requests, or contributions, please open an issue or pull request on the GitHub repository.

---

**Last Updated**: 2024-06  
**Version**: 1.0.0
