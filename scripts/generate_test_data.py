#!/usr/bin/env python3
"""
Test Dataset Generator - Creates mock audio files for demonstration
"""

import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_test_audio(duration=3.0, sr=16000):
    """Generate synthetic test audio"""
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave with some noise
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio += 0.05 * np.random.randn(len(audio))
    return audio.astype(np.float32), sr

# Create test data for each dataset
base_path = Path("data/raw")

# 1. Common Voice test data
cv_path = base_path / "common_voice" / "clips"
cv_path.mkdir(parents=True, exist_ok=True)

accents_cv = ['arabic', 'english', 'indian', 'chinese', 'french']
cv_data = []

for i in range(10):
    accent = random.choice(accents_cv)
    audio, sr = generate_test_audio()
    filename = f"cv_sample_{i:04d}.wav"
    sf.write(cv_path / filename, audio, sr)
    
    cv_data.append({
        'path': f'clips/{filename}',
        'sentence': f'This is a test sentence number {i}',
        'accent': accent,
        'client_id': f'speaker_{i % 5}'
    })

cv_df = pd.DataFrame(cv_data)
cv_df.to_csv(base_path / "common_voice" / "validated.tsv", sep='\t', index=False)
print(f"✓ Created {len(cv_df)} Common Voice test samples")

# 2. LibriSpeech test data
ls_path = base_path / "librispeech" / "train-clean-100" / "123" / "456"
ls_path.mkdir(parents=True, exist_ok=True)

ls_trans = {}
for i in range(10):
    audio, sr = generate_test_audio()
    filename = f"123-456-{i:04d}.flac"
    sf.write(ls_path / filename, audio, sr)
    file_id = f"123-456-{i:04d}"
    ls_trans[file_id] = f"This is test sentence number {i}"

with open(ls_path / "123-456.trans.txt", 'w') as f:
    for file_id, trans in ls_trans.items():
        f.write(f"{file_id} {trans}\n")

print(f"✓ Created {len(ls_trans)} LibriSpeech test samples")

# 3. Speech Accent Archive test data
saa_path = base_path / "speech_accent_archive"
saa_path.mkdir(parents=True, exist_ok=True)

saa_accents = ['arabic', 'egyptian', 'indian', 'chinese', 'french', 'german', 'spanish', 'korean', 'vietnamese']
for i in range(15):
    accent = saa_accents[i % len(saa_accents)]
    audio, sr = generate_test_audio()
    filename = f"{i:04d}_{accent}_0_0.wav"
    sf.write(saa_path / filename, audio, sr)

print(f"✓ Created 15 Speech Accent Archive test samples")

# 4. VoxPopuli test data
vp_path = base_path / "voxpopuli" / "asr" / "en" / "2019"
vp_path.mkdir(parents=True, exist_ok=True)

for i in range(10):
    audio, sr = generate_test_audio()
    filename = f"vp_{i:04d}.wav"
    sf.write(vp_path / filename, audio, sr)

print(f"✓ Created 10 VoxPopuli test samples")

print("\n✅ Test datasets created successfully!")
