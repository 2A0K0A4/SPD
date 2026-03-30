"""
feature_extraction.py — Feature Extraction Pipeline
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar
"""

import os
import numpy as np
import torch
import whisper
import librosa
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper requires 16kHz
CHUNK_SECONDS = 30           # Whisper's expected input length
OVERLAP_SECONDS = 5          # Overlap between chunks to avoid cutting words
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS


# ──────────────────────────────────────────────
# STEP 1: Generate Log-Mel Spectrogram
# ──────────────────────────────────────────────
def extract_log_mel(audio: np.ndarray) -> torch.Tensor:
    """
    Generate an 80-channel log-mel spectrogram from audio.
    This is Whisper's expected input format:
      - 80 mel filterbanks
      - 25ms window, 10ms hop
      - Log-scaled
      - Padded/trimmed to exactly 30 seconds
    """
    # Pad or trim to exactly 30 seconds
    audio = whisper.pad_or_trim(audio)
    audio = audio.astype(np.float32)  # ensure float32 — whisper requires this
    # Generate log-mel spectrogram (shape: [80, 3000])
    mel = whisper.log_mel_spectrogram(audio)

    return mel


# ──────────────────────────────────────────────
# STEP 2: Chunk Long Audio
# ──────────────────────────────────────────────
def chunk_audio(audio: np.ndarray) -> list:
    """
    Split audio longer than 30 seconds into overlapping chunks.
    
    - Chunk size: 30 seconds
    - Overlap: 5 seconds
    - Each chunk is padded to exactly 30 seconds if shorter
    
    Returns a list of (start_time_seconds, audio_chunk) tuples.
    """
    duration = len(audio) / SAMPLE_RATE

    # Short audio — no chunking needed
    if duration <= CHUNK_SECONDS:
        return [(0.0, audio)]

    chunks = []
    step = CHUNK_SAMPLES - OVERLAP_SAMPLES  # advance by 25 seconds each time
    start = 0

    while start < len(audio):
        end = start + CHUNK_SAMPLES
        chunk = audio[start:end]
        start_time = start / SAMPLE_RATE
        chunks.append((start_time, chunk))
        start += step

    print(f"  Audio duration: {duration:.1f}s → Split into {len(chunks)} chunks "
          f"({CHUNK_SECONDS}s each, {OVERLAP_SECONDS}s overlap)")
    return chunks


# ──────────────────────────────────────────────
# STEP 3: Extract Features from Chunks
# ──────────────────────────────────────────────
def extract_features_from_file(audio: np.ndarray) -> list:
    """
    Given a preprocessed audio array, return a list of log-mel spectrograms,
    one per chunk.

    Returns:
        List of dicts:
            - "start_time": float (seconds)
            - "mel": torch.Tensor of shape [80, 3000]
    """
    chunks = chunk_audio(audio)
    features = []

    for start_time, chunk in chunks:
        mel = extract_log_mel(chunk)
        features.append({
            "start_time": start_time,
            "mel": mel
        })

    return features


# ──────────────────────────────────────────────
# STEP 4: Merge Transcriptions from Overlapping Chunks
# ──────────────────────────────────────────────
def merge_transcriptions(segments: list) -> str:
    """
    Merge transcription segments from overlapping chunks into one clean string.
    
    Each segment is a dict with:
        - "start_time": float
        - "text": str
    
    Simple strategy: deduplicate by trimming the overlap region from each chunk.
    """
    if not segments:
        return ""

    if len(segments) == 1:
        return segments[0]["text"].strip()

    merged = segments[0]["text"].strip()

    for i in range(1, len(segments)):
        current_text = segments[i]["text"].strip()
        overlap_duration = OVERLAP_SECONDS

        # Rough word-based deduplication:
        # Drop the first few words of the current segment that likely overlap
        words = current_text.split()
        words_per_second = max(len(words) / CHUNK_SECONDS, 1)
        overlap_words = int(overlap_duration * words_per_second)
        trimmed = " ".join(words[overlap_words:])

        if trimmed:
            merged += " " + trimmed

    return merged.strip()


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
def test_feature_extraction(file_path: str):
    """Test feature extraction on a single audio file."""
    print(f"\n🔬 Testing feature extraction on: {file_path}\n")

    # Load audio (assumes already preprocessed at 16kHz)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Loaded: {duration:.2f} seconds, {len(audio)} samples\n")

    # Extract features
    features = extract_features_from_file(audio)

    # Print summary
    print(f"\n✅ Extracted {len(features)} chunk(s):")
    for i, feat in enumerate(features):
        mel = feat["mel"]
        print(f"   Chunk {i+1}: start={feat['start_time']:.1f}s | "
              f"mel shape={tuple(mel.shape)} | dtype={mel.dtype}")

    # Verify shape is correct for Whisper
    expected_shape = (80, 3000)
    for feat in features:
        assert feat["mel"].shape == expected_shape, \
            f"❌ Wrong shape: {feat['mel'].shape}, expected {expected_shape}"

    print(f"\n✅ All chunks have correct shape {expected_shape} — ready for Whisper!")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to a preprocessed WAV file")
    args = parser.parse_args()

    test_feature_extraction(args.input)
