"""
preprocess.py — Audio Preprocessing Pipeline
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar
"""

import os
import csv
import argparse
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper requires 16kHz
TARGET_DBFS = -20            # Target volume normalization level
SILENCE_TOP_DB = 20          # Threshold for silence trimming
OUTPUT_FORMAT = "wav"        # Output file format


# ──────────────────────────────────────────────
# STEP 1: Load Audio
# ──────────────────────────────────────────────
def load_audio(file_path: str) -> np.ndarray:
    """Load an audio file and resample to 16kHz mono."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio


# ──────────────────────────────────────────────
# STEP 2: Normalize Volume to -20 dBFS
# ──────────────────────────────────────────────
def normalize_volume(audio: np.ndarray) -> np.ndarray:
    """Normalize audio volume to TARGET_DBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    current_dBFS = 20 * np.log10(rms + 1e-9)
    gain = 10 ** ((TARGET_DBFS - current_dBFS) / 20)
    return audio * gain


# ──────────────────────────────────────────────
# STEP 3: Noise Reduction
# ──────────────────────────────────────────────
def reduce_noise(audio: np.ndarray) -> np.ndarray:
    """Apply spectral noise reduction."""
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE)


# ──────────────────────────────────────────────
# STEP 4: Trim Silence
# ──────────────────────────────────────────────
def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Remove leading and trailing silence."""
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
    return audio_trimmed


# ──────────────────────────────────────────────
# STEP 5: Voice Activity Detection (VAD)
# ──────────────────────────────────────────────
def apply_vad(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Remove non-speech segments using energy-based VAD.
    Keeps only frames where RMS energy is above a threshold.
    """
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))
    threshold = np.mean(rms_per_frame) * 0.5  # adaptive threshold

    # Build a mask of which samples to keep
    keep_mask = np.zeros(len(audio), dtype=bool)
    for i, keep in enumerate(rms_per_frame > threshold):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        if keep:
            keep_mask[start:end] = True

    audio_vad = audio[keep_mask]

    # Fallback: if VAD removes everything, return original
    if len(audio_vad) < SAMPLE_RATE * 0.5:
        return audio
    return audio_vad


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────
def preprocess(file_path: str) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single audio file.
    Returns a clean numpy array at 16kHz.
    """
    print(f"  [1/5] Loading:          {os.path.basename(file_path)}")
    audio = load_audio(file_path)

    print(f"  [2/5] Normalizing volume...")
    audio = normalize_volume(audio)

    print(f"  [3/5] Reducing noise...")
    audio = reduce_noise(audio)

    print(f"  [4/5] Applying VAD...")
    audio = apply_vad(audio)

    print(f"  [5/5] Trimming silence...")
    audio = trim_silence(audio)

    return audio


# ──────────────────────────────────────────────
# SAVE OUTPUT
# ──────────────────────────────────────────────
def save_audio(audio: np.ndarray, output_path: str):
    """Save preprocessed audio to disk as WAV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE)


# ──────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────
def process_dataset(input_dir: str, output_dir: str, manifest_path: str):
    """
    Process all WAV files in input_dir, save to output_dir,
    and write a manifest CSV with columns: audio_path, transcript, accent_type, split.

    Expects input_dir to have subfolders named by accent:
        input_dir/
            arabic/
                file1.wav
                file1.txt   ← transcript (same name, .txt)
            south_asian/
            east_asian/
            european/
            north_american/
    """
    accent_types = ["arabic", "south_asian", "east_asian", "european", "north_american"]
    manifest_rows = []

    for accent in accent_types:
        accent_input = os.path.join(input_dir, accent)
        accent_output = os.path.join(output_dir, accent)

        if not os.path.exists(accent_input):
            print(f"[SKIP] Folder not found: {accent_input}")
            continue

        wav_files = list(Path(accent_input).glob("*.wav"))
        print(f"\n[{accent.upper()}] Processing {len(wav_files)} files...")

        for wav_file in wav_files:
            print(f"\nProcessing: {wav_file.name}")
            try:
                # Preprocess audio
                audio = preprocess(str(wav_file))

                # Save to output directory
                out_path = os.path.join(accent_output, wav_file.name)
                save_audio(audio, out_path)

                # Read transcript if available
                txt_file = wav_file.with_suffix(".txt")
                transcript = ""
                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()

                manifest_rows.append({
                    "audio_path": out_path,
                    "transcript": transcript,
                    "accent_type": accent,
                    "split": "train"  # Alaa will update this with proper splits
                })

            except Exception as e:
                print(f"  [ERROR] Failed to process {wav_file.name}: {e}")

    # Write manifest CSV
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "accent_type", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n✅ Manifest saved to: {manifest_path}")
    print(f"✅ Total files processed: {len(manifest_rows)}")


# ──────────────────────────────────────────────
# QUICK TEST (single file)
# ──────────────────────────────────────────────
def test_single_file(file_path: str, output_path: str):
    """Test the pipeline on a single file."""
    print(f"\n🎙️  Testing preprocessing on: {file_path}\n")
    audio = preprocess(file_path)
    save_audio(audio, output_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"\n✅ Done! Output saved to: {output_path}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Samples: {len(audio)}")
    print(f"   Sample rate: {SAMPLE_RATE} Hz")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Preprocessing Pipeline")
    parser.add_argument("--mode", choices=["test", "batch"], default="test",
                        help="'test' for single file, 'batch' for full dataset")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--manifest", default="./data/manifest.csv",
                        help="Path to save manifest CSV (batch mode only)")
    args = parser.parse_args()

    if args.mode == "test":
        test_single_file(args.input, args.output)
    elif args.mode == "batch":
        process_dataset(args.input, args.output, args.manifest)
