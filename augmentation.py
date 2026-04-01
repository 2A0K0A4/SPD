"""
augmentation.py — Data Augmentation Pipeline
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar
"""

import numpy as np
import librosa
import random


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
AUGMENT_PROBABILITY = 0.5    # 50% chance each augmentation is applied per sample


# ──────────────────────────────────────────────
# AUGMENTATION 1: Pitch Shifting
# ──────────────────────────────────────────────
def pitch_shift(audio: np.ndarray) -> np.ndarray:
    """Shift pitch randomly between -2 and +2 semitones."""
    n_steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)


# ──────────────────────────────────────────────
# AUGMENTATION 2: Time Stretching
# ──────────────────────────────────────────────
def time_stretch(audio: np.ndarray) -> np.ndarray:
    """Stretch or compress audio speed between 0.9x and 1.1x."""
    rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(audio, rate=rate)


# ──────────────────────────────────────────────
# AUGMENTATION 3: Background Noise Injection
# ──────────────────────────────────────────────
def add_noise(audio: np.ndarray) -> np.ndarray:
    """
    Add Gaussian background noise at a random SNR between 10-20 dB.
    Higher SNR = less noise.
    """
    snr_db = random.uniform(10, 20)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return audio + noise


# ──────────────────────────────────────────────
# AUGMENTATION 4: Speed Perturbation
# ──────────────────────────────────────────────
def speed_perturbation(audio: np.ndarray) -> np.ndarray:
    """
    Resample audio at a slightly different rate (±5%)
    to simulate natural speaking speed variation.
    """
    speed_factor = random.uniform(0.95, 1.05)
    original_length = len(audio)
    new_length = int(original_length / speed_factor)
    audio_resampled = librosa.resample(audio, orig_sr=SAMPLE_RATE,
                                       target_sr=int(SAMPLE_RATE * speed_factor))
    # Resample back to original sample rate
    return librosa.resample(audio_resampled,
                            orig_sr=int(SAMPLE_RATE * speed_factor),
                            target_sr=SAMPLE_RATE)


# ──────────────────────────────────────────────
# FULL AUGMENTATION PIPELINE
# ──────────────────────────────────────────────
def augment(audio: np.ndarray, probability: float = AUGMENT_PROBABILITY) -> np.ndarray:
    """
    Randomly apply augmentations to an audio sample.
    Each augmentation has an independent chance of being applied.

    Args:
        audio: numpy array at 16kHz
        probability: chance each augmentation is applied (default 0.5)

    Returns:
        Augmented audio numpy array
    
    NOTE: Only call this during training, never on validation/test data.
    """
    augmentations = [
        ("Pitch Shift",        pitch_shift),
        ("Time Stretch",       time_stretch),
        ("Background Noise",   add_noise),
        ("Speed Perturbation", speed_perturbation),
    ]

    applied = []
    for name, fn in augmentations:
        if random.random() < probability:
            try:
                audio = fn(audio)
                applied.append(name)
            except Exception as e:
                print(f"  [WARNING] Augmentation '{name}' failed: {e} — skipping")

    if applied:
        print(f"  Applied augmentations: {', '.join(applied)}")
    else:
        print(f"  No augmentations applied this sample")

    return audio


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
def test_augmentation(file_path: str):
    """Test augmentation pipeline on a single audio file."""
    import soundfile as sf

    print(f"\n🎛️  Testing augmentation on: {file_path}\n")
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"  Original: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s\n")

    # Run 3 times to show randomness
    for i in range(3):
        print(f"  Run {i+1}:")
        augmented = augment(audio)
        out_path = file_path.replace(".wav", f"_aug_{i+1}.wav")
        sf.write(out_path, augmented, SAMPLE_RATE)
        print(f"  Saved: {out_path}\n")

    print("✅ Augmentation test complete!")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Augmentation Pipeline")
    parser.add_argument("--input", required=True, help="Path to a WAV file to test augmentation on")
    args = parser.parse_args()

    test_augmentation(args.input)
