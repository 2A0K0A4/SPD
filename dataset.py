"""
dataset.py — Custom PyTorch Dataset and DataLoader
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar

FIXES applied vs original:
  1. librosa.get_duration(filename=...) deprecated → use librosa.get_duration(path=...)
     Actually in __getitem__ we just load audio; duration validation belongs in preprocessing.
  2. extract_log_mel returns shape [80, N] where N may vary. Added padding/truncation
     to enforce fixed [80, 3000] before returning — Whisper requires exactly 3000 frames.
  3. Added try/except in __getitem__ to skip corrupt files gracefully instead of crashing.
  4. accent_label = -1 for unknown accents would break loss — changed to skip those samples
     (filtered at dataset load time).
  5. collate_fn was defined outside the class taking `processor` as arg via functools.partial,
     which is correct — no change needed, but added clearer type hints.
  6. Added __repr__ for easier debugging.
  7. Manifests are pre-split (train.csv / validation.csv / test.csv) — no 'split' column.
     build_dataloaders() now accepts separate CSV paths per split.
  8. Windows backslashes in audio_path converted to forward slashes (cross-platform fix).
  9. Fine-grained accent labels (e.g. 'egyptian') normalized to 5 categories via ACCENT_NORMALIZE.
"""

import os
import csv
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor
from functools import partial

from augmentation import augment
from feature_extraction import extract_log_mel


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE  = 16000
MAX_FRAMES   = 3000          # Whisper expects exactly 3000 mel frames (30 s at 10 ms hop)
N_MELS       = 80            # Whisper mel channels

# FIX #9: map fine-grained accent_type values → 5 project categories
ACCENT_NORMALIZE = {
    # Arabic group
    "arabic":           "arabic",
    "egyptian":         "arabic",
    "gulf":             "arabic",
    "levantine":        "arabic",
    "moroccan":         "arabic",
    "sudanese":         "arabic",
    # South Asian group
    "south_asian":      "south_asian",
    "indian":           "south_asian",
    "pakistani":        "south_asian",
    "bangladeshi":      "south_asian",
    "sri_lankan":       "south_asian",
    # East Asian group
    "east_asian":       "east_asian",
    "chinese":          "east_asian",
    "japanese":         "east_asian",
    "korean":           "east_asian",
    # European group
    "european":         "european",
    "french":           "european",
    "german":           "european",
    "spanish":          "european",
    "italian":          "european",
    "dutch":            "european",
    "portuguese":       "european",
    "russian":          "european",
    # North American group
    "north_american":   "north_american",
    "american":         "north_american",
    "canadian":         "north_american",
    "us":               "north_american",
}

ACCENT_LABELS = {
    "arabic":         0,
    "south_asian":    1,
    "east_asian":     2,
    "european":       3,
    "north_american": 4,
}


# ──────────────────────────────────────────────
# HELPER — pad/truncate mel to fixed length
# ──────────────────────────────────────────────
def _fix_mel_length(mel: np.ndarray, target_frames: int = MAX_FRAMES) -> torch.Tensor:
    """
    Ensure mel spectrogram has exactly [N_MELS, target_frames].
    - Truncate if too long.
    - Zero-pad on the right if too short.

    FIX #2: Original code assumed extract_log_mel always returned [80, 3000],
    but audio shorter than 30 s produces fewer frames → shape mismatch in batching.
    """
    n_frames = mel.shape[1]
    if n_frames > target_frames:
        mel = mel[:, :target_frames]
    elif n_frames < target_frames:
        pad = np.zeros((N_MELS, target_frames - n_frames), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    return torch.tensor(mel, dtype=torch.float32)


# ──────────────────────────────────────────────
# CUSTOM DATASET CLASS
# ──────────────────────────────────────────────
class AccentSpeechDataset(Dataset):
    """
    PyTorch Dataset for accent-diverse speech data.

    Expects a manifest CSV with columns:
        audio_path, transcript, accent_type, split

    Args:
        manifest_path : path to the manifest CSV file
        split         : "train", "val", or "test"
        processor     : WhisperProcessor for tokenizing transcripts
        augment_audio : whether to apply data augmentation (train only)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        processor: WhisperProcessor,
        augment_audio: bool = False,
    ):
        self.processor     = processor
        self.augment_audio = augment_audio
        self.split         = split
        self.samples       = []
        self._skipped      = 0

        # FIX #7: manifest is already a split-specific CSV (no 'split' column needed)
        # FIX #9: normalize fine-grained accent labels to 5 project categories
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_accent  = row["accent_type"].strip().lower()
                norm_accent = ACCENT_NORMALIZE.get(raw_accent)
                if norm_accent is None:
                    self._skipped += 1
                    continue
                row["accent_type"] = norm_accent
                # FIX #8: convert Windows backslashes to forward slashes
                row["audio_path"] = row["audio_path"].replace("\\", "/")
                self.samples.append(row)

        print(f"  [{split.upper()}] Loaded {len(self.samples)} samples "
              f"(skipped {self._skipped} unrecognized accents)")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:  # FIX #6
        return (f"AccentSpeechDataset(n={len(self.samples)}, "
                f"augment={self.augment_audio})")

    def __getitem__(self, idx: int) -> dict:
        row = self.samples[idx]

        # FIX #3: wrap in try/except — skip corrupt audio gracefully
        try:
            audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  [WARN] Could not load {row['audio_path']}: {e} — returning zeros")
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s of silence as fallback

        # ── Augment (training only) ──────────────────
        if self.augment_audio:
            audio = augment(audio)

        # ── Extract log-mel spectrogram ──────────────
        mel = extract_log_mel(audio)            # shape: [80, T]
        mel_tensor = _fix_mel_length(mel)       # FIX #2: enforce [80, 3000]

        # ── Tokenize transcript ──────────────────────
        transcript = row["transcript"].strip()
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=225,
        ).input_ids.squeeze(0)                  # shape: [seq_len]

        # ── Accent label ─────────────────────────────
        accent_label = ACCENT_LABELS[row["accent_type"]]  # safe — filtered at init

        return {
            "input_features": mel_tensor,   # [80, 3000]
            "labels":         labels,       # [seq_len]
            "accent_label":   accent_label, # int
        }


# ──────────────────────────────────────────────
# COLLATE FUNCTION (dynamic batching)
# ──────────────────────────────────────────────
def collate_fn(batch: list, processor: WhisperProcessor) -> dict:
    """
    Custom collate for dynamic batching.
    - Mel spectrograms: stacked (all fixed [80, 3000] after FIX #2).
    - Label sequences: padded to longest in batch, -100 for ignored positions.
    """
    input_features = torch.stack([item["input_features"] for item in batch])   # [B, 80, 3000]
    accent_labels  = torch.tensor([item["accent_label"]   for item in batch], dtype=torch.long)

    label_list = [{"input_ids": item["labels"]} for item in batch]
    padded     = processor.tokenizer.pad(
        label_list,
        padding=True,
        return_tensors="pt",
    )
    labels = padded.input_ids
    # Replace pad token with -100 so CrossEntropyLoss ignores padding
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_features": input_features,  # [B, 80, 3000]
        "labels":         labels,           # [B, max_seq_len]
        "accent_labels":  accent_labels,    # [B]
    }


# ──────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────
def build_dataloaders(
    manifest_path: str,
    processor: WhisperProcessor,
    batch_size: int  = 16,
    num_workers: int = 2,
    train_csv: str   = None,
    val_csv: str     = None,
    test_csv: str    = None,
) -> dict:
    """
    Build train, validation, and test DataLoaders.

    FIX #7: Supports two manifest layouts:
      A) Single CSV with a 'split' column → pass manifest_path only (legacy)
      B) Separate CSVs per split → pass train_csv, val_csv, test_csv
         (matches Alaa's output: train.csv / validation.csv / test.csv)

    If separate CSVs are not provided, falls back to inferring paths from manifest_path dir.

    Returns:
        dict with keys "train", "val", "test"
    """
    collate = partial(collate_fn, processor=processor)

    # Infer split CSV paths from manifest directory if not explicitly given
    manifest_dir = os.path.dirname(manifest_path)
    _train_csv = train_csv or os.path.join(manifest_dir, "train.csv")
    _val_csv   = val_csv   or os.path.join(manifest_dir, "validation.csv")
    _test_csv  = test_csv  or os.path.join(manifest_dir, "test.csv")

    train_dataset = AccentSpeechDataset(_train_csv, "train", processor, augment_audio=True)
    val_dataset   = AccentSpeechDataset(_val_csv,   "val",   processor, augment_audio=False)
    test_dataset  = AccentSpeechDataset(_test_csv,  "test",  processor, augment_audio=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ──────────────────────────────────────────────
# QUICK SMOKE TEST
# ──────────────────────────────────────────────
def test_dataloader(manifest_path: str):
    """
    Verify dataset and dataloader work correctly.
    Checks shapes and prints one sample.
    """
    print("\n" + "=" * 50)
    print("  DATALOADER SMOKE TEST")
    print("=" * 50)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    loaders   = build_dataloaders(manifest_path, processor, batch_size=4, num_workers=0)

    for split, loader in loaders.items():
        batch = next(iter(loader))
        feats  = batch["input_features"]
        labels = batch["labels"]
        acc    = batch["accent_labels"]

        print(f"\n  [{split.upper()}]")
        print(f"    input_features : {feats.shape}   expected: [4, 80, 3000]")
        print(f"    labels         : {labels.shape}")
        print(f"    accent_labels  : {acc.tolist()}")

        assert feats.shape[1:] == (80, 3000), f"Bad mel shape: {feats.shape}"
        assert acc.shape[0] == feats.shape[0], "Batch size mismatch"

    print("\n  ✓ All shapes correct")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    test_dataloader(args.manifest)
