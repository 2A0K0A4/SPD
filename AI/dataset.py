"""
dataset.py — Custom PyTorch Dataset and DataLoader
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar
"""

import os
import csv
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

from augmentation import augment
from feature_extraction import extract_log_mel


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
ACCENT_LABELS = {
    "arabic":         0,
    "south_asian":    1,
    "east_asian":     2,
    "european":       3,
    "north_american": 4,
}


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

    def __init__(self,
                 manifest_path: str,
                 split: str,
                 processor: WhisperProcessor,
                 augment_audio: bool = False):

        self.processor = processor
        self.augment_audio = augment_audio
        self.samples = []

        # Load manifest and filter by split
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append(row)

        print(f"  [{split.upper()}] Loaded {len(self.samples)} samples from manifest")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # ── Load audio ──────────────────────────────
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE)

        # ── Augment (training only) ──────────────────
        if self.augment_audio:
            audio = augment(audio)

        # ── Extract log-mel spectrogram ──────────────
        mel = extract_log_mel(audio)  # shape: [80, 3000]

        # ── Tokenize transcript ──────────────────────
        transcript = row["transcript"]
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=225
        ).input_ids.squeeze(0)  # shape: [seq_len]

        # ── Accent label ─────────────────────────────
        accent_label = ACCENT_LABELS.get(row["accent_type"], -1)

        return {
            "input_features": mel,          # [80, 3000]
            "labels": labels,               # [seq_len]
            "accent_label": accent_label,   # int
        }


# ──────────────────────────────────────────────
# COLLATE FUNCTION (dynamic batching)
# ──────────────────────────────────────────────
def collate_fn(batch: list, processor: WhisperProcessor) -> dict:
    """
    Custom collate function for dynamic batching.
    Pads label sequences to the longest in the batch.
    Mel spectrograms are already fixed size [80, 3000].
    """
    input_features = torch.stack([item["input_features"] for item in batch])
    accent_labels = torch.tensor([item["accent_label"] for item in batch], dtype=torch.long)

    # Pad label sequences to the same length
    label_list = [item["labels"] for item in batch]
    labels = processor.tokenizer.pad(
        {"input_ids": label_list},
        padding=True,
        return_tensors="pt"
    ).input_ids

    # Replace padding token id with -100 so it's ignored in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_features": input_features,   # [batch, 80, 3000]
        "labels": labels,                   # [batch, max_seq_len]
        "accent_labels": accent_labels,     # [batch]
    }


# ──────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────
def build_dataloaders(manifest_path: str,
                      processor: WhisperProcessor,
                      batch_size: int = 16,
                      num_workers: int = 2) -> dict:
    """
    Build train, validation, and test DataLoaders.

    Returns:
        dict with keys "train", "val", "test"
    """
    from functools import partial

    collate = partial(collate_fn, processor=processor)

    train_dataset = AccentSpeechDataset(manifest_path, split="train",
                                        processor=processor, augment_audio=True)
    val_dataset   = AccentSpeechDataset(manifest_path, split="val",
                                        processor=processor, augment_audio=False)
    test_dataset  = AccentSpeechDataset(manifest_path, split="test",
                                        processor=processor, augment_audio=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate,
                              num_workers=num_workers, pin_memory=True)

    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate,
                              num_workers=num_workers, pin_memory=True)

    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate,
                              num_workers=num_workers, pin_memory=True)

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
def test_dataloader(manifest_path: str):
    """
    Test the dataset and dataloader with a small subset.
    Verifies shapes and that everything loads without errors.
    """
    print(f"\n🧪 Testing DataLoader with manifest: {manifest_path}\n")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    loaders = build_dataloaders(manifest_path, processor, batch_size=4, num_workers=0)

    print("\n  Fetching one batch from train loader...")
    batch = next(iter(loaders["train"]))

    print(f"\n  ✅ Batch loaded successfully!")
    print(f"     input_features : {batch['input_features'].shape}")   # [4, 80, 3000]
    print(f"     labels         : {batch['labels'].shape}")           # [4, seq_len]
    print(f"     accent_labels  : {batch['accent_labels']}")          # [4]

    # Verify shapes
    assert batch["input_features"].shape[1:] == (80, 3000), "❌ Wrong mel shape"
    assert batch["labels"].dim() == 2,                       "❌ Wrong labels shape"

    print(f"\n  ✅ All shapes correct — DataLoader is ready!")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the AccentSpeechDataset")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    args = parser.parse_args()

    test_dataloader(args.manifest)
