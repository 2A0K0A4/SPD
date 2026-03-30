"""
train.py — Main Training Script
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar

Phase 3: Training Pipeline Development
- Trains Whisper (primary) on accent-diverse speech data
- Logs WER per epoch and per accent type
- Saves best checkpoint based on validation WER
- Compatible with AWS SageMaker (reads env vars for paths)

Usage (local):
    python train.py --manifest data/manifests/master_manifest.csv \
                    --output checkpoints/ \
                    --model openai/whisper-small \
                    --epochs 10 \
                    --batch_size 8 \
                    --debug

Usage (SageMaker):
    Paths injected automatically via SM_CHANNEL_* env variables.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)

from dataset import build_dataloaders
from evaluate import compute_wer_per_accent


# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper on accent speech data")

    # Paths — fall back to SageMaker env vars if not provided
    parser.add_argument(
        "--manifest",
        default=os.environ.get("SM_CHANNEL_MANIFEST", "data/manifests/master_manifest.csv"),
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("SM_MODEL_DIR", "checkpoints/"),
        help="Directory to save model checkpoints",
    )

    # Model
    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="HuggingFace model ID or local path",
    )

    # Training hyperparameters
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--warmup_steps",type=int,   default=100)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")

    # Debug mode: uses only first 50 samples per split
    parser.add_argument("--debug", action="store_true",
                        help="Run a quick smoke test on a tiny subset")

    return parser.parse_args()


# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS (M-series chip)")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — training on CPU (slow)")
    return device


# ──────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, grad_accum, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_features = batch["input_features"].to(device)  # [B, 80, 3000]
        labels         = batch["labels"].to(device)          # [B, seq_len]

        outputs = model(
            input_features=input_features,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

        if step % 20 == 0:
            log.info(f"  Epoch {epoch} | Step {step}/{len(loader)} | Loss: {outputs.loss.item():.4f}")

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# VALIDATE ONE EPOCH
# ──────────────────────────────────────────────
@torch.no_grad()
def validate_epoch(model, loader, processor, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references  = []
    all_accents     = []

    for batch in loader:
        input_features = batch["input_features"].to(device)
        labels         = batch["labels"].to(device)
        accent_labels  = batch["accent_labels"]

        # Forward pass (loss)
        outputs = model(input_features=input_features, labels=labels)
        total_loss += outputs.loss.item()

        # Generate transcriptions
        generated_ids = model.generate(input_features)
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode reference labels (replace -100 with pad token)
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
        references = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

        all_predictions.extend(predictions)
        all_references.extend(references)
        all_accents.extend(accent_labels.tolist())

    avg_loss = total_loss / len(loader)
    wer_results = compute_wer_per_accent(all_predictions, all_references, all_accents)

    return avg_loss, wer_results


# ──────────────────────────────────────────────
# SAVE CHECKPOINT
# ──────────────────────────────────────────────
def save_checkpoint(model, processor, output_dir, epoch, val_wer, is_best=False):
    epoch_dir = Path(output_dir) / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    meta = {"epoch": epoch, "val_wer_overall": val_wer, "timestamp": datetime.now().isoformat()}
    with open(epoch_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if is_best:
        best_dir = Path(output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        with open(best_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"  ★ New best model saved → {best_dir}  (WER: {val_wer:.4f})")

    log.info(f"  Checkpoint saved → {epoch_dir}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()

    log.info("=" * 60)
    log.info("AI-Based Accent Transcribing System | Phase 3 Training")
    log.info(f"Model      : {args.model}")
    log.info(f"Manifest   : {args.manifest}")
    log.info(f"Output     : {args.output}")
    log.info(f"Epochs     : {args.epochs}")
    log.info(f"Batch size : {args.batch_size}  (grad_accum x{args.grad_accum})")
    log.info(f"Device     : {device}")
    log.info(f"Debug mode : {args.debug}")
    log.info("=" * 60)

    # ── Load processor and model ───────────────
    log.info("Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model     = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)

    # Force English transcription (no translation)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    # ── Build dataloaders ──────────────────────
    log.info("Building dataloaders...")
    loaders = build_dataloaders(
        manifest_path=args.manifest,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # Debug mode: limit steps
    if args.debug:
        log.warning("DEBUG MODE: truncating loaders to first 5 batches each")
        # Wrap loaders with islice for debug
        from itertools import islice
        class DebugLoader:
            def __init__(self, loader, n=5):
                self.loader = loader
                self.n = n
            def __iter__(self):
                return islice(self.loader, self.n)
            def __len__(self):
                return self.n
        train_loader = DebugLoader(train_loader)
        val_loader   = DebugLoader(val_loader)
        args.epochs  = 2

    # ── Optimizer and scheduler ────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps   = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps  = min(args.warmup_steps, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ──────────────────────────
    best_wer  = float("inf")
    train_log = []

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        log.info(f"\n{'─'*50}")
        log.info(f"EPOCH {epoch}/{args.epochs}")
        log.info(f"{'─'*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, epoch
        )
        log.info(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_loss, wer_results = validate_epoch(model, val_loader, processor, device)
        overall_wer = wer_results.get("overall", float("inf"))

        log.info(f"  Val loss  : {val_loss:.4f}")
        log.info(f"  WER (overall): {overall_wer:.4f}")
        for accent, wer in wer_results.items():
            if accent != "overall":
                log.info(f"    WER [{accent:>14}]: {wer:.4f}")

        # Save checkpoint
        is_best = overall_wer < best_wer
        if is_best:
            best_wer = overall_wer

        save_checkpoint(model, processor, args.output, epoch, overall_wer, is_best=is_best)

        # Log to file
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wer": wer_results,
        }
        train_log.append(epoch_record)
        with open(Path(args.output) / "training_log.json", "w") as f:
            json.dump(train_log, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"Training complete. Best validation WER: {best_wer:.4f}")
    log.info(f"Best model saved at: {Path(args.output) / 'best'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
