"""
evaluate.py — WER Computation Per Accent Type
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar

Provides compute_wer_per_accent() used by the training loop,
and a standalone CLI for evaluating a saved checkpoint.

Usage (standalone):
    python evaluate.py \
        --model checkpoints/best \
        --manifest data/manifests/master_manifest.csv \
        --split test
"""

import re
import json
import logging
import argparse
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Map int label → accent name (must match dataset.py ACCENT_LABELS)
LABEL_TO_ACCENT = {
    0: "arabic",
    1: "south_asian",
    2: "east_asian",
    3: "european",
    4: "north_american",
}


# ──────────────────────────────────────────────
# TEXT NORMALISATION
# ──────────────────────────────────────────────
def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# WORD ERROR RATE (pure Python — no jiwer needed)
# ──────────────────────────────────────────────
def _wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between two strings using dynamic programming.
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count.
    """
    ref_words = normalise(reference).split()
    hyp_words = normalise(hypothesis).split()

    N = len(ref_words)
    if N == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    M = len(hyp_words)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (M + 1) for _ in range(N + 1)]

    for i in range(N + 1):
        dp[i][0] = i
    for j in range(M + 1):
        dp[0][j] = j

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[N][M] / N


# ──────────────────────────────────────────────
# BATCH WER PER ACCENT
# ──────────────────────────────────────────────
def compute_wer_per_accent(
    predictions: list[str],
    references:  list[str],
    accent_labels: list[int],
) -> dict:
    """
    Compute overall WER and WER broken down by accent type.

    Args:
        predictions   : list of decoded hypothesis strings
        references    : list of ground-truth transcript strings
        accent_labels : list of int accent labels (matching LABEL_TO_ACCENT)

    Returns:
        dict e.g. {
            "overall": 0.21,
            "arabic": 0.28,
            "south_asian": 0.19,
            ...
        }
    """
    assert len(predictions) == len(references) == len(accent_labels), \
        "predictions, references, and accent_labels must have the same length"

    # Group by accent
    buckets = defaultdict(lambda: {"refs": [], "hyps": []})
    all_refs, all_hyps = [], []

    for pred, ref, label in zip(predictions, references, accent_labels):
        accent = LABEL_TO_ACCENT.get(label, f"unknown_{label}")
        buckets[accent]["refs"].append(ref)
        buckets[accent]["hyps"].append(pred)
        all_refs.append(ref)
        all_hyps.append(pred)

    results = {}

    # Per-accent WER
    for accent, data in buckets.items():
        wer_scores = [_wer(r, h) for r, h in zip(data["refs"], data["hyps"])]
        results[accent] = round(sum(wer_scores) / len(wer_scores), 4)

    # Overall WER
    overall_scores = [_wer(r, h) for r, h in zip(all_refs, all_hyps)]
    results["overall"] = round(sum(overall_scores) / len(overall_scores), 4)

    return results


# ──────────────────────────────────────────────
# STANDALONE EVALUATION CLI
# ──────────────────────────────────────────────
def evaluate_checkpoint(model_path: str, manifest_path: str, split: str = "test"):
    """
    Load a saved checkpoint and evaluate it on a manifest split.
    Prints and saves a full WER report.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from dataset import build_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading model from {model_path}")

    processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    loaders = build_dataloaders(manifest_path, processor, batch_size=8, num_workers=2)
    loader  = loaders[split]

    all_predictions, all_references, all_accents = [], [], []

    log.info(f"Running inference on {split} split...")
    with torch.no_grad():
        for batch in loader:
            input_features = batch["input_features"].to(device)
            labels         = batch["labels"]
            accent_labels  = batch["accent_labels"]

            generated_ids = model.generate(input_features)
            predictions   = processor.batch_decode(generated_ids, skip_special_tokens=True)

            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(labels_for_decode, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)
            all_accents.extend(accent_labels.tolist())

    wer_results = compute_wer_per_accent(all_predictions, all_references, all_accents)

    # Print report
    print("\n" + "=" * 50)
    print(f"  WER REPORT — {split.upper()} SPLIT")
    print("=" * 50)
    print(f"  Overall WER       : {wer_results['overall']:.4f} ({wer_results['overall']*100:.1f}%)")
    print("-" * 50)
    for accent, wer in wer_results.items():
        if accent != "overall":
            status = "✓" if wer < 0.30 else "✗"
            print(f"  {status} {accent:<18}: {wer:.4f} ({wer*100:.1f}%)")
    print("=" * 50)

    # Save report
    report_path = Path(model_path) / f"wer_report_{split}.json"
    with open(report_path, "w") as f:
        json.dump(wer_results, f, indent=2)
    print(f"\n  Report saved → {report_path}")

    return wer_results


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate a Whisper checkpoint")
    parser.add_argument("--model",    required=True, help="Path to saved checkpoint dir")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--split",    default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate_checkpoint(args.model, args.manifest, args.split)
