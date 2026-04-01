"""
baseline_test.py — Baseline Evaluation of Pre-trained Whisper Small
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar
"""

import os
import csv
import time
import whisper
import librosa
import numpy as np
from jiwer import wer, cer
from pathlib import Path
from datetime import datetime


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
ACCENT_TYPES = ["arabic", "south_asian", "east_asian", "european", "north_american"]
MODEL_SIZE = "small"         # Whisper small as per project spec


# ──────────────────────────────────────────────
# STEP 1: Load Model
# ──────────────────────────────────────────────
def load_model():
    """Load pre-trained Whisper small model."""
    print(f"⏳ Loading Whisper {MODEL_SIZE} model...")
    model = whisper.load_model(MODEL_SIZE)
    print(f"✅ Model loaded!\n")
    return model


# ──────────────────────────────────────────────
# STEP 2: Transcribe a Single File
# ──────────────────────────────────────────────
def transcribe_file(model, audio_path: str) -> dict:
    """
    Transcribe a single WAV file using Whisper.

    Returns:
        dict with:
            - "text"            : full transcription string
            - "processing_time" : seconds taken
            - "audio_duration"  : seconds of audio
            - "rtf"             : real-time factor (processing_time / audio_duration)
    """
    # Measure audio duration
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_duration = len(audio) / SAMPLE_RATE

    # Transcribe and measure time
    start = time.time()
    result = model.transcribe(audio_path, language="en", task="transcribe")
    processing_time = time.time() - start

    rtf = processing_time / audio_duration if audio_duration > 0 else 0

    return {
        "text": result["text"].strip(),
        "processing_time": round(processing_time, 3),
        "audio_duration": round(audio_duration, 3),
        "rtf": round(rtf, 4),
    }


# ──────────────────────────────────────────────
# STEP 3: Evaluate on Accent Samples
# ──────────────────────────────────────────────
def evaluate_accent(model, audio_paths: list, references: list, accent: str) -> dict:
    """
    Evaluate the model on a list of audio files for one accent type.

    Args:
        model       : loaded Whisper model
        audio_paths : list of WAV file paths
        references  : list of ground-truth transcripts (same order)
        accent      : accent type label (for display)

    Returns:
        dict with WER, CER, avg RTF, and per-file results
    """
    print(f"\n{'─'*50}")
    print(f"  Accent: {accent.upper().replace('_', ' ')}")
    print(f"  Files : {len(audio_paths)}")
    print(f"{'─'*50}")

    hypotheses = []
    rtf_list = []
    per_file_results = []

    for i, (audio_path, reference) in enumerate(zip(audio_paths, references)):
        print(f"\n  [{i+1}/{len(audio_paths)}] {os.path.basename(audio_path)}")

        result = transcribe_file(model, audio_path)
        hypothesis = result["text"]

        file_wer = wer(reference, hypothesis)
        file_cer = cer(reference, hypothesis)

        print(f"    Reference  : {reference}")
        print(f"    Hypothesis : {hypothesis}")
        print(f"    WER        : {file_wer:.2%}")
        print(f"    CER        : {file_cer:.2%}")
        print(f"    RTF        : {result['rtf']:.4f} "
              f"({'✅ PASS' if result['rtf'] < 0.5 else '❌ FAIL — too slow'})")

        hypotheses.append(hypothesis)
        rtf_list.append(result["rtf"])
        per_file_results.append({
            "file": os.path.basename(audio_path),
            "accent": accent,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(file_wer, 4),
            "cer": round(file_cer, 4),
            "rtf": result["rtf"],
            "audio_duration": result["audio_duration"],
            "processing_time": result["processing_time"],
        })

    # Overall WER/CER for this accent (across all samples)
    overall_wer = wer(references, hypotheses)
    overall_cer = cer(references, hypotheses)
    avg_rtf = round(np.mean(rtf_list), 4)

    print(f"\n  ── Summary for {accent} ──")
    print(f"  Overall WER : {overall_wer:.2%}")
    print(f"  Overall CER : {overall_cer:.2%}")
    print(f"  Avg RTF     : {avg_rtf:.4f}")

    return {
        "accent": accent,
        "wer": round(overall_wer, 4),
        "cer": round(overall_cer, 4),
        "avg_rtf": avg_rtf,
        "num_samples": len(audio_paths),
        "per_file": per_file_results,
    }


# ──────────────────────────────────────────────
# STEP 4: Run Full Baseline Evaluation
# ──────────────────────────────────────────────
def run_baseline(manifest_path: str, output_dir: str = "./baseline_results"):
    """
    Run baseline evaluation on all accent types using the manifest CSV.
    Saves results to a summary CSV and a per-file CSV.

    Manifest CSV columns: audio_path, transcript, accent_type, split
    Only uses samples where split == "test"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model()

    # Load test samples from manifest
    test_samples = {}  # accent_type -> list of (audio_path, transcript)
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                accent = row["accent_type"]
                if accent not in test_samples:
                    test_samples[accent] = []
                test_samples[accent].append((row["audio_path"], row["transcript"]))

    print(f"📊 Loaded test samples:")
    for accent, samples in test_samples.items():
        print(f"   {accent}: {len(samples)} files")

    # Evaluate each accent
    all_results = []
    all_per_file = []

    for accent in ACCENT_TYPES:
        if accent not in test_samples:
            print(f"\n[SKIP] No test samples found for: {accent}")
            continue

        paths = [s[0] for s in test_samples[accent]]
        refs  = [s[1] for s in test_samples[accent]]

        result = evaluate_accent(model, paths, refs, accent)
        all_results.append(result)
        all_per_file.extend(result["per_file"])

    # ── Print Summary Table ──────────────────────
    print(f"\n{'='*65}")
    print(f"  BASELINE RESULTS — Whisper Small (Pre-trained, No Fine-tuning)")
    print(f"{'='*65}")
    print(f"  {'Accent':<20} {'WER':>8} {'CER':>8} {'Avg RTF':>10} {'Samples':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")

    all_wer_values = []
    for r in all_results:
        wer_str = f"{r['wer']:.2%}"
        cer_str = f"{r['cer']:.2%}"
        rtf_str = f"{r['avg_rtf']:.4f}"
        target  = "✅" if r["wer"] < 0.15 else "❌"
        print(f"  {r['accent']:<20} {wer_str:>8} {cer_str:>8} {rtf_str:>10} {r['num_samples']:>8}  {target}")
        all_wer_values.append(r["wer"])

    if all_wer_values:
        overall = np.mean(all_wer_values)
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        print(f"  {'OVERALL':<20} {overall:.2%}")
        print(f"  Target: WER < 15%")

    print(f"{'='*65}\n")

    # ── Save Summary CSV ─────────────────────────
    summary_path = os.path.join(output_dir, "baseline_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accent", "wer", "cer", "avg_rtf", "num_samples"])
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k != "per_file"} for r in all_results])
    print(f"✅ Summary saved to: {summary_path}")

    # ── Save Per-file CSV ────────────────────────
    perfile_path = os.path.join(output_dir, "baseline_per_file.csv")
    if all_per_file:
        with open(perfile_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_per_file[0].keys())
            writer.writeheader()
            writer.writerows(all_per_file)
        print(f"✅ Per-file results saved to: {perfile_path}")

    # ── Save Error Analysis ──────────────────────
    error_path = os.path.join(output_dir, "baseline_error_analysis.txt")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(f"BASELINE ERROR ANALYSIS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Whisper {MODEL_SIZE} (pre-trained, no fine-tuning)\n\n")
        for r in all_results:
            f.write(f"\n{'─'*50}\n")
            f.write(f"Accent: {r['accent'].upper()}\n")
            f.write(f"WER: {r['wer']:.2%} | CER: {r['cer']:.2%}\n\n")
            f.write("Common errors (reference → hypothesis):\n")
            for pf in r["per_file"]:
                if pf["wer"] > 0:
                    f.write(f"  REF : {pf['reference']}\n")
                    f.write(f"  HYP : {pf['hypothesis']}\n")
                    f.write(f"  WER : {pf['wer']:.2%}\n\n")
    print(f"✅ Error analysis saved to: {error_path}")

    return all_results


# ──────────────────────────────────────────────
# QUICK SINGLE FILE TEST (no manifest needed)
# ──────────────────────────────────────────────
def test_single(audio_path: str, reference: str, accent: str = "unknown"):
    """Quick test on a single file without a manifest."""
    model = load_model()
    print(f"🎙️  Transcribing: {audio_path}")
    result = transcribe_file(model, audio_path)

    file_wer = wer(reference, result["text"])
    file_cer = cer(reference, result["text"])

    print(f"\n  Reference  : {reference}")
    print(f"  Hypothesis : {result['text']}")
    print(f"  WER        : {file_wer:.2%}")
    print(f"  CER        : {file_cer:.2%}")
    print(f"  RTF        : {result['rtf']:.4f}")
    print(f"  Duration   : {result['audio_duration']}s")
    print(f"  Proc. Time : {result['processing_time']}s")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Evaluation — Whisper Small")
    parser.add_argument("--mode", choices=["single", "full"], default="single",
                        help="'single' for one file test, 'full' for full evaluation")
    parser.add_argument("--input", help="Audio file path (single mode)")
    parser.add_argument("--reference", help="Ground truth transcript (single mode)")
    parser.add_argument("--accent", default="unknown", help="Accent type (single mode)")
    parser.add_argument("--manifest", help="Manifest CSV path (full mode)")
    parser.add_argument("--output", default="./baseline_results",
                        help="Output directory for results (full mode)")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.input or not args.reference:
            print("❌ Single mode requires --input and --reference")
        else:
            test_single(args.input, args.reference, args.accent)

    elif args.mode == "full":
        if not args.manifest:
            print("❌ Full mode requires --manifest")
        else:
            run_baseline(args.manifest, args.output)
