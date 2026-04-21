"""
compare_models.py
Pre-trained model comparison: Whisper (tiny/base/small/medium), wav2vec 2.0, DeepSpeech
Evaluates WER, CER, RTF, model size, and VRAM usage across all accent groups.
Saves results to comparison_results.csv
"""

import os
import time
import csv
import torch
import librosa
import numpy as np
import pandas as pd
from jiwer import wer, cer

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEST_CSV      = "data/manifests/test.csv"   # columns: audio_path, transcript, accent_type
OUTPUT_CSV    = "comparison_results.csv"
SAMPLE_RATE   = 16000
ACCENTS       = ["arabic", "south_asian", "east_asian", "european", "north_american"]

ACCENT_NORMALIZE = {
    # arabic
    "arabic": "arabic", "egyptian": "arabic", "levantine": "arabic",
    "gulf": "arabic", "saudi": "arabic", "moroccan": "arabic",
    # south asian
    "south_asian": "south_asian", "indian": "south_asian",
    "pakistani": "south_asian", "bangladeshi": "south_asian",
    # east asian
    "east_asian": "east_asian", "chinese": "east_asian",
    "mandarin": "east_asian", "japanese": "east_asian", "korean": "east_asian",
    # european
    "european": "european", "french": "european", "german": "european",
    "spanish": "european", "italian": "european", "british": "european",
    # north american
    "north_american": "north_american", "american": "north_american",
    "canadian": "north_american",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)

def get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total / 1e6, 1)

def peak_vram_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return 0.0

def normalize_accent(label):
    return ACCENT_NORMALIZE.get(label.lower().strip(), None)

def load_test_data():
    df = pd.read_csv(TEST_CSV)
    df["accent_norm"] = df["accent_type"].apply(normalize_accent)
    df = df[df["accent_norm"].notna()].reset_index(drop=True)
    # fix Windows backslashes
    df["audio_path"] = df["audio_path"].str.replace("\\", "/", regex=False)
    return df

# ── WHISPER ───────────────────────────────────────────────────────────────────

def evaluate_whisper(model_size, df):
    import whisper
    print(f"\n[Whisper {model_size}] Loading...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = whisper.load_model(model_size)
    size_mb = get_model_size_mb(model)

    rows = []
    for accent in ACCENTS:
        subset = df[df["accent_norm"] == accent]
        if subset.empty:
            continue

        refs, hyps, rtfs = [], [], []
        for _, row in subset.iterrows():
            if not os.path.isfile(row["audio_path"]):
                continue
            audio = load_audio(row["audio_path"])
            duration = len(audio) / SAMPLE_RATE

            t0 = time.time()
            result = model.transcribe(row["audio_path"], fp16=False)
            elapsed = time.time() - t0

            refs.append(row["transcript"].lower().strip())
            hyps.append(result["text"].lower().strip())
            rtfs.append(elapsed / duration if duration > 0 else 0)

        if not refs:
            continue

        rows.append({
            "model": f"whisper-{model_size}",
            "accent": accent,
            "wer": round(wer(refs, hyps), 4),
            "cer": round(cer(refs, hyps), 4),
            "avg_rtf": round(np.mean(rtfs), 4),
            "model_size_mb": size_mb,
            "peak_vram_mb": peak_vram_mb(),
            "num_samples": len(refs),
        })

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows

# ── WAV2VEC 2.0 ───────────────────────────────────────────────────────────────

WAV2VEC_MODELS = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-lv60",
]

def evaluate_wav2vec(model_id, df):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"\n[wav2vec 2.0] Loading {model_id}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()
    size_mb = get_model_size_mb(model)

    short_name = model_id.split("/")[-1]
    rows = []
    for accent in ACCENTS:
        subset = df[df["accent_norm"] == accent]
        if subset.empty:
            continue

        refs, hyps, rtfs = [], [], []
        for _, row in subset.iterrows():
            if not os.path.isfile(row["audio_path"]):
                continue
            audio = load_audio(row["audio_path"])
            duration = len(audio) / SAMPLE_RATE

            inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            t0 = time.time()
            with torch.no_grad():
                logits = model(input_values).logits
            elapsed = time.time() - t0

            pred_ids = torch.argmax(logits, dim=-1)
            hypothesis = processor.batch_decode(pred_ids)[0].lower().strip()

            refs.append(row["transcript"].lower().strip())
            hyps.append(hypothesis)
            rtfs.append(elapsed / duration if duration > 0 else 0)

        if not refs:
            continue

        rows.append({
            "model": short_name,
            "accent": accent,
            "wer": round(wer(refs, hyps), 4),
            "cer": round(cer(refs, hyps), 4),
            "avg_rtf": round(np.mean(rtfs), 4),
            "model_size_mb": size_mb,
            "peak_vram_mb": peak_vram_mb(),
            "num_samples": len(refs),
        })

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading test data...")
    df = load_test_data()
    print(f"  {len(df)} samples across {df['accent_norm'].nunique()} accents")

    all_rows = []

    # Whisper variants
    for size in ["tiny", "base", "small", "medium"]:
        try:
            all_rows.extend(evaluate_whisper(size, df))
        except Exception as e:
            print(f"[Whisper {size}] ERROR: {e}")

    # wav2vec 2.0 variants
    for model_id in WAV2VEC_MODELS:
        try:
            all_rows.extend(evaluate_wav2vec(model_id, df))
        except Exception as e:
            print(f"[wav2vec {model_id}] ERROR: {e}")



    # Save CSV
    if not all_rows:
        print("\nNo results to save.")
        return

    fieldnames = ["model", "accent", "wer", "cer", "avg_rtf",
                  "model_size_mb", "peak_vram_mb", "num_samples"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ Results saved to {OUTPUT_CSV}")

    # Print summary table
    results_df = pd.DataFrame(all_rows)
    summary = (
        results_df.groupby("model")[["wer", "cer", "avg_rtf", "model_size_mb"]]
        .mean()
        .round(4)
        .rename(columns={"wer": "avg_WER", "cer": "avg_CER",
                         "avg_rtf": "avg_RTF", "model_size_mb": "size_MB"})
    )
    print("\n── Overall Summary ──")
    print(summary.to_string())


if __name__ == "__main__":
    main()
