import os
import warnings
warnings.filterwarnings("ignore")

import librosa
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import soundfile as sf

# Path to your fine-tuned model — place the unzipped best_model folder here
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final-model")


class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.status.emit("Loading model...")
            self.progress.emit(10)

            import torch
            import numpy as np
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            # Use MPS on Apple Silicon, CUDA if available, else CPU
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            processor = WhisperProcessor.from_pretrained(MODEL_PATH)
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
            model.to(device)
            model.eval()

            # Force English transcription
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )

            self.status.emit("Loading audio...")
            self.progress.emit(30)

            # Load audio at 16kHz mono (required by Whisper)
            audio, _ = librosa.load(self.file_path, sr=16000, mono=True)

            self.status.emit("Transcribing...")
            self.progress.emit(50)

            # Process in 30-second chunks (Whisper's context window)
            chunk_size = 16000 * 30  # 30 seconds
            segments = []
            total_chunks = max(1, len(audio) // chunk_size + 1)

            for i, start in enumerate(range(0, len(audio), chunk_size)):
                chunk = audio[start:start + chunk_size]

                inputs = processor(
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                input_features = inputs.input_features.to(device)

                with torch.no_grad():
                    generated_ids = model.generate(input_features)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                start_sec = start / 16000
                end_sec = min((start + chunk_size) / 16000, len(audio) / 16000)

                if text.strip():
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text.strip()
                    })

                progress = 50 + int((i + 1) / total_chunks * 50)
                self.progress.emit(progress)

            self.progress.emit(100)
            self.status.emit("Done")
            self.finished.emit({"segments": segments})

        except Exception as e:
            self.error.emit(str(e))


def validate_audio_file(parent, file_path):
    if not file_path.lower().endswith((".wav", ".mp3", ".m4a")):
        QMessageBox.warning(parent, "Invalid File", "Use wav/mp3/m4a")
        return False

    try:
        duration = librosa.get_duration(path=file_path)
        if duration > 600:
            QMessageBox.warning(parent, "Too long", "Max 10 minutes")
            return False
    except:
        QMessageBox.warning(parent, "Error", "Cannot read file")
        return False

    return True


def export_txt(parent, result):
    if not result:
        return
    path, _ = QFileDialog.getSaveFileName(parent, "Save TXT", "", "*.txt")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(seg["text"].strip() + "\n")


def export_srt(parent, result):
    if not result:
        return
    path, _ = QFileDialog.getSaveFileName(parent, "Save SRT", "", "*.srt")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                f.write(f"{i}\n")
                f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                f.write(seg["text"].strip() + "\n\n")


def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_srt_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def save_recording(data, samplerate, filename):
    folder = "recordings"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    sf.write(path, data, samplerate)
    return path
