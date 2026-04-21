import os
import warnings
warnings.filterwarnings("ignore")

import librosa
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import soundfile as sf


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

            import whisper
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device)

            self.status.emit("Transcribing...")
            self.progress.emit(40)

            result = model.transcribe(self.file_path, fp16=False)

            formatted = {
                "segments": [
                    {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in result["segments"]
                ]
            }

            self.progress.emit(100)
            self.status.emit("Done")

            self.finished.emit(formatted)

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