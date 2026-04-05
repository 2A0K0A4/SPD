

# coreAL.py

import os
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
import librosa
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog

# Optional NLP post-processor if you have it
try:
    from nlp_postprocessor import NLPPostProcessor
except ImportError:
    NLPPostProcessor = None

# ----------------------------
# Transcription Worker Thread
# ----------------------------
class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)   # 0-100 %
    status = pyqtSignal(str)     # status messages
    finished = pyqtSignal(dict)  # final result
    error = pyqtSignal(str)      # error messages

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            self.status.emit("Loading Whisper model...")
            self.progress.emit(10)

            import whisper
            # Force CPU to avoid GPU/DLL issues
            model = whisper.load_model("small", device="cpu")

            self.status.emit("Transcribing audio...")
            self.progress.emit(30)

            result = model.transcribe(self.file_path, fp16=False)

            formatted = {
                "segments": [
                    {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in result["segments"]
                ]
            }

            if NLPPostProcessor:
                self.status.emit("Applying NLP corrections...")
                self.progress.emit(80)
                try:
                    nlp = NLPPostProcessor()
                    formatted = nlp.process(formatted)
                except Exception as nlp_e:
                    print(f"NLP skipping: {nlp_e}")

            self.progress.emit(100)
            self.status.emit("Done!")
            self.finished.emit(formatted)

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# File Validation
# ----------------------------
def validate_audio_file(parent, file_path):
    valid_exts = (".wav", ".mp3", ".m4a")
    if not file_path.lower().endswith(valid_exts):
        QMessageBox.warning(parent, "Invalid File", f"Please select {valid_exts}")
        return False
    try:
        # Fixed warning: use path instead of filename
        duration = librosa.get_duration(path=file_path)
        if duration > 600:
            QMessageBox.warning(parent, "File Too Long", "Please select an audio file under 10 minutes.")
            return False
    except Exception:
        QMessageBox.warning(parent, "Error", "Unable to read audio file.")
        return False
    return True


# ----------------------------
# Export TXT
# ----------------------------
def export_txt(parent, result):
    if not result:
        QMessageBox.warning(parent, "Error", "Nothing to export.")
        return
    file_path, _ = QFileDialog.getSaveFileName(parent, "Save TXT", "", "Text Files (*.txt)")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(f"[{format_time(seg['start'])}] {seg['text'].strip()}\n")


# ----------------------------
# Export SRT
# ----------------------------
def export_srt(parent, result):
    if not result:
        return
    file_path, _ = QFileDialog.getSaveFileName(parent, "Save SRT", "", "SRT Files (*.srt)")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                start = format_srt_time(seg["start"])
                end = format_srt_time(seg["end"])
                f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")


# ----------------------------
# Helper Functions
# ----------------------------
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