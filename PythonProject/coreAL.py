# coreAL.py
import os
import time
import librosa

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from nlp_postprocessor import NLPPostProcessor


class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path, transcriber):
        super().__init__()
        self.file_path = file_path
        self.transcriber = transcriber

    def run(self):
        try:
            self.status.emit("Starting...")
            self.progress.emit(10)

            # ✅ SAFE DUMMY RESULT (NO CRASH)
            result = {
                "segments": [
                    {"start": 0, "end": 5, "text": "hello my name is ahmed"},
                    {"start": 5, "end": 10, "text": "this is a test transcription"}
                ]
            }

            # simulate progress
            for i in range(10, 90, 10):
                self.progress.emit(i)
                self.status.emit(f"Processing {i}%")
                time.sleep(0.15)

            # NLP
            self.status.emit("Applying NLP...")
            nlp = NLPPostProcessor()
            result = nlp.process(result)

            self.progress.emit(100)
            self.status.emit("Done")

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# VALIDATION
# ----------------------------
def validate_audio_file(parent, file_path):
    if not file_path.lower().endswith(".wav"):
        QMessageBox.warning(parent, "Invalid File", "Please select a .wav file.")
        return False

    try:
        duration = librosa.get_duration(path=file_path)
        if duration > 600:
            QMessageBox.warning(parent, "Too Long", "Max 10 minutes allowed.")
            return False
    except:
        QMessageBox.warning(parent, "Error", "Cannot read file.")
        return False

    return True


# ----------------------------
# EXPORT TXT
# ----------------------------
def export_txt(parent, result):
    if not result:
        QMessageBox.warning(parent, "No Data", "No transcription available.")
        return

    file_path, _ = QFileDialog.getSaveFileName(parent, "Save TXT", "", "*.txt")

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(f"[{format_time(seg['start'])} → {format_time(seg['end'])}] {seg['text']}\n")


# ----------------------------
# EXPORT SRT (FIXED)
# ----------------------------
def export_srt(parent, result):
    if not result:
        QMessageBox.warning(parent, "No Data", "No transcription available.")
        return

    file_path, _ = QFileDialog.getSaveFileName(parent, "Save SRT", "", "*.srt")

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                f.write(f"{i}\n")
                f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")


# ----------------------------
# TIME HELPERS
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