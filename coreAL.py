# coreAL.py

import os
import time
import librosa

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from nlp_postprocessor import NLPPostProcessor


# ----------------------------
# Transcription Worker Thread
# ----------------------------
class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)   # 0-100 %
    status = pyqtSignal(str)     # status messages
    finished = pyqtSignal(dict)  # final result
    error = pyqtSignal(str)      # error messages

    def __init__(self, file_path, transcriber):
        super().__init__()
        self.file_path = file_path
        self.transcriber = transcriber  # <-- Ammar's module will go here

    def run(self):
        try:
            self.status.emit("Loading audio file...")
            self.progress.emit(10)

            # -----------------------------------
            # Ammar's transcription (NOT READY)
            # -----------------------------------
            # result = self.transcriber.transcribe(
            #     self.file_path,
            #     progress_callback=self.update_progress
            # )

            # TEMP DUMMY RESULT (for testing)
            result = {
                "segments": [
                    {"start": 0, "end": 5, "text": "hello my name is ahmed"},
                    {"start": 5, "end": 10, "text": "ze importance of technology is big"}
                ]
            }

            # Simulate progress
            for i in range(10, 80, 10):
                self.progress.emit(i)
                self.status.emit(f"Processing... {i}%")
                time.sleep(0.2)

            # -----------------------------------
            # NLP POST-PROCESSING (Phase 4)
            # -----------------------------------
            self.status.emit("Applying NLP corrections...")
            self.progress.emit(85)

            nlp = NLPPostProcessor()
            result = nlp.process(result)

            # -----------------------------------
            self.progress.emit(100)
            self.status.emit("Done!")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def update_progress(self, value):
        # Map Ammar's progress (0-100) → (10-85)
        mapped = 10 + int(value * 0.75)
        self.progress.emit(mapped)


# ----------------------------
# File Validation
# ----------------------------
def validate_audio_file(parent, file_path):
    if not file_path.lower().endswith(".wav"):
        QMessageBox.warning(parent, "Invalid File", "Please select a .wav file.")
        return False

    try:
        duration = librosa.get_duration(filename=file_path)
        if duration > 600:
            QMessageBox.warning(parent, "File Too Long", "Please select an audio file under 10 minutes.")
            return False
    except Exception:
        QMessageBox.warning(parent, "Corrupted File", "Unable to read audio file.")
        return False

    return True


# ----------------------------
# Export TXT
# ----------------------------
def export_txt(parent, result):
    if not result:
        QMessageBox.warning(parent, "No Transcription", "No transcription to export.")
        return

    file_path, _ = QFileDialog.getSaveFileName(parent, "Save as TXT", "", "Text Files (*.txt)")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                start = format_time(seg["start"])
                end = format_time(seg["end"])
                f.write(f"[{start} → {end}] {seg['text'].strip()}\n")


# ----------------------------
# Export SRT
# ----------------------------
def export_srt(parent, result):
    if not result:
        QMessageBox.warning(parent, "No Transcription", "No transcription to export.")
        return

    file_path, _ = QFileDialog.getSaveFileName(parent, "Save as SRT", "", "SubRip Files (*.srt)")
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                start = format_srt_time(seg["start"])
                end = format_srt_time(seg["end"])
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{seg['text'].strip()}\n\n")


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