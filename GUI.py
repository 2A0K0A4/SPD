import sys
import os
import numpy as np
import sounddevice as sd
import re
import asyncio
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

from coreAL import (
    TranscriptionWorker, validate_audio_file,
    export_txt, export_srt, save_recording
)

import edge_tts
import pygame

pygame.mixer.init()

# ---------------- STYLE (UNCHANGED) ----------------
STYLE = """
QWidget {
    background-color: #0f0f10;
    color: #ffffff;
    font-family: "Segoe UI";
}

#sidebar {
    background-color: #0a0a0a;
    border-right: 1px solid #1f1f1f;
}

QPushButton {
    border-radius: 14px;
    padding: 10px;
    font-weight: bold;
}

QPushButton#primary {
    background-color: #1db954;
}
QPushButton#primary:hover {
    background-color: #17a74a;
}

QPushButton#secondary {
    background-color: #2a2a2a;
}
QPushButton#secondary:hover {
    background-color: #3a3a3a;
}

QPushButton#ai {
    background-color: #3b3bff;
}
QPushButton#ai:hover {
    background-color: #5a5aff;
}

QPushButton#delete {
    background-color: #2a2a2a;
    color: #ff4d4d;
}
QPushButton#delete:hover {
    background-color: #ff4d4d;
    color: white;
}

QTextEdit {
    background-color: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 10px;
}

QProgressBar {
    background-color: #1a1a1a;
    border-radius: 8px;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #1db954;
}
"""

VOICE = "en-US-GuyNeural"
VOICE_FOLDER = "tts_audio"
RECORDINGS_FOLDER = "recordings"

os.makedirs(VOICE_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)


# ---------------- TEXT CLEAN ----------------
def clean(text):
    return re.sub(r"\[\d{2}:\d{2}.*?\]", "", text).strip()


# ---------------- VOICE ENGINE (FIXED) ----------------
async def generate_voice(text, path):
    tts = edge_tts.Communicate(text, VOICE)
    await tts.save(path)


def speak(text):
    text = clean(text)
    if not text:
        return

    path = os.path.join(VOICE_FOLDER, "temp.mp3")

    async def run():
        await generate_voice(text, path)

    asyncio.run(run())

    # FIX: stop old sound so replay works
    pygame.mixer.music.stop()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


# ---------------- APP ----------------
class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Transcriber")
        self.resize(1100, 650)
        self.setStyleSheet(STYLE)

        self.recording = False
        self.audio_data = []

        layout = QHBoxLayout(self)

        # ---------------- LEFT ----------------
        left_box = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setObjectName("sidebar")
        left_widget.setLayout(left_box)
        left_widget.setFixedWidth(220)

        title = QLabel("🎧 Library")
        title.setStyleSheet("font-size:18px; padding:10px;")
        left_box.addWidget(title)

        self.saved_layout = QVBoxLayout()
        left_box.addLayout(self.saved_layout)
        left_box.addStretch()

        layout.addWidget(left_widget)

        # ---------------- CENTER ----------------
        center = QVBoxLayout()

        header = QLabel("AI Transcriber")
        header.setStyleSheet("font-size:24px; font-weight:bold;")
        center.addWidget(header)

        self.file_btn = QPushButton("Select Audio")
        self.file_btn.setObjectName("secondary")
        self.file_btn.clicked.connect(self.pick_file)
        center.addWidget(self.file_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.setObjectName("primary")
        self.record_btn.clicked.connect(self.toggle_record)
        center.addWidget(self.record_btn)

        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setObjectName("primary")
        self.transcribe_btn.clicked.connect(self.run_transcription)
        center.addWidget(self.transcribe_btn)

        self.status_label = QLabel("")
        center.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.hide()
        center.addWidget(self.progress)

        self.output = QTextEdit()
        center.addWidget(self.output)

        # EXPORT
        export_layout = QHBoxLayout()

        txt_btn = QPushButton("Export TXT")
        txt_btn.setObjectName("secondary")
        txt_btn.clicked.connect(lambda: export_txt(self, getattr(self, "result", None)))

        srt_btn = QPushButton("Export SRT")
        srt_btn.setObjectName("secondary")
        srt_btn.clicked.connect(lambda: export_srt(self, getattr(self, "result", None)))

        export_layout.addWidget(txt_btn)
        export_layout.addWidget(srt_btn)

        center.addLayout(export_layout)

        layout.addLayout(center)

        # ---------------- RIGHT ----------------
        right = QVBoxLayout()

        voice_title = QLabel("Voice Tools")
        voice_title.setStyleSheet("font-size:18px;")
        right.addWidget(voice_title)

        play_btn = QPushButton("Play Voice")
        play_btn.setObjectName("ai")
        play_btn.clicked.connect(lambda: speak(self.output.toPlainText()))
        right.addWidget(play_btn)

        save_btn = QPushButton("Save Voice")
        save_btn.setObjectName("ai")
        save_btn.clicked.connect(self.save_voice)
        right.addWidget(save_btn)

        right.addStretch()
        layout.addLayout(right)

        self.load_saved()

    # ---------------- FILE ----------------
    def pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select", "", "*.wav *.mp3")
        if f:
            self.file = f
            self.file_btn.setText(os.path.basename(f))

    # ---------------- RECORD ----------------
    def toggle_record(self):
        if not self.recording:
            self.recording = True
            self.record_btn.setText("Stop")
            self.audio_data = []
            self.stream = sd.InputStream(
                channels=1,
                samplerate=44100,
                callback=self.audio_callback
            )
            self.stream.start()
        else:
            self.recording = False
            self.record_btn.setText("Record")
            self.stream.stop()
            self.stream.close()

            if self.audio_data:
                data = np.concatenate(self.audio_data, axis=0)
                name = f"rec_{len(os.listdir(RECORDINGS_FOLDER))}.wav"
                self.file = save_recording(data, 44100, name)
                self.file_btn.setText(name)
                self.load_saved()

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    # ---------------- TRANSCRIBE ----------------
    def run_transcription(self):
        if not hasattr(self, "file"):
            QMessageBox.warning(self, "Error", "Select file first")
            return

        if not validate_audio_file(self, self.file):
            return

        self.progress.show()
        self.progress.setValue(0)
        self.status_label.setText("Starting...")

        self.worker = TranscriptionWorker(self.file)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.show_result)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))

        self.worker.start()

    def show_result(self, result):
        self.result = result
        self.output.clear()

        for seg in result["segments"]:
            self.output.append(seg["text"])

        self.progress.hide()
        self.status_label.setText("Done ✅")

    # ---------------- SAVED FILES ----------------
    def load_saved(self):
        for i in reversed(range(self.saved_layout.count())):
            w = self.saved_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        for file in os.listdir(RECORDINGS_FOLDER):
            if file.endswith(".wav"):
                row = QHBoxLayout()

                btn = QPushButton(file)
                btn.setObjectName("secondary")
                btn.clicked.connect(lambda _, f=file: self.select_file(f))

                delete_btn = QPushButton("X")
                delete_btn.setObjectName("delete")
                delete_btn.clicked.connect(lambda _, f=file: self.delete_file(f))

                row.addWidget(btn)
                row.addWidget(delete_btn)

                container = QWidget()
                container.setLayout(row)
                self.saved_layout.addWidget(container)

    def select_file(self, file):
        self.file = os.path.join(RECORDINGS_FOLDER, file)
        self.file_btn.setText(file)

    def delete_file(self, file):
        path = os.path.join(RECORDINGS_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)
        self.load_saved()

    # ---------------- SAVE VOICE ----------------
    def save_voice(self):
        text = clean(self.output.toPlainText())
        if not text:
            return

        name = f"voice_{datetime.now().strftime('%H%M%S')}.mp3"
        path = os.path.join(VOICE_FOLDER, name)

        async def run():
            await generate_voice(text, path)

        asyncio.run(run())


app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())