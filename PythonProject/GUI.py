# GUI.py

import sys
import os
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QProgressBar, QInputDialog
)
from PyQt5.QtCore import Qt
from coreAL import (
    TranscriptionWorker, validate_audio_file,
    export_txt, export_srt, format_time, save_recording
)

STYLE = """
QWidget {
    background-color: #121212;
    color: white;
    font-family: "Segoe UI", sans-serif;
}

#sidebar {
    background-color: #000000;
    border-right: 1px solid #282828;
}

QPushButton {
    background-color: #1db954;
    color: white;
    border-radius: 22px;
    padding: 10px 25px;
    font-weight: bold;
    font-size: 14px;
    min-height: 44px;
    border: none;
}

QPushButton:hover {
    background-color: #17a74a;
}

QPushButton:pressed {
    background-color: #12833a;
}

QTextEdit {
    background-color: #181818;
    border: 1px solid #282828;
    border-radius: 15px;
    padding: 10px;
}

QProgressBar {
    background-color: #282828;
    border-radius: 10px;
    text-align: center;
    border: none;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #1db954;
    border-radius: 10px;
}
"""

RECORDINGS_FOLDER = "recordings"


class AccentTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Accent Transcriber")
        self.setGeometry(200, 100, 1000, 600)
        self.setStyleSheet(STYLE)

        self.recording = False
        self.audio_data = []
        self.selected_device = None

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # SIDEBAR
        sidebar = QVBoxLayout()
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(200)

        title = QLabel("🎧 Transcriber")
        title.setStyleSheet("font-size:18px; font-weight:bold; padding:10px;")
        sidebar.addWidget(title)

        saved_label = QLabel("Saved Transcriptions")
        saved_label.setStyleSheet("font-size:14px; font-weight:bold; padding-left:10px;")
        sidebar.addWidget(saved_label)

        self.saved_layout = QVBoxLayout()
        sidebar.addLayout(self.saved_layout)
        sidebar.addStretch()

        main_layout.addWidget(sidebar_widget)

        # MAIN
        content = QVBoxLayout()

        header = QLabel("Accent Transcriber")
        header.setStyleSheet("font-size:24px; font-weight:bold;")
        content.addWidget(header)

        self.browse_btn = QPushButton("Select Audio")
        self.browse_btn.clicked.connect(self.browse_file)
        content.addWidget(self.browse_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        content.addWidget(self.record_btn)

        self.transcribe_btn = QPushButton("▶ Start Transcription")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        content.addWidget(self.transcribe_btn)

        self.status_label = QLabel("")
        content.addWidget(self.status_label)

        self.progress = QProgressBar()
        content.addWidget(self.progress)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        content.addWidget(self.output_text)

        export_layout = QHBoxLayout()
        txt_btn = QPushButton("Export TXT")
        srt_btn = QPushButton("Export SRT")
        txt_btn.clicked.connect(lambda: export_txt(self, getattr(self, 'result', None)))
        srt_btn.clicked.connect(lambda: export_srt(self, getattr(self, 'result', None)))
        export_layout.addWidget(txt_btn)
        export_layout.addWidget(srt_btn)
        content.addLayout(export_layout)

        main_layout.addLayout(content)

        os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
        self.load_saved_recordings()

    # MICROPHONE
    def choose_microphone(self):
        devices = sd.query_devices()
        input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

        names = [name for i, name in input_devices]

        choice, ok = QInputDialog.getItem(self, "Select Microphone", "Choose:", names, 0, False)

        if ok:
            for i, name in input_devices:
                if name == choice:
                    return i
        return None

    # RECORD
    def toggle_recording(self):
        if not self.recording:
            self.selected_device = self.choose_microphone()
            if self.selected_device is None:
                return

            self.recording = True
            self.record_btn.setText("Stop Recording")
            self.audio_data = []
            self.status_label.setText("Recording...")
            self.record_audio()
        else:
            self.recording = False
            self.record_btn.setText("Record")
            self.status_label.setText("Saving...")
            self.save_audio_file()
            self.load_saved_recordings()
            self.status_label.setText("Saved")

    def record_audio(self):
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(device=self.selected_device, channels=1, samplerate=44100, callback=callback)
        self.stream.start()

    def save_audio_file(self):
        self.stream.stop()
        self.stream.close()

        data = np.concatenate(self.audio_data, axis=0)
        filename = f"recording_{len(os.listdir(RECORDINGS_FOLDER))+1}.wav"
        self.file_path = save_recording(data, 44100, filename)
        self.browse_btn.setText(filename)

    # FILE SELECT
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "*.wav")
        if file_path:
            self.file_path = file_path
            self.browse_btn.setText(os.path.basename(file_path))

    # SIDEBAR
    def load_saved_recordings(self):
        for i in reversed(range(self.saved_layout.count())):
            widget = self.saved_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for file in os.listdir(RECORDINGS_FOLDER):
            if file.endswith(".wav"):
                btn = QPushButton(file)
                btn.clicked.connect(lambda _, f=file: self.select_saved_file(f))
                self.saved_layout.addWidget(btn)

    def select_saved_file(self, filename):
        self.file_path = os.path.join(RECORDINGS_FOLDER, filename)
        self.browse_btn.setText(filename)

    # TRANSCRIBE
    def start_transcription(self):
        if not hasattr(self, 'file_path'):
            QMessageBox.warning(self, "Error", "Select a file first")
            return

        if not validate_audio_file(self, self.file_path):
            return

        self.worker = TranscriptionWorker(self.file_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))

        self.worker.start()
        self.transcribe_btn.setEnabled(False)

    def display_results(self, result):
        self.output_text.clear()
        self.result = result

        for seg in result["segments"]:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            self.output_text.append(f"[{start} → {end}] {seg['text']}\n")

        self.transcribe_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AccentTranscriberApp()
    window.show()
    sys.exit(app.exec_())