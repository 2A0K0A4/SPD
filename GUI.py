# gui.py

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from coreAL import TranscriptionWorker, validate_audio_file, export_txt, export_srt, format_time

# -------------------
# Dark Theme Stylesheet
# -------------------
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #585b70;
}
QPushButton:pressed {
    background-color: #313244;
}
QTextEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    padding: 10px;
}
QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    text-align: center;
    color: #cdd6f4;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px;
}
"""

class AccentTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Based Accent Transcriber")
        self.setGeometry(100, 100, 600, 500)
        self.setStyleSheet(DARK_STYLESHEET)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # ----------------
        # Header
        # ----------------
        header = QLabel("AI-Based Accent Transcriber")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(header)

        # ----------------
        # File selection
        # ----------------
        self.file_label = QLabel("No file selected...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        self.browse_btn = browse_btn
        self.layout.addWidget(browse_btn)
        self.layout.addWidget(self.file_label)

        # ----------------
        # Transcribe button & status
        # ----------------
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.layout.addWidget(self.transcribe_btn)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        # ----------------
        # Progress bar
        # ----------------
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        # ----------------
        # Transcription output
        # ----------------
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # ----------------
        # Export buttons
        # ----------------
        export_txt_btn = QPushButton("Export TXT")
        export_srt_btn = QPushButton("Export SRT")
        export_txt_btn.clicked.connect(lambda: export_txt(self, getattr(self, 'result', None)))
        export_srt_btn.clicked.connect(lambda: export_srt(self, getattr(self, 'result', None)))
        self.layout.addWidget(export_txt_btn)
        self.layout.addWidget(export_srt_btn)

    # ----------------
    # Browse file
    # ----------------
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "WAV Files (*.wav)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            # Validate file duration
            if validate_audio_file(self, self.file_path):
                self.output_text.setText(f"Selected file: {os.path.basename(file_path)}")

    # ----------------
    # Start transcription
    # ----------------
    def start_transcription(self):
        if not hasattr(self, 'file_path'):
            QMessageBox.warning(self, "No File", "Please select an audio file first.")
            return

        if not validate_audio_file(self, self.file_path):
            return

        # ---------------------------
        # Placeholder: Ammar's transcriber
        # Replace None with real transcriber when available
        # ---------------------------
        self.worker = TranscriptionWorker(self.file_path, transcriber=None)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

        self.transcribe_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)

    # ----------------
    # Display results
    # ----------------
    def display_results(self, result):
        self.output_text.clear()
        self.result = result  # store for export
        for seg in result["segments"]:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            self.output_text.append(f"[{start} → {end}] {text}\n")

        self.transcribe_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = AccentTranscriberApp()
    window.show()
    sys.exit(app.exec_())


# ----------------
# Run app
# ----------------
if __name__ == "__main__":
    main()