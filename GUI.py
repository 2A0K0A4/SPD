import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt
from coreAL import TranscriptionWorker, validate_audio_file, export_txt, export_srt, format_time


STYLE = """
QWidget {
    background-color: #121212;
    color: white;
    font-family: "Segoe UI", sans-serif;
}

/* Sidebar - Pure Black */
#sidebar {
    background-color: #000000;
    border-right: 1px solid #282828;
}

/* Sidebar buttons */
QPushButton#sidebtn {
    background: transparent;
    color: #b3b3b3;
    border: none;
    border-radius: 10px;
    padding: 10px;
    text-align: left;
}
QPushButton#sidebtn:hover {
    color: white;
    background-color: #181818;
}

/* MAIN BUTTONS (Pill Shaped) */
QPushButton {
    background-color: #1db954;
    color: white;
    border-radius: 22px; /* Half of min-height for perfect circles on ends */
    padding: 10px 25px;
    font-weight: bold;
    font-size: 14px;
    min-height: 44px;    /* Forces the button to be tall enough for the radius */
    border: none;
}

QPushButton:hover {
    background-color: #17a74a;
}

QPushButton:pressed {
    background-color: #12833a;
}

/* TEXT AREA */
QTextEdit {
    background-color: #181818;
    border: 1px solid #282828;
    border-radius: 15px;
    padding: 10px;
    selection-background-color: #1db954;
}

/* PROGRESS BAR */
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


class AccentTranscriberApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Accent Transcriber")
        self.setGeometry(200, 100, 900, 600)
        self.setStyleSheet(STYLE)

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
        sidebar.addStretch()

        main_layout.addWidget(sidebar_widget)

        # MAIN
        content = QVBoxLayout()

        header = QLabel("Accent Transcriber")
        header.setStyleSheet("font-size:24px; font-weight:bold;")
        content.addWidget(header)

        self.file_label = QLabel("No file selected")
        content.addWidget(self.file_label)

        browse_btn = QPushButton("Select Audio")
        browse_btn.clicked.connect(self.browse_file)
        content.addWidget(browse_btn)

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

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "*.wav")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))

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

