# gui.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt

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
        self.setGeometry(100, 100, 600, 500)  # window size
        self.setStyleSheet(DARK_STYLESHEET)   # <-- Apply the dark theme

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        header = QLabel("AI-Based Accent Transcriber")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(header)

        # File section
        self.file_label = QLabel("No file selected...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        self.layout.addWidget(browse_btn)
        self.layout.addWidget(self.file_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        # Transcription output
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # Export buttons
        export_txt_btn = QPushButton("Export TXT")
        export_srt_btn = QPushButton("Export SRT")
        self.layout.addWidget(export_txt_btn)
        self.layout.addWidget(export_srt_btn)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
        if file_name:
            self.file_label.setText(file_name)
            # Placeholder for transcription
            self.output_text.setText("[00:00] Hello, my name is Ahmed and I am...\n[00:05] speaking about the importance of...\n[00:10] technology in modern education...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AccentTranscriberApp()
    window.show()
    sys.exit(app.exec_())