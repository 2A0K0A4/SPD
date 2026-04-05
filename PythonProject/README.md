# AI-Based Accent Transcriber

## Description
This application transcribes audio files into text while handling different accents.
It includes a GUI built with PyQt5 and supports exporting results as TXT and SRT.

## How to Run

1. Install Python (3.10+ recommended)

2. Create virtual environment:
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the app:
   python main.py

## Features
- Audio file selection (.wav)
- Transcription (currently dummy for testing)
- NLP post-processing (grammar, punctuation, accent correction)
- Export as TXT
- Export as SRT

## Notes
- The trained model is NOT included yet.
- When available, place it in:
  models/final-model/

## Project Structure
PythonProject/
- ├── main.py
- ├── gui.py
- ├── coreAL.py
- ├── nlp_postprocessor.py
- ── models/
- └── README.md

## Dependencies
- PyQt5
- librosa
- language-tool-python
- numpy

## Author
Qais Hassan & Team