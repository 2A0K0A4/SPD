"""
Accent Transcriber - Data Preparation Module
"""

__version__ = "1.0.0"
__author__ = "Accent Transcriber Team"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model and scripts
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MANIFESTS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "SCRIPTS_DIR",
    "DOCS_DIR",
    "TESTS_DIR"
]
