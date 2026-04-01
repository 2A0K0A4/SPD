"""
Main Entry Point - Accent Transcriber Project
Coordinates between different modules
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point"""
    print("""
    🎵 ACCENT TRANSCRIBER 🎵
    
    Choose an option:
    1. Run GUI Application
    2. Run Data Processing Pipeline
    3. Run Model Training
    4. Run Evaluation
    5. Run Quality Assurance
    0. Exit
    """)
    
    choice = input("Enter your choice (0-5): ").strip()
    
    if choice == "1":
        # Run GUI
        try:
            from app.GUI import main as run_gui
            run_gui()
        except ImportError:
            print("✗ GUI module not found. Check src/app/GUI.py")
    
    elif choice == "2":
        # Run data processing
        print("Running data processing pipeline...")
        os.system("python scripts/organize_datasets.py")
        os.system("python scripts/generate_statistics.py")
        os.system("python scripts/split_data.py")
    
    elif choice == "3":
        # Run training
        print("Starting model training...")
        print("(Training code to be integrated from src/training/)")
    
    elif choice == "4":
        # Run evaluation
        print("Running model evaluation...")
        os.system("python scripts/evaluate_model.py")
    
    elif choice == "5":
        # Run QA
        print("Running quality assurance...")
        os.system("python scripts/quality_assurance.py")
    
    elif choice == "0":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
