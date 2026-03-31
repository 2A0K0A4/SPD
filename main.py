"""
Main Entry Point - Accent Transcriber Project
Coordinates between different modules
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src and project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).parent


def main():
    """Main entry point"""
    print("""
    ACCENT TRANSCRIBER
    
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
            from GUI import main as run_gui
            run_gui()
        except ImportError:
            print("GUI module not found. Check GUI.py")
    
    elif choice == "2":
        # Run data processing
        print("Running data processing pipeline...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "organize_datasets.py")], cwd=PROJECT_ROOT, env=env)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_statistics.py")], cwd=PROJECT_ROOT, env=env)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "split_data.py")], cwd=PROJECT_ROOT, env=env)
    
    elif choice == "3":
        # Run training
        print("Starting model training...")
        print("(Training code to be integrated from src/training/)")
    
    elif choice == "4":
        # Run evaluation
        print("Running model evaluation...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_model.py")], cwd=PROJECT_ROOT, env=env)
    
    elif choice == "5":
        # Run QA
        print("Running quality assurance...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "quality_assurance.py")], cwd=PROJECT_ROOT, env=env)
    
    elif choice == "0":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
