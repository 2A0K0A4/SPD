"""
Master Initialization and Testing Script
Coordinates all phases of the project
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectCoordinator:
    """Coordinates project phases"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "scripts"
        self.data_dir = self.project_root / "data"
        self.docs_dir = self.project_root / "docs"
        
        self.phases = {
            1: "Environment Setup ✓",
            2: "Dataset Download",
            3: "Data Organization & Splitting",
            4: "Quality Assurance",
            5: "Model Testing",
            6: "Documentation",
            7: "Git Management"
        }
    
    def print_header(self, text: str) -> None:
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}".center(70))
        print("="*70)
    
    def print_menu(self) -> None:
        """Print main menu"""
        self.print_header("🎵 ACCENT TRANSCRIBER - PROJECT COORDINATOR 🎵")
        print("\nAVAILABLE PHASES:\n")
        
        for phase_num, phase_name in self.phases.items():
            print(f"  {phase_num}. {phase_name}")
        
        print("\nOPTIONS:")
        print("  0. Run All Phases")
        print("  S. Setup/Verify Environment")
        print("  Q. Quit")
    
    def verify_environment(self) -> bool:
        """Verify project environment is set up correctly"""
        print("\n" + "="*70)
        print("VERIFYING ENVIRONMENT")
        print("="*70)
        
        checks = []
        
        # Check Python version
        print(f"✓ Python: {sys.version.split()[0]}")
        checks.append(True)
        
        # Check required directories
        required_dirs = [
            'data/raw/common_voice',
            'data/raw/librispeech',
            'data/raw/speech_accent_archive',
            'data/raw/voxpopuli',
            'data/processed/train',
            'data/processed/validation',
            'data/processed/test',
            'data/manifests',
            'src/app',
            'src/training',
            'src/nlp',
            'tests/samples',
            'tests/results',
            'docs/images',
            'scripts'
        ]
        
        print("\nDirectory Structure:")
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {dir_path}")
            checks.append(exists)
        
        # Check required files
        print("\nRequired Files:")
        required_files = [
            'requirements.txt',
            'README.md',
            '.gitignore',
            'data/accent_mapping.json'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")
            checks.append(exists)
        
        # Check Python packages
        print("\nPython Packages:")
        try:
            import librosa
            print("  ✓ librosa")
            checks.append(True)
        except:
            print("  ✗ librosa")
            checks.append(False)
        
        try:
            import pandas
            print("  ✓ pandas")
            checks.append(True)
        except:
            print("  ✗ pandas")
            checks.append(False)
        
        try:
            import torch
            print("  ✓ torch")
            checks.append(True)
        except:
            print("  ✗ torch")
            checks.append(False)
        
        try:
            from jiwer import wer
            print("  ✓ jiwer")
            checks.append(True)
        except:
            print("  ✗ jiwer")
            checks.append(False)
        
        # Summary
        success_count = sum(checks)
        total = len(checks)
        
        print(f"\nEnvironment Check: {success_count}/{total} passed")
        
        if success_count < 10:
            print("\n⚠ Some issues detected. Run: pip install -r requirements.txt")
            return False
        
        print("\n✓ Environment is ready!")
        return True
    
    def run_phase(self, phase_num: int) -> bool:
        """Run specific phase"""
        phase_scripts = {
            1: None,  # Already done
            2: "download_datasets.py",
            3: ["organize_datasets.py", "generate_statistics.py", "split_data.py"],
            4: "quality_assurance.py",
            5: "evaluate_model.py",
            6: None,  # Documentation already created
            7: None   # Git management
        }
        
        if phase_num == 1:
            print("✓ Phase 1 already completed")
            return True
        
        elif phase_num == 6:
            self.print_header(f"PHASE {phase_num}: {self.phases[phase_num]}")
            print("""
Documentation has been generated:
  - README.md
  - docs/IMPLEMENTATION_GUIDE.md
  - docs/dataset_statistics.md (after Phase 3)
  - docs/data_quality_report.md (after Phase 4)
  - docs/user_testing_report.md (template created)

Review and update these files with your project details.
            """)
            return True
        
        elif phase_num == 7:
            self.print_header(f"PHASE {phase_num}: {self.phases[phase_num]}")
            print("""
Git Repository Management:

Current Status:
  """)
            result = subprocess.run(["git", "status"], cwd=self.project_root)
            print("""
Recommended Git Workflow:
  1. git add scripts/ docs/ requirements.txt
  2. git commit -m "feat: add data preparation scripts and docs"
  3. git push origin feature/data-and-docs

See docs/IMPLEMENTATION_GUIDE.md for more details.
            """)
            return True
        
        scripts = phase_scripts.get(phase_num)
        
        if scripts is None:
            print(f"Phase {phase_num}: {self.phases[phase_num]} - Completed (Manual)")
            return True
        
        self.print_header(f"PHASE {phase_num}: {self.phases[phase_num]}")
        
        if isinstance(scripts, str):
            scripts = [scripts]
        
        for script in scripts:
            script_path = self.scripts_dir / script
            
            if not script_path.exists():
                print(f"✗ Script not found: {script_path}")
                return False
            
            print(f"\nRunning: {script}")
            print("-" * 70)
            
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                print(f"✗ Script failed: {script}")
                return False
            
            print("-" * 70)
        
        return True
    
    def run_all_phases(self) -> None:
        """Run all phases sequentially"""
        self.print_header("RUNNING ALL PHASES")
        
        completed = []
        failed = []
        
        for phase_num in range(1, 8):
            try:
                if self.run_phase(phase_num):
                    completed.append(phase_num)
                else:
                    failed.append(phase_num)
            except Exception as e:
                logger.error(f"Phase {phase_num} failed: {e}")
                failed.append(phase_num)
        
        # Summary
        self.print_header("EXECUTION SUMMARY")
        print(f"Completed: {len(completed)}/7 phases")
        print(f"  Phases: {', '.join(map(str, completed))}")
        
        if failed:
            print(f"Failed: {len(failed)}/7 phases")
            print(f"  Phases: {', '.join(map(str, failed))}")
    
    def run(self) -> None:
        """Main interactive loop"""
        while True:
            self.print_menu()
            
            choice = input("\nEnter your choice (0-7, S, Q): ").strip().upper()
            
            if choice == 'Q':
                print("\n✓ Goodbye!")
                break
            
            elif choice == 'S':
                self.verify_environment()
            
            elif choice == '0':
                confirm = input("Run all phases? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.run_all_phases()
            
            elif choice.isdigit() and 1 <= int(choice) <= 7:
                phase_num = int(choice)
                self.run_phase(phase_num)
            
            else:
                print("✗ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    coordinator = ProjectCoordinator()
    coordinator.run()
