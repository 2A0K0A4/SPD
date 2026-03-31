"""
Quick Setup Script
One-command initialization of the project
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Error: {description}")
        return False
    
    print(f"✓ {description}")
    return True

def main():
    """Quick setup"""
    print("\n" + "QUICK SETUP - ACCENT TRANSCRIBER".center(60))
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("✗ Python 3.9+ required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Activate virtual environment
    print("\nSteps:")
    print("1. Virtual environment created at: accent-env/")
    
    # Install requirements
    if run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        print("  1. Run: python coordinator.py")
        print("  2. Follow the interactive menu")
        print("  3. Or run individual phases with scripts in scripts/ folder")
    else:
        print("\n✗ Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
