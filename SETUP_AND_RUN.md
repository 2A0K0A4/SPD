# SPD Project - Complete Setup and Run Guide

All commands to setup and run the project from start to finish.

---

## Step 1: Setup Virtual Environment

```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv accent-env
```

## Step 2: Activate Virtual Environment

```powershell
# Activate the environment (Windows PowerShell)
accent-env\Scripts\activate

# NOTE: After activation, your prompt will show:
# (accent-env) PS C:\Users\Assam\Documents\GitHub\SPD>
```

## Step 3: Upgrade pip and Build Tools

```powershell
# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
```

## Step 4: Install All Dependencies

```powershell
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

---

## Step 5: Run the Project

### Option A: Interactive Menu (RECOMMENDED)

```powershell
# Start the interactive coordinator with menu-driven interface
python coordinator.py
```

Then follow the on-screen menu to select options:
- 1. Run GUI Application
- 2. Run Data Processing Pipeline
- 3. Run Model Training
- 4. Run Evaluation
- 5. Run Quality Assurance
- 0. Exit

### Option B: Main Entry Point

```powershell
# Run main entry point
python main.py
```

Then choose option `1` from the menu to launch the GUI application.

---

## Step 6: Individual Scripts (Optional)

Run specific data processing steps:

```powershell
# Generate test datasets (if data not available)
python scripts/generate_test_data.py

# Organize datasets into master manifest
python scripts/organize_datasets.py

# Create balanced train/validation/test splits
python scripts/split_data.py

# Generate dataset statistics and visualizations
python scripts/generate_statistics.py

# Run quality assurance validation
python scripts/quality_assurance.py

# Evaluate model performance
python scripts/evaluate_model.py

# Download datasets
python scripts/download_datasets.py
```

---

## Step 7: Testing (Optional)

```powershell
# Run integration tests
python tests/test_pipeline.py

# Run tests with coverage report
pytest tests/ --cov

# Run specific test file
pytest tests/test_pipeline.py -v
```

---

## Step 8: Deactivate Environment (When Done)

```powershell
# Exit the virtual environment
deactivate
```

---

## Quick Reference - Full Setup in One Block

```powershell
# 1. Create and activate environment
py -3.12 -m venv accent-env
accent-env\Scripts\activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the project
python coordinator.py
```

---

## Troubleshooting

### If Installation Fails
```powershell
# Ensure you're using Python 3.12
python --version

# Check installed Python versions
py -0p

# If needed, create environment with specific Python version
py -3.12 -m venv accent-env
```

### If Packages Don't Install
```powershell
# Reinstall with explicit Python 3.12
Remove-Item -Recurse accent-env
py -3.12 -m venv accent-env
accent-env\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Check Virtual Environment Status
```powershell
# Verify which Python is active
python --version

# Check installed packages
pip list

# Verify core dependencies
python -c "import librosa, pandas, torch, transformers; print('✓ All core dependencies installed')"
```

---

## Project Structure

```
SPD/
├── data/                    # Datasets and processed data
├── scripts/                 # Individual processing scripts
├── src/                     # Source code modules
├── tests/                   # Test files
├── docs/                    # Documentation
├── coordinator.py           # Main interactive CLI (RECOMMENDED)
├── main.py                  # Alternative entry point
├── requirements.txt         # Python dependencies
└── setup.py                 # Package setup
```

---

## Next Steps

1. **First time**: Run `python coordinator.py` and follow the menu
2. **Data processing**: Use scripts in `scripts/` folder for individual steps
3. **Training**: Check `src/training/` for model training code
4. **Testing**: Run `pytest tests/` to verify setup

---

**Created**: March 27, 2026
**Python Version**: 3.12
**Environment**: accent-env
