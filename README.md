# SPD Project - Run All Phases

This README shows the full command sequence to set up the environment, install dependencies, run every phase of the project, and complete testing.

---

## Phase 1: Create and Activate Virtual Environment

```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv accent-env

# Activate the environment
accent-env\Scripts\activate
```

After activation, your prompt should look like:

```powershell
(accent-env) PS C:\Users\Assam\Documents\GitHub\SPD>
```

---

## Phase 2: Upgrade Pip and Install Dependencies

```powershell
# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install required packages
pip install -r requirements.txt
```

---

## Phase 3: Run the Project

### Option A: Interactive Menu (Recommended)

```powershell
python coordinator.py
```

Use the menu to run:
- 1. Run GUI Application
- 2. Run Data Processing Pipeline
- 3. Run Model Training
- 4. Run Evaluation
- 5. Run Quality Assurance
- 0. Exit

### Option B: Main Entry Point

```powershell
python main.py
```

Then choose option `1` from the menu to launch the GUI application.

---

## Phase 4: Run All Individual Scripts One-by-One

Use these commands in order to execute every processing and evaluation phase manually.

```powershell
# 1. Generate test datasets (if needed)
python .\scripts\generate_test_data.py

# 2. Download datasets (if needed)
python .\scripts\download_datasets.py

# 3. Organize datasets into master manifest
python .\scripts\organize_datasets.py

# 4. Create balanced train/validation/test splits
python .\scripts\split_data.py

# 5. Generate dataset statistics and visualizations
python .\scripts\generate_statistics.py

# 6. Run quality assurance validation
python .\scripts\quality_assurance.py

# 7. Evaluate model performance
python .\scripts\evaluate_model.py
```

Example output for dataset statistics:

```text
            🎵 DATASET STATISTICS - PHASE 3.4 🎵
============================================================
Calculating statistics...

Dataset Overview:
  Total files: 25
  Total duration: 0.0 hours
  Unique speakers: 25
  Accents: 4
Generating visualizations...
```

---

## Phase 5: Run Tests

```powershell
# Run the integration test file
python tests/test_pipeline.py

# Run all tests with coverage
pytest tests/ --cov

# Run a specific test file with verbosity
pytest tests/test_pipeline.py -v
```

---

## Phase 6: Deactivate Environment

```powershell
# Exit the virtual environment when done
deactivate
```

---

## Quick Reference

```powershell
py -3.12 -m venv accent-env
accent-env\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python coordinator.py
```

---

## Notes

- Use `coordinator.py` when you want the menu-based project runner.
- Use the `scripts/` commands when you want to run each phase manually.
- Ensure the virtual environment is activated before running any Python command.
