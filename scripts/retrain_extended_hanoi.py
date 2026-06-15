"""Regenerate Extended Hanoi datasets + retrain models end-to-end.

This script:
1. Backs up current (Balerma) DATASETS
2. Generates Extended Hanoi datasets using graph_dataset/dataset.py
3. Cleans features (removes leak-related features)
4. Retrains detection and localization models
5. Copies all artifacts to models/extended_hanoi/
6. Restores Balerma DATASETS
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "DATASETS"
BACKUP = ROOT / "DATASETS_balerma_backup"
MODELS = ROOT / "models"
EH_MODELS = MODELS / "extended_hanoi"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

def run(cmd, **kwargs):
    print(f"\n{'='*60}\n>>> {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT), **kwargs)
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(1)

# Step 1: Backup current Balerma datasets
print("\n[1/7] Backing up Balerma datasets...")
if BACKUP.exists():
    shutil.rmtree(BACKUP)
# Only backup the key files, not subdirectories
key_files = [
    "X_classification.npy", "y_classification.npy",
    "X_classification_no_leak.npy",
    "X_localization.npy", "y_localization.npy",
    "X_localization_no_leak.npy",
    "feature_names.json", "feature_names_no_leak.json",
    "localization_feature_names.json", "localization_feature_names_no_leak.json",
    "zone_definitions.json", "baseline_pressure_reference.json",
    "classification_B.pt",
]
BACKUP.mkdir(exist_ok=True)
for f in key_files:
    src = DATASETS / f
    if src.exists():
        shutil.copy2(src, BACKUP / f)
print(f"  Backed up {len(list(BACKUP.iterdir()))} files to {BACKUP}")

# Step 2: Generate Extended Hanoi datasets
print("\n[2/7] Generating Extended Hanoi datasets (this takes a few minutes)...")
run(f'"{PYTHON}" graph_dataset/dataset.py')

# Step 3: Clean features (remove leak-related features)
print("\n[3/7] Cleaning detection features...")
run(f'"{PYTHON}" scripts/clean_leak_features.py')

print("\n[4/7] Cleaning localization features...")
run(f'"{PYTHON}" scripts/clean_localization_leak_features.py')

# Step 4: Retrain detection model
print("\n[5/7] Retraining detection model...")
run(f'"{PYTHON}" models/train_leak_detection.py --use-cleaned')

# Step 5: Retrain localization model
print("\n[6/7] Retraining localization model...")
run(f'"{PYTHON}" models/train_localization.py --use-cleaned')

# Step 6: Copy all artifacts to models/extended_hanoi/
print("\n[7/7] Copying artifacts to models/extended_hanoi/...")
EH_MODELS.mkdir(exist_ok=True)

model_files = [
    "leak_detection_model_cleaned.pkl",
    "leak_detection_metrics_cleaned.json",
    "stage2_zone_classifier_cleaned.pkl",
    "localization_metrics_cleaned.json",
    "baseline_pressure_model.json",
]
for f in model_files:
    src = MODELS / f
    if src.exists():
        shutil.copy2(src, EH_MODELS / f)
        print(f"  Copied {f}")

# Copy feature names and zone definitions
dataset_files = [
    "feature_names_no_leak.json",
    "localization_feature_names_no_leak.json",
    "zone_definitions.json",
]
for f in dataset_files:
    src = DATASETS / f
    if src.exists():
        shutil.copy2(src, EH_MODELS / f)
        print(f"  Copied {f}")

# Step 7: Restore Balerma datasets
print("\n[RESTORE] Restoring Balerma datasets...")
for f in BACKUP.iterdir():
    shutil.copy2(f, DATASETS / f.name)
print(f"  Restored {len(list(BACKUP.iterdir()))} Balerma dataset files")

# Restore Balerma models to root models/ dir
balerma_models = MODELS / "balerma"
for f in model_files:
    src = balerma_models / f
    if src.exists():
        shutil.copy2(src, MODELS / f)

print("\n" + "="*60)
print("DONE! Extended Hanoi models retrained and saved to models/extended_hanoi/")
print("Balerma datasets restored to DATASETS/")
print("="*60)
