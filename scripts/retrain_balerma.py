"""Retrain Balerma models with more training data for better localization accuracy.

This script:
1. Backs up current DATASETS
2. Generates Balerma datasets with ~15,000 samples (up from 3,600)
3. Cleans features (removes leak-related features)
4. Retrains detection and localization models
5. Copies all artifacts to models/balerma/
6. Restores backed-up DATASETS
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "DATASETS"
BACKUP = ROOT / "DATASETS_backup_before_balerma_retrain"
MODELS = ROOT / "models"
BALERMA_MODELS = MODELS / "balerma"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INP_FILE = ROOT / "EPANETINPUTFILESFOR7NEWORKS" / "5_Balerma.inp"

# 454 pipes × 3 fault types × ~11 samples each ≈ 15,000 fault samples + 3,750 normal
N_SAMPLES = 18000


def run(cmd, **kwargs):
    print(f"\n{'='*60}\n>>> {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT), **kwargs)
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(1)


def main():
    start_time = time.time()

    # Step 1: Backup current DATASETS
    print("\n[1/7] Backing up current DATASETS...")
    key_files = [
        "X_classification.npy", "y_classification.npy",
        "X_classification_no_leak.npy",
        "X_localization.npy", "y_localization.npy",
        "X_localization_no_leak.npy",
        "feature_names.json", "feature_names_no_leak.json",
        "localization_feature_names.json", "localization_feature_names_no_leak.json",
        "zone_definitions.json", "baseline_pressure_reference.json",
        "classification_B.pt",
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "X_loc_train.npy", "X_loc_val.npy", "X_loc_test.npy",
        "y_loc_train.npy", "y_loc_val.npy", "y_loc_test.npy",
    ]
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    BACKUP.mkdir(exist_ok=True)
    backed_up = 0
    for f in key_files:
        src = DATASETS / f
        if src.exists():
            shutil.copy2(src, BACKUP / f)
            backed_up += 1
    print(f"  Backed up {backed_up} files to {BACKUP}")

    # Step 2: Generate Balerma datasets with MORE samples
    print(f"\n[2/7] Generating Balerma dataset with {N_SAMPLES} samples...")
    print(f"  INP file: {INP_FILE}")
    print(f"  This will take a LONG time (~30-60 minutes for {N_SAMPLES} samples)")
    print(f"  Previous dataset had 3,600 samples -> now {N_SAMPLES} (5x more)")
    run(f'"{PYTHON}" graph_dataset/dataset.py --inp-file "{INP_FILE}" --samples {N_SAMPLES}')

    elapsed = time.time() - start_time
    print(f"\n  Dataset generation completed in {elapsed/60:.1f} minutes")

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

    # Step 6: Copy all artifacts to models/balerma/
    print("\n[7/7] Copying artifacts to models/balerma/...")
    BALERMA_MODELS.mkdir(exist_ok=True)

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
            shutil.copy2(src, BALERMA_MODELS / f)
            print(f"  Copied {f}")

    dataset_files = [
        "feature_names_no_leak.json",
        "localization_feature_names_no_leak.json",
        "zone_definitions.json",
    ]
    for f in dataset_files:
        src = DATASETS / f
        if src.exists():
            shutil.copy2(src, BALERMA_MODELS / f)
            print(f"  Copied {f}")

    # Step 7: Restore original DATASETS
    print("\n[RESTORE] Restoring original DATASETS...")
    for f in BACKUP.iterdir():
        shutil.copy2(f, DATASETS / f.name)
    print(f"  Restored {len(list(BACKUP.iterdir()))} dataset files")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE! Balerma models retrained with {N_SAMPLES} samples")
    print(f"Models saved to: {BALERMA_MODELS}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
