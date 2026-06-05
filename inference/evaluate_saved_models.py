"""Evaluate saved models with extended metrics and latency measurements.

Loads saved models from `models/`, the datasets from `DATASETS/`, computes
precision/recall/F1, confusion matrices, balanced accuracy, top-k accuracy
(for localization), and simple inference latency estimates.

Saves a consolidated JSON report to `models/evaluation_summary.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATASETS"
MODELS_DIR = ROOT / "models"


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, hold_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[hold_idx], y[hold_idx]

    val_ratio = 0.5
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    val_idx, test_idx = next(splitter2.split(X_hold, y_hold))
    X_val, y_val = X_hold[val_idx], y_hold[val_idx]
    X_test, y_test = X_hold[test_idx], y_hold[test_idx]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def evaluate_classifier(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()
    bal = float(balanced_accuracy_score(y, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(y, y_pred, zero_division=0)
    detailed = {
        "classification_report": report,
        "confusion_matrix": cm,
        "balanced_accuracy": bal,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
    }
    return detailed


def measure_latency(model: Any, X_sample: np.ndarray, repeats: int = 200) -> float:
    # Warm-up
    for _ in range(5):
        _ = model.predict(X_sample.reshape(1, -1))
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(X_sample.reshape(1, -1))
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / repeats * 1000.0
    return float(avg_ms)


def main() -> None:
    summary: Dict[str, Any] = {}

    # Leak detection evaluation
    clf_path = MODELS_DIR / "leak_detection_model.pkl"
    X_clf_path = DATA_DIR / "X_classification.npy"
    y_clf_path = DATA_DIR / "y_classification.npy"

    if clf_path.exists() and X_clf_path.exists() and y_clf_path.exists():
        clf = joblib.load(clf_path)
        Xc = np.load(X_clf_path)
        yc = np.load(y_clf_path)
        splits = split_train_val_test(Xc, yc)
        summary["leak_detection"] = {
            "model_type": clf.__class__.__name__,
            "train": evaluate_classifier(clf, splits["X_train"], splits["y_train"]),
            "val": evaluate_classifier(clf, splits["X_val"], splits["y_val"]),
            "test": evaluate_classifier(clf, splits["X_test"], splits["y_test"]),
        }
        # latency on a random test sample
        sample = splits["X_test"][0]
        summary["leak_detection"]["latency_ms"] = measure_latency(clf, sample)
    else:
        summary["leak_detection"] = {"error": "Missing model or dataset files."}

    # Localization evaluation
    loc_path = MODELS_DIR / "stage2_zone_classifier.pkl"
    X_loc_path = DATA_DIR / "X_localization.npy"
    y_loc_path = DATA_DIR / "y_localization.npy"

    if loc_path.exists() and X_loc_path.exists() and y_loc_path.exists():
        loc = joblib.load(loc_path)
        Xl = np.load(X_loc_path)
        yl = np.load(y_loc_path)
        splits_l = split_train_val_test(Xl, yl)
        summary["localization"] = {
            "model_type": loc.__class__.__name__,
            "train": evaluate_classifier(loc, splits_l["X_train"], splits_l["y_train"]),
            "val": evaluate_classifier(loc, splits_l["X_val"], splits_l["y_val"]),
            "test": evaluate_classifier(loc, splits_l["X_test"], splits_l["y_test"]),
        }
        # top-2 accuracy if probabilities are available
        try:
            y_score = loc.predict_proba(splits_l["X_test"])
            top2 = float(top_k_accuracy_score(yl, y_score, k=2))
            summary["localization"]["top2_accuracy"] = top2
        except Exception:
            summary["localization"]["top2_accuracy"] = None
        sample_l = splits_l["X_test"][0]
        summary["localization"]["latency_ms"] = measure_latency(loc, sample_l)
    else:
        summary["localization"] = {"error": "Missing model or dataset files."}

    out_path = MODELS_DIR / "evaluation_summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved evaluation summary to: {out_path}")


if __name__ == "__main__":
    main()
