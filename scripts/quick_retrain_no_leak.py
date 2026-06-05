"""Quick retrain using fixed hyperparameters on cleaned datasets (no grid search).
Saves metrics to models/*_no_leak_quick.json and models/*_no_leak_quick.pkl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATASETS"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, hold_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[hold_idx], y[hold_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
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


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)
    return {
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def extract_feature_importances(model: Any, feature_names: List[str] | None = None) -> List[Dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    indexed = list(enumerate(importances))
    indexed.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, importance in indexed:
        result.append({
            "feature_index": idx,
            "feature_name": feature_names[idx] if feature_names and idx < len(feature_names) else None,
            "importance": float(importance),
        })
    return result


def retrain_detection_quick():
    X = np.load(DATA_DIR / "X_classification_no_leak.npy")
    y = np.load(DATA_DIR / "y_classification.npy")
    feature_names = json.loads((DATA_DIR / "feature_names_no_leak.json").read_text())
    splits = split_train_val_test(X, y)

    clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1, n_estimators=100, max_depth=20, min_samples_leaf=1, max_features="sqrt")
    clf.fit(splits["X_train"], splits["y_train"])
    joblib.dump(clf, MODELS_DIR / "leak_detection_model_no_leak_quick.pkl")

    metrics = {
        "model_type": clf.__class__.__name__,
        "selected_model": "RandomForest",
        "best_params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1, "max_features": "sqrt"},
        "feature_names": feature_names,
        "feature_importances": extract_feature_importances(clf, feature_names=feature_names),
        "train": evaluate_model(clf, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(clf, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(clf, splits["X_test"], splits["y_test"]),
    }
    out = MODELS_DIR / "leak_detection_metrics_no_leak_quick.json"
    out.write_text(json.dumps(metrics, indent=2))
    print("Saved", out)


def retrain_localization_quick():
    X = np.load(DATA_DIR / "X_localization_no_leak.npy")
    y = np.load(DATA_DIR / "y_localization.npy")
    feature_names = json.loads((DATA_DIR / "localization_feature_names_no_leak.json").read_text())
    splits = split_train_val_test(X, y)

    clf = ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=1, n_estimators=200, max_depth=15, min_samples_leaf=1, max_features="log2")
    clf.fit(splits["X_train"], splits["y_train"])
    joblib.dump(clf, MODELS_DIR / "stage2_zone_classifier_no_leak_quick.pkl")

    metrics = {
        "selected_model": "ExtraTrees",
        "best_params": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 1, "max_features": "log2"},
        "feature_names": feature_names,
        "feature_importances": extract_feature_importances(clf, feature_names=feature_names),
        "train": evaluate_model(clf, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(clf, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(clf, splits["X_test"], splits["y_test"]),
    }
    out = MODELS_DIR / "localization_metrics_no_leak_quick.json"
    out.write_text(json.dumps(metrics, indent=2))
    print("Saved", out)


def main():
    print("Quick retrain on cleaned datasets (no leak-derived features)")
    retrain_localization_quick()
    retrain_detection_quick()


if __name__ == "__main__":
    main()
