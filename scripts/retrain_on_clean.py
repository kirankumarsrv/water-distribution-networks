"""Retrain localization and leak-detection models using cleaned datasets
(no leak-derived features). Saves metrics and models with a `_no_leak` suffix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

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


def train_and_save_classification(clean_suffix: str = "no_leak") -> None:
    X = np.load(DATA_DIR / f"X_classification_{clean_suffix}.npy")
    y = np.load(DATA_DIR / "y_classification.npy")
    feature_names = json.loads((DATA_DIR / f"feature_names_{clean_suffix}.json").read_text())

    splits = split_train_val_test(X, y)

    search_space = {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]},
        ),
    }

    best_score = -1.0
    best_model = None
    best_name = ""
    best_params = {}
    summary = []
    for model_name, (estimator, param_grid) in search_space.items():
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0, refit=True)
        grid.fit(splits["X_train"], splits["y_train"])
        summary.append({"model_name": model_name, "best_score": float(grid.best_score_), "best_params": grid.best_params_})
        if grid.best_score_ > best_score:
            best_score = float(grid.best_score_)
            best_model = grid.best_estimator_
            best_name = model_name
            best_params = grid.best_params_

    joblib.dump(best_model, MODELS_DIR / f"leak_detection_model_{clean_suffix}.pkl")

    metrics = {
        "model_type": best_model.__class__.__name__,
        "selected_model": best_name,
        "best_params": best_params,
        "model_selection": summary,
        "feature_names": feature_names,
        "feature_importances": extract_feature_importances(best_model, feature_names=feature_names),
        "train": evaluate_model(best_model, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(best_model, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(best_model, splits["X_test"], splits["y_test"]),
    }

    out = MODELS_DIR / f"leak_detection_metrics_{clean_suffix}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print("Saved leak detection metrics to", out)


def train_and_save_localization(clean_suffix: str = "no_leak") -> None:
    X = np.load(DATA_DIR / f"X_localization_{clean_suffix}.npy")
    y = np.load(DATA_DIR / "y_localization.npy")
    feature_names = json.loads((DATA_DIR / f"localization_feature_names_{clean_suffix}.json").read_text())

    splits = split_train_val_test(X, y)

    search_space = {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [15, 25, None], "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [15, None], "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"]},
        ),
    }

    best_score = -1.0
    best_model = None
    best_name = ""
    best_params = {}
    summary = []
    for model_name, (estimator, param_grid) in search_space.items():
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0, refit=True)
        grid.fit(splits["X_train"], splits["y_train"])
        summary.append({"model_name": model_name, "best_score": float(grid.best_score_), "best_params": grid.best_params_})
        if grid.best_score_ > best_score:
            best_score = float(grid.best_score_)
            best_model = grid.best_estimator_
            best_name = model_name
            best_params = grid.best_params_

    joblib.dump(best_model, MODELS_DIR / f"stage2_zone_classifier_{clean_suffix}.pkl")

    metrics = {
        "selected_model": best_name,
        "best_params": best_params,
        "model_selection": summary,
        "feature_names": feature_names,
        "feature_importances": extract_feature_importances(best_model, feature_names=feature_names),
        "train": evaluate_model(best_model, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(best_model, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(best_model, splits["X_test"], splits["y_test"]),
    }

    out = MODELS_DIR / f"localization_metrics_{clean_suffix}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print("Saved localization metrics to", out)


def main() -> None:
    print("Retraining on cleaned datasets (no leak-derived features)")
    train_and_save_localization(clean_suffix="no_leak")
    train_and_save_classification(clean_suffix="no_leak")


if __name__ == "__main__":
    main()
