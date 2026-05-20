"""Train and save the fault localisation pipeline for Objective 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "DATASETS"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_baseline_reference() -> Dict[str, float]:
    baseline_path = DATA_DIR / "baseline_pressure_reference.json"
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"Baseline reference not found at {baseline_path}")


def load_localization_data() -> tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / "X_localization.npy"
    y_path = DATA_DIR / "y_localization.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path), np.load(y_path)
    raise FileNotFoundError("Localization dataset files are missing in DATASETS/")


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
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


def build_localization_search_space() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    return {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [15, 25, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [15, None],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
        ),
    }


def select_best_localization_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[str, Any, dict, List[Dict[str, Any]]]:
    best_score = -1.0
    best_model = None
    best_name = ""
    best_params: dict = {}
    summary: List[Dict[str, Any]] = []

    for model_name, (estimator, param_grid) in build_localization_search_space().items():
        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid.fit(X_train, y_train)
        summary.append(
            {
                "model_name": model_name,
                "best_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                "estimator_class": estimator.__class__.__name__,
            }
        )

        if grid.best_score_ > best_score:
            best_score = float(grid.best_score_)
            best_model = grid.best_estimator_
            best_name = model_name
            best_params = grid.best_params_

    if best_model is None:
        raise RuntimeError("Failed to select a best localization model from the search grid.")

    return best_name, best_model, best_params, summary


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)
    return {
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def main() -> None:
    X, y = load_localization_data()
    splits = split_train_val_test(X, y)
    baseline_reference = load_baseline_reference()
    with (MODELS_DIR / "baseline_pressure_model.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_reference, handle, indent=2)

    best_name, model, best_params, model_selection_summary = select_best_localization_model(
        splits["X_train"], splits["y_train"]
    )
    joblib.dump(model, MODELS_DIR / "stage2_zone_classifier.pkl")

    metrics = {
        "selected_model": best_name,
        "best_params": best_params,
        "model_selection": model_selection_summary,
        "train": evaluate_model(model, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(model, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(model, splits["X_test"], splits["y_test"]),
    }
    with (MODELS_DIR / "localization_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Saved localisation model and metrics to", MODELS_DIR)
    print(f"Selected model: {best_name}")
    print(f"Localization test accuracy: {metrics['test']['classification_report']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
