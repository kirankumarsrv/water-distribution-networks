"""Train and save the leak detection pipeline for Objective 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "DATASETS"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_classification_data() -> tuple[np.ndarray, np.ndarray]:
    x_path = DATA_DIR / "X_classification.npy"
    y_path = DATA_DIR / "y_classification.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path), np.load(y_path)

    dataset_path = DATA_DIR / "classification_B.pt"
    if dataset_path.exists():
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
            data = torch.load(dataset_path, weights_only=False)
        return data["X"], data["y"]

    raise FileNotFoundError(
        f"Classification dataset not found. Expected either {x_path} and {y_path} or {dataset_path}"
    )


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


def build_model_search_space() -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    return {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
        ),
    }


def select_best_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[str, Any, dict, List[Dict[str, Any]]]:
    best_score = -1.0
    best_model = None
    best_name = ""
    best_params: dict = {}
    summary: List[Dict[str, Any]] = []

    for model_name, (estimator, param_grid) in build_model_search_space().items():
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
        raise RuntimeError("Failed to select a best model from the search grid.")

    return best_name, best_model, best_params, summary


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X)
    return {
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    X, y = load_classification_data()
    splits = split_train_val_test(X, y)

    print("\n" + "="*70)
    print("LEAK DETECTION TRAINING")
    print("="*70)
    print("Classes: 0=Normal, 1=Leak, 2=Burst, 3=Blockage")
    print(f"Training on: {splits['X_train'].shape[0]} samples")
    print(f"Validating on: {splits['X_val'].shape[0]} samples")
    print(f"Testing on: {splits['X_test'].shape[0]} samples")

    best_name, classifier, best_params, model_selection_summary = select_best_model(
        splits["X_train"], splits["y_train"]
    )
    joblib.dump(classifier, MODELS_DIR / "leak_detection_model.pkl")

    print("\n" + "="*70)
    print("PERFORMANCE EVALUATION")
    print("="*70)
    metrics = {
        "model_type": classifier.__class__.__name__,
        "num_classes": 4,
        "class_names": ["Normal", "Leak", "Burst", "Blockage"],
        "selected_model": best_name,
        "best_params": best_params,
        "model_selection": model_selection_summary,
        "train": evaluate_model(classifier, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(classifier, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(classifier, splits["X_test"], splits["y_test"]),
    }
    save_metrics(metrics, MODELS_DIR / "leak_detection_metrics.json")

    test_accuracy = metrics["test"]["classification_report"]["accuracy"]
    print(f"\n✓ Selected model: {best_name}")
    print(f"✓ Test Set Accuracy: {test_accuracy:.4f}")
    print(f"✓ Model saved to: {MODELS_DIR / 'leak_detection_model.pkl'}")
    print(f"✓ Metrics saved to: {MODELS_DIR / 'leak_detection_metrics.json'}")

    if test_accuracy < 0.50:
        print(
            "\nWARNING: Test accuracy is below 0.50. "
            "The current detection features may not contain a strong signal for the 4-way leak classification task."
        )


if __name__ == "__main__":
    main()
