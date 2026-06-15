"""Train and save the leak detection pipeline for Objective 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "DATASETS"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_classification_data(use_cleaned: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x_file = "X_classification_no_leak.npy" if use_cleaned else "X_classification.npy"
    x_path = DATA_DIR / x_file
    y_path = DATA_DIR / "y_classification.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path), np.load(y_path)

    if not use_cleaned:
        dataset_path = DATA_DIR / "classification_B.pt"
        if dataset_path.exists():
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                data = torch.load(dataset_path, weights_only=False)
            return data["X"], data["y"]

    raise FileNotFoundError(
        f"Classification dataset not found. Expected either {x_path} and {y_path} or classification_B.pt"
    )


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    """Split data into train/val/test without stratification to handle rare classes."""
    # First split: 70% train, 30% hold (val+test)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=seed, shuffle=True
    )
    # Second split: 50% val, 50% test from hold set
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=0.5, random_state=seed, shuffle=True
    )

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


def load_feature_names(use_cleaned: bool = False) -> list[str] | None:
    feature_file = "feature_names_no_leak.json" if use_cleaned else "feature_names.json"
    feature_path = DATA_DIR / feature_file
    if feature_path.exists():
        with feature_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def extract_feature_importances(model: Any, feature_names: list[str] | None = None) -> list[Dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    indexed = list(enumerate(importances))
    indexed.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, importance in indexed:
        result.append(
            {
                "feature_index": idx,
                "feature_name": feature_names[idx] if feature_names and idx < len(feature_names) else None,
                "importance": float(importance),
            }
        )
    return result


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leak detection model")
    parser.add_argument(
        "--use-cleaned",
        action="store_true",
        help="Use cleaned classification features from DATASETS/X_classification_no_leak.npy",
    )
    args = parser.parse_args()

    X, y = load_classification_data(use_cleaned=args.use_cleaned)
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
    model_output = "leak_detection_model_cleaned.pkl" if args.use_cleaned else "leak_detection_model.pkl"
    joblib.dump(classifier, MODELS_DIR / model_output)

    feature_names = load_feature_names(use_cleaned=args.use_cleaned)
    feature_importances = extract_feature_importances(classifier, feature_names=feature_names)

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
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "train": evaluate_model(classifier, splits["X_train"], splits["y_train"]),
        "val": evaluate_model(classifier, splits["X_val"], splits["y_val"]),
        "test": evaluate_model(classifier, splits["X_test"], splits["y_test"]),
    }
    metrics_output = "leak_detection_metrics_cleaned.json" if args.use_cleaned else "leak_detection_metrics.json"
    save_metrics(metrics, MODELS_DIR / metrics_output)

    test_accuracy = metrics["test"]["classification_report"]["accuracy"]
    print(f"\nSelected model: {best_name}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {MODELS_DIR / 'leak_detection_model.pkl'}")
    print(f"Metrics saved to: {MODELS_DIR / 'leak_detection_metrics.json'}")

    if test_accuracy < 0.50:
        print(
            "\nWARNING: Test accuracy is below 0.50. "
            "The current detection features may not contain a strong signal for the 4-way leak classification task."
        )


if __name__ == "__main__":
    main()
