"""Real-time leak detection and localisation pipeline.

This module loads trained models and applies them to new EPANET-derived samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional




import joblib
import numpy as np
import pandas as pd

from physics.FeatureExtractor import FeatureExtractor
from physics.LocalizationFeatureExtractor import LocalizationFeatureExtractor
from models.baseline_pressure_model import BaselinePressureModel

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


class RealTimeLeakDetector:
    """Encapsulate the single-stage leak detection and localisation pipeline."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        # Try to load cleaned models first (for network-specific models like balerma)
        # Otherwise fall back to standard model names
        detection_model = self.models_dir / "leak_detection_model_cleaned.pkl"
        if not detection_model.exists():
            detection_model = self.models_dir / "leak_detection_model.pkl"
        
        zone_model = self.models_dir / "stage2_zone_classifier_cleaned.pkl"
        if not zone_model.exists():
            zone_model = self.models_dir / "stage2_zone_classifier.pkl"
        
        baseline_model = self.models_dir / "baseline_pressure_model.json"
        
        self.classifier = joblib.load(detection_model)
        self.zone_model = joblib.load(zone_model)
        self.baseline = BaselinePressureModel.load_json(baseline_model)

        self._detection_feature_names = (
            self._load_feature_names(self.models_dir / "feature_names_no_leak.json")
        )
        self._localization_feature_names = (
            self._load_feature_names(self.models_dir / "localization_feature_names_no_leak.json")
        )
        self._zone_definitions = (
            self._load_zone_definitions(self.models_dir / "zone_definitions.json")
            or self._load_zone_definitions(ROOT_DIR / "DATASETS" / "zone_definitions.json")
        )

    def _load_feature_names(self, path: Path) -> list[str] | None:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None
        return None

    def _load_zone_definitions(self, path: Path) -> dict[str, int]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _filter_feature_vector(features: np.ndarray, feature_names: list[str], keep_names: list[str]) -> np.ndarray:
        if len(features) != len(feature_names):
            raise ValueError("Feature vector length does not match feature names length")
        keep_indices = [idx for idx, name in enumerate(feature_names) if name in keep_names]
        return features[keep_indices]

    def extract_detection_features(self, sample_df: pd.DataFrame) -> np.ndarray:
        features = FeatureExtractor.build_detection_features(sample_df)
        if self._detection_feature_names is not None:
            names = FeatureExtractor.get_feature_names(sensor_count=len(sample_df))
            return self._filter_feature_vector(features, names, self._detection_feature_names)
        return features

    def extract_localization_features(self, sample_df: pd.DataFrame) -> np.ndarray:
        features = LocalizationFeatureExtractor.build_localization_features(sample_df, baseline_hm=self.baseline.to_dict())
        if self._localization_feature_names is not None:
            names = LocalizationFeatureExtractor.get_feature_names(sensor_count=len(sample_df))
            return self._filter_feature_vector(features, names, self._localization_feature_names)
        return features

    def detect(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        x = self.extract_detection_features(sample_df).reshape(1, -1)
        label = int(self.classifier.predict(x)[0])
        confidence = float(np.max(self.classifier.predict_proba(x))) if hasattr(self.classifier, "predict_proba") else 1.0
        anomaly_flag = 1 if label != 0 else 0
        detection = {
            "anomaly_flag": anomaly_flag,
            "stage1_raw": label,
            "label": label,
            "confidence": confidence,
        }
        return detection


    def localize(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        x = self.extract_localization_features(sample_df).reshape(1, -1)
            
        prediction = self.zone_model.predict(x)[0]
        predicted_zone_id = int(prediction)
        proba = self.zone_model.predict_proba(x)[0] if hasattr(self.zone_model, 'predict_proba') else None

        valid_zone_ids = set(self._zone_definitions.values())
        zone_id = predicted_zone_id if predicted_zone_id in valid_zone_ids else None
        if proba is None:
            result = {
                "zone_id": zone_id,
                "zone_confidence": 1.0,
                "top_zones": [{"zone_id": predicted_zone_id, "probability": 1.0}],
            }
            if zone_id is None:
                result["rare_label"] = int(predicted_zone_id)
            return result

        classes = self.zone_model.classes_
        zone_probs = [float(p) for p in proba]
        sorted_indices = list(np.argsort(zone_probs)[::-1])
        top_zones = [
            {"zone_id": int(classes[i]), "probability": zone_probs[i]}
            for i in sorted_indices
        ]
        result = {
            "zone_id": zone_id,
            "zone_confidence": float(np.max(zone_probs)),
            "top_zones": top_zones,
        }
        if zone_id is None:
            result["rare_label"] = int(predicted_zone_id)
        return result

    def infer(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        detection = self.detect(sample_df)
        result = {"detection": detection}
        if detection.get("label", 0) != 0:
            result["localization"] = self.localize(sample_df)
        return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run real-time leak detection on one sample DataFrame JSON file.")
    parser.add_argument("--sample", required=True, help="Path to a JSON file containing one sample as a list of dict rows.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Directory with trained models.")
    args = parser.parse_args()

    with open(args.sample, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    sample_df = pd.DataFrame(rows)
    detector = RealTimeLeakDetector(models_dir=Path(args.models_dir))
    result = detector.infer(sample_df)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
