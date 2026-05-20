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
        self.classifier = joblib.load(self.models_dir / "leak_detection_model.pkl")
        self.zone_model = joblib.load(self.models_dir / "stage2_zone_classifier.pkl")
        self.baseline = BaselinePressureModel.load_json(self.models_dir / "baseline_pressure_model.json")

    def extract_detection_features(self, sample_df: pd.DataFrame) -> np.ndarray:
        return FeatureExtractor.build_detection_features(sample_df)

    def extract_localization_features(self, sample_df: pd.DataFrame) -> np.ndarray:
        return LocalizationFeatureExtractor.build_localization_features(sample_df, baseline_hm=self.baseline.to_dict())

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
        zone_id = int(self.zone_model.predict(x)[0])
        proba = float(np.max(self.zone_model.predict_proba(x))) if hasattr(self.zone_model, "predict_proba") else 1.0
        return {"zone_id": zone_id, "zone_confidence": proba}

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
