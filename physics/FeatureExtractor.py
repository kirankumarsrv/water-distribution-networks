"""Feature extraction for leak detection classification.

This module converts per-pipe EPANET simulation outputs into a fixed-size
feature vector suitable for training a tabular classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureExtractor:
    """Extract detection features from EPANET + leak physics outputs."""

    SENSOR_COUNT: Optional[int] = None
    EPS = 1e-8

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / (abs(denominator) + FeatureExtractor.EPS)

    @staticmethod
    def select_sensor_pipes(df: pd.DataFrame, sensor_count: Optional[int] = None) -> List[str]:
        """Select the most informative pipes to act as virtual sensors."""
        available_count = len(df)
        if sensor_count is None or sensor_count > available_count:
            sensor_count = available_count

        if "Q_EPANET" in df.columns:
            sorted_df = df.copy()
            sorted_df["abs_flow"] = sorted_df["Q_EPANET"].abs()
            selected = sorted_df.sort_values(by="abs_flow", ascending=False).head(sensor_count)["pipe"].tolist()
        else:
            selected = df["pipe"].tolist()[:sensor_count]
        return selected

    @staticmethod
    def build_detection_features(df: pd.DataFrame, sensor_count: Optional[int] = None) -> np.ndarray:
        """Build a feature vector for leak detection using the strongest pipes."""
        pipes = FeatureExtractor.select_sensor_pipes(df, sensor_count=sensor_count)
        features: List[float] = []

        for pipe in pipes:
            row = df[df["pipe"] == pipe]
            if row.empty:
                values = [0.0] * 9
            else:
                row = row.iloc[0]
                H_in = float(row.get("H_in", 0.0))
                H_out = float(row.get("H_out", 0.0))
                Hm = float(row.get("Hm", 0.0))
                Q_epanet = float(row.get("Q_EPANET", 0.0))
                Q1 = float(row.get("Q1", 0.0))
                Q2 = float(row.get("Q2", 0.0))
                Q_leak = float(row.get("Q_leak", 0.0))

                features.extend([
                    H_in,
                    H_out,
                    Hm,
                    Q_epanet,
                    Q1,
                    Q2,
                    abs(H_in - H_out),
                    abs(Q1 - Q2),
                    FeatureExtractor._safe_ratio(Q_leak, Q_epanet),
                ])

        Hm = df["Hm"].astype(float)
        H_in = df["H_in"].astype(float)
        H_out = df["H_out"].astype(float)
        Q1 = df["Q1"].astype(float)
        Q2 = df["Q2"].astype(float)
        Q_epanet = df["Q_EPANET"].astype(float)
        Q_leak = df["Q_leak"].astype(float)

        features.extend([
            float(Q_leak.sum()),
            float(np.abs(Q_epanet).sum()),
            float(np.abs(Q1).sum()),
            float(np.abs(Q2).sum()),
            float(np.abs(H_in - H_out).mean()),
            float(np.abs(Q1 - Q2).mean()),
            float(Hm.mean()),
            float(Hm.std()),
            float(FeatureExtractor._safe_ratio(Q_leak.sum(), np.abs(Q_epanet).sum())),
            float(FeatureExtractor._safe_ratio(Q_leak.sum(), np.abs(Q1).sum())),
            float(FeatureExtractor._safe_ratio(Q_leak.sum(), np.abs(Q2).sum())),
        ])

        if sensor_count is None:
            sensor_count = len(df)
        expected_length = sensor_count * 9 + 11
        while len(features) < expected_length:
            features.append(0.0)

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def get_feature_names(sensor_count: Optional[int] = None) -> List[str]:
        """Return the ordered feature names for the detection vector."""
        if sensor_count is None:
            raise ValueError("sensor_count must be provided to generate deterministic feature names")

        names = []
        for idx in range(sensor_count):
            prefix = f"sensor_{idx+1}"
            names.extend([
                f"{prefix}_H_in",
                f"{prefix}_H_out",
                f"{prefix}_Hm",
                f"{prefix}_Q_epanet",
                f"{prefix}_Q1",
                f"{prefix}_Q2",
                f"{prefix}_delta_H",
                f"{prefix}_delta_Q",
                f"{prefix}_leak_ratio",
            ])

        names.extend([
            "total_Q_leak",
            "total_abs_Q_epanet",
            "total_abs_Q1",
            "total_abs_Q2",
            "mean_abs_delta_H",
            "mean_abs_delta_Q",
            "mean_Hm",
            "std_Hm",
            "leak_share_Q_epanet",
            "leak_share_Q1",
            "leak_share_Q2",
        ])
        return names
