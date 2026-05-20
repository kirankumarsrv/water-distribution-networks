"""Feature extraction for fault localisation using residual and spatial cues."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class LocalizationFeatureExtractor:
    """Extract localization features from one sample."""

    SENSOR_COUNT: Optional[int] = None
    EPS = 1e-8

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / (abs(denominator) + LocalizationFeatureExtractor.EPS)

    @staticmethod
    def select_sensor_pipes(df: pd.DataFrame, sensor_count: Optional[int] = None) -> List[str]:
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
    def build_localization_features(
        df: pd.DataFrame,
        baseline_hm: Optional[Dict[str, float]] = None,
        sensor_count: Optional[int] = None,
    ) -> np.ndarray:
        """Build a feature vector for localisation training."""
        pipes = LocalizationFeatureExtractor.select_sensor_pipes(df, sensor_count=sensor_count)
        features: List[float] = []

        for pipe in pipes:
            row = df[df["pipe"] == pipe]
            if row.empty:
                features.append(0.0)
                continue
            row = row.iloc[0]
            Hm = float(row.get("Hm", 0.0))
            baseline = float(baseline_hm.get(pipe, Hm)) if baseline_hm else Hm
            features.append(Hm - baseline)

        for pipe in pipes:
            row = df[df["pipe"] == pipe]
            if row.empty:
                features.append(0.0)
                continue
            row = row.iloc[0]
            features.append(float(row.get("H_in", 0.0)) - float(row.get("H_out", 0.0)))

        total_Q1 = df["Q1"].abs().sum() if "Q1" in df else 0.0
        total_Q2 = df["Q2"].abs().sum() if "Q2" in df else 0.0
        total_Q = df["Q_EPANET"].abs().sum() if "Q_EPANET" in df else 0.0
        features.extend([
            total_Q1 - total_Q2,
            total_Q - total_Q1,
        ])

        for pipe in pipes[:3]:
            row = df[df["pipe"] == pipe]
            if row.empty:
                features.extend([0.0, 0.0, 0.0])
                continue
            row = row.iloc[0]
            Q1 = float(row.get("Q1", 0.0))
            Q2 = float(row.get("Q2", 0.0))
            Q_epanet = float(row.get("Q_EPANET", 0.0))
            features.extend([
                LocalizationFeatureExtractor._safe_ratio(Q1, Q_epanet),
                LocalizationFeatureExtractor._safe_ratio(Q2, Q_epanet),
                LocalizationFeatureExtractor._safe_ratio(abs(Q1 - Q2), abs(Q_epanet)),
            ])

        Hm = df["Hm"].astype(float)
        H_in = df["H_in"].astype(float)
        H_out = df["H_out"].astype(float)
        Q1 = df["Q1"].astype(float)
        Q2 = df["Q2"].astype(float)
        Q_leak = df["Q_leak"].astype(float)
        features.extend([
            float(Hm.mean()),
            float(Hm.std()),
            float(H_in.mean()),
            float(H_out.mean()),
            float((H_in - H_out).abs().mean()),
            float(Q1.mean()),
            float(Q2.mean()),
            float(Q_leak.mean()),
            float(LocalizationFeatureExtractor._safe_ratio(Q_leak.sum(), Q1.abs().sum())),
            float(LocalizationFeatureExtractor._safe_ratio(Q_leak.sum(), Q2.abs().sum())),
            float(LocalizationFeatureExtractor._safe_ratio(Q1.abs().sum(), Q2.abs().sum())),
        ])

        if sensor_count is None:
            sensor_count = len(df)
        expected_length = sensor_count * 2 + 2 + 9 + 11
        while len(features) < expected_length:
            features.append(0.0)

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def get_feature_names(sensor_count: Optional[int] = None) -> List[str]:
        if sensor_count is None:
            raise ValueError("sensor_count must be provided to generate deterministic feature names")

        names: List[str] = []
        for idx in range(sensor_count):
            names.append(f"residual_{idx+1}")
        for idx in range(sensor_count):
            names.append(f"gradient_{idx+1}")
        names.extend(["imbalance_Q1_Q2", "imbalance_Q_total"])
        for idx in range(min(3, sensor_count)):
            names.extend([
                f"ratio_Q1_Q_epanet_{idx+1}",
                f"ratio_Q2_Q_epanet_{idx+1}",
                f"ratio_deltaQ_Q_epanet_{idx+1}",
            ])
        names.extend([
            "Hm_mean",
            "Hm_std",
            "H_in_mean",
            "H_out_mean",
            "delta_H_mean",
            "Q1_mean",
            "Q2_mean",
            "Q_leak_mean",
            "leak_share_Q1",
            "leak_share_Q2",
            "flow_balance_ratio",
        ])
        return names
