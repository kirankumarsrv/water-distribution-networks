"""Baseline pressure model for localization residual features."""

from __future__ import annotations

import json
from typing import Dict, Any


class BaselinePressureModel:
    """Store a baseline pressure lookup and compute residuals."""

    def __init__(self, baseline_hm: Dict[str, float]):
        self.baseline_hm = dict(baseline_hm)

    def predict(self, pipe_name: str) -> float:
        return self.baseline_hm.get(pipe_name, 0.0)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.baseline_hm)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BaselinePressureModel":
        return cls(data)

    @classmethod
    def load_json(cls, path: str) -> "BaselinePressureModel":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.baseline_hm, handle, indent=2)
