"""Performance metrics calculations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DetectionMetrics:
    """Metrics for fault detection (Objective 1-2)."""
    
    accuracy: float
    """Overall detection accuracy (0-1, target ≥0.90)."""
    
    precision: float
    """Precision of fault detection (0-1)."""
    
    recall: float
    """Recall/sensitivity of fault detection (0-1, target ≥0.90)."""
    
    f1_score: float
    """F1-score combining precision and recall."""
    
    false_positive_rate: float
    """False alarm rate (0-1, target <0.05)."""
    
    false_negative_rate: float
    """Missed fault rate (0-1, target <0.10)."""
    
    detection_latency_seconds: float
    """Mean time from fault onset to detection alert (target <30s)."""
    
    detection_latency_std: float
    """Standard deviation of detection latency."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LocalizationMetrics:
    """Metrics for fault localization (Objective 3)."""
    
    zone_accuracy: float
    """Accuracy of zone classification (0-1, target ≥0.80)."""
    
    zone_precision: float
    """Precision of zone prediction."""
    
    zone_recall: float
    """Recall of zone prediction."""
    
    top_3_accuracy: float
    """Accuracy when considering top-3 predicted zones."""
    
    mean_reciprocal_rank: float
    """Mean reciprocal rank of ground truth in predictions (0-1)."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class IsolationMetrics:
    """Metrics for valve isolation (Objective 4)."""
    
    mean_response_time_seconds: float
    """Mean time from detection to valve closure command (target <120s)."""
    
    response_time_std: float
    """Standard deviation of response time."""
    
    mean_customers_affected: float
    """Mean number of customers in isolation zone."""
    
    max_customers_affected: float
    """Maximum customers affected in any isolation event."""
    
    customer_disruption_index: float
    """Customers × disruption duration / total customers (target <0.15)."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RestorationMetrics:
    """Metrics for supply restoration (Objective 5)."""
    
    restoration_success_rate: float
    """% of isolated customers successfully restored (target ≥0.60)."""
    
    mean_restoration_time_seconds: float
    """Mean time to validate and execute restoration (~120s)."""
    
    mean_customers_restored: float
    """Mean number of customers restored per event."""
    
    restoration_feasibility_rate: float
    """% of isolation events with feasible restoration paths."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemMetrics:
    """Overall system performance metrics."""
    
    water_loss_reduction_percent: float
    """Reduction in water loss vs manual response (target ≥85%)."""
    
    end_to_end_latency_seconds: float
    """Total time from fault onset to supply restoration (target <300s)."""
    
    system_reliability_percent: float
    """Uptime % of monitoring system (target >99%)."""
    
    detection: DetectionMetrics
    """Fault detection metrics."""
    
    localization: LocalizationMetrics
    """Fault localization metrics."""
    
    isolation: IsolationMetrics
    """Valve isolation metrics."""
    
    restoration: RestorationMetrics
    """Supply restoration metrics."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "water_loss_reduction_percent": self.water_loss_reduction_percent,
            "end_to_end_latency_seconds": self.end_to_end_latency_seconds,
            "system_reliability_percent": self.system_reliability_percent,
            "detection": self.detection.to_dict(),
            "localization": self.localization.to_dict(),
            "isolation": self.isolation.to_dict(),
            "restoration": self.restoration.to_dict(),
        }
