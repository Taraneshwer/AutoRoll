"""
AutoRoll Evaluation Package.
"""

from app.ml.evaluation.metrics import (
    DistributionStats,
    TARAtFAR,
    VerificationMetrics,
    VerificationMetricsCalculator,
)
from app.ml.evaluation.plots import DistributionHistogram, EvaluationPlotter, ROCPlotData
from app.ml.evaluation.report import EvaluationReportBuilder
from app.ml.evaluation.threshold import CalibratedThreshold, ThresholdCalibrator
from app.ml.evaluation.verification import (
    FacePair,
    PairEvaluationResult,
    VerificationEvaluator,
)

__all__ = [
    "DistributionStats",
    "TARAtFAR",
    "VerificationMetrics",
    "VerificationMetricsCalculator",
    "ROCPlotData",
    "DistributionHistogram",
    "EvaluationPlotter",
    "CalibratedThreshold",
    "ThresholdCalibrator",
    "FacePair",
    "PairEvaluationResult",
    "VerificationEvaluator",
    "EvaluationReportBuilder",
]
