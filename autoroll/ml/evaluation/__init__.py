"""
AutoRoll Evaluation Package.
"""

from autoroll.ml.evaluation.metrics import (
    DistributionStats,
    TARAtFAR,
    VerificationMetrics,
    VerificationMetricsCalculator,
)
from autoroll.ml.evaluation.plots import DistributionHistogram, EvaluationPlotter, ROCPlotData
from autoroll.ml.evaluation.report import EvaluationReportBuilder
from autoroll.ml.evaluation.threshold import CalibratedThreshold, ThresholdCalibrator
from autoroll.ml.evaluation.verification import (
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
