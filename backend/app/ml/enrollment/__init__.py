"""
AutoRoll Privacy-Preserving Enrollment Package.
"""

from app.ml.enrollment.aggregator import EmbeddingAggregator
from app.ml.enrollment.pipeline import PrivacyPreservingEnrollmentPipeline
from app.ml.enrollment.result import EnrollmentResult, EnrollmentSampleMetadata

__all__ = [
    "EmbeddingAggregator",
    "PrivacyPreservingEnrollmentPipeline",
    "EnrollmentResult",
    "EnrollmentSampleMetadata",
]
