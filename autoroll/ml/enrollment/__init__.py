"""
AutoRoll Privacy-Preserving Enrollment Package.
"""

from autoroll.ml.enrollment.aggregator import EmbeddingAggregator
from autoroll.ml.enrollment.pipeline import PrivacyPreservingEnrollmentPipeline
from autoroll.ml.enrollment.result import EnrollmentResult, EnrollmentSampleMetadata

__all__ = [
    "EmbeddingAggregator",
    "PrivacyPreservingEnrollmentPipeline",
    "EnrollmentResult",
    "EnrollmentSampleMetadata",
]
