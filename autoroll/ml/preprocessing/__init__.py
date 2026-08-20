"""
AutoRoll Dataset Preprocessing Package.
"""

from autoroll.ml.preprocessing.aligner import DatasetFaceAligner
from autoroll.ml.preprocessing.dataset_loader import (
    BaseDatasetLoader,
    DatasetConfig,
    DirectoryDatasetLoader,
    RawImageRecord,
    SyntheticDatasetLoader,
)
from autoroll.ml.preprocessing.detector import DatasetFaceDetector
from autoroll.ml.preprocessing.metadata import MetadataManager, PreprocessingReport
from autoroll.ml.preprocessing.quality import FaceQualityFilter, QualityCheckResult
from autoroll.ml.preprocessing.splitter import IdentityDisjointSplitter

__all__ = [
    "DatasetConfig",
    "RawImageRecord",
    "BaseDatasetLoader",
    "DirectoryDatasetLoader",
    "SyntheticDatasetLoader",
    "DatasetFaceDetector",
    "DatasetFaceAligner",
    "FaceQualityFilter",
    "QualityCheckResult",
    "IdentityDisjointSplitter",
    "MetadataManager",
    "PreprocessingReport",
]
