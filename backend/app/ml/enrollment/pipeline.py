"""
Privacy-Preserving Multi-Sample Face Enrollment Pipeline.
"""

import os

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.enrollment.aggregator import EmbeddingAggregator
from app.ml.enrollment.result import EnrollmentResult, EnrollmentSampleMetadata
from app.ml.preprocessing.quality import FaceQualityFilter
from app.ml.recognition.arcface_iresnet import ArcFaceRecognizer

logger = get_logger("privacy_enrollment_pipeline")
settings = get_settings()


class PrivacyPreservingEnrollmentPipeline:
    """
    Privacy-Preserving Multi-Sample Face Enrollment Pipeline.
    Extracts, filters, and aggregates 512-d ArcFace embeddings without storing raw face photos.
    """

    def __init__(
        self,
        detector: SCRFDDetector | None = None,
        aligner: FaceAligner | None = None,
        recognizer: ArcFaceRecognizer | None = None,
        quality_filter: FaceQualityFilter | None = None,
        device: str = "auto",
        min_accepted_samples: int = 1,
    ):
        self.device = device
        self.min_accepted_samples = min_accepted_samples

        self.detector = detector or SCRFDDetector(device=device)
        self.aligner = aligner or FaceAligner()
        self.recognizer = recognizer or ArcFaceRecognizer(device=device)
        self.quality_filter = quality_filter or FaceQualityFilter(
            min_face_size=40, min_blur_score=15.0, min_confidence=0.60
        )

    def enroll(
        self,
        student_code: str,
        full_name: str,
        sample_inputs: list[str | np.ndarray],
        delete_raw_images: bool = True,
    ) -> EnrollmentResult:
        """
        Processes face samples for student enrollment.
        Aggregates embeddings into a single 512-d normalized centroid and deletes
        temporary sample images.
        """
        logger.info(
            f"Starting Privacy-Preserving Enrollment for '{full_name}' ({student_code}) "
            f"with {len(sample_inputs)} sample inputs..."
        )

        sample_metadatas: list[EnrollmentSampleMetadata] = []
        valid_embeddings: list[list[float]] = []
        rejection_reasons: list[str] = []
        paths_to_cleanup: list[str] = []

        for idx, sample_input in enumerate(sample_inputs):
            image: np.ndarray | None = None
            if isinstance(sample_input, str):
                if delete_raw_images:
                    paths_to_cleanup.append(sample_input)
                if not os.path.exists(sample_input):
                    reason = f"Sample #{idx + 1}: Image file '{sample_input}' not found."
                    rejection_reasons.append(reason)
                    continue
                image = cv2.imread(sample_input)
            elif isinstance(sample_input, np.ndarray):
                image = sample_input

            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                reason = f"Sample #{idx + 1}: Invalid or unreadable image frame."
                rejection_reasons.append(reason)
                continue

            # 1. SCRFD Detection
            detections = self.detector.detect(image)

            # 2. Reject if no face or multiple faces
            if len(detections) == 0:
                reason = f"Sample #{idx + 1}: No face detected."
                rejection_reasons.append(reason)
                sample_metadatas.append(
                    EnrollmentSampleMetadata(
                        sample_index=idx + 1,
                        passed=False,
                        reason=reason,
                        blur_score=0.0,
                        face_width=0.0,
                        face_height=0.0,
                        detection_confidence=0.0,
                    )
                )
                continue

            if len(detections) > 1:
                reason = (
                    f"Sample #{idx + 1}: Multiple faces ({len(detections)}) detected. "
                    "Enrollment image must contain exactly one face."
                )
                rejection_reasons.append(reason)
                sample_metadatas.append(
                    EnrollmentSampleMetadata(
                        sample_index=idx + 1,
                        passed=False,
                        reason=reason,
                        blur_score=0.0,
                        face_width=0.0,
                        face_height=0.0,
                        detection_confidence=0.0,
                    )
                )
                continue

            det = detections[0]

            # 3. Quality Filtering
            q_res = self.quality_filter.evaluate(image, det)
            if not q_res.passed:
                reason = f"Sample #{idx + 1} Quality Failed: {q_res.reason}"
                rejection_reasons.append(reason)
                sample_metadatas.append(
                    EnrollmentSampleMetadata(
                        sample_index=idx + 1,
                        passed=False,
                        reason=reason,
                        blur_score=q_res.blur_score,
                        face_width=q_res.face_width,
                        face_height=q_res.face_height,
                        detection_confidence=det.det_confidence,
                    )
                )
                continue

            # 4. Alignment & ArcFace Feature Extraction
            try:
                aligned_chip = self.aligner.align(image, det.landmarks)
                rec_res = self.recognizer.extract_embedding(aligned_chip)
                valid_embeddings.append(rec_res.embedding)

                sample_metadatas.append(
                    EnrollmentSampleMetadata(
                        sample_index=idx + 1,
                        passed=True,
                        reason="Accepted",
                        blur_score=q_res.blur_score,
                        face_width=q_res.face_width,
                        face_height=q_res.face_height,
                        detection_confidence=det.det_confidence,
                    )
                )
            except Exception as e:
                reason = f"Sample #{idx + 1}: Embedding extraction error ({e})."
                rejection_reasons.append(reason)

        # Cleanup raw image files from disk if requested
        if delete_raw_images:
            for p in paths_to_cleanup:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        logger.info(f"Deleted temporary raw sample image '{p}'.")
                except Exception as e:
                    logger.warning(f"Could not delete temp image '{p}': {e}")

        accepted_count = len(valid_embeddings)
        if accepted_count < self.min_accepted_samples:
            logger.warning(
                f"Enrollment FAILED for '{full_name}': Only "
                f"{accepted_count}/{self.min_accepted_samples} "
                "required samples passed quality checks."
            )
            return EnrollmentResult(
                success=False,
                student_code=student_code,
                full_name=full_name,
                samples_processed=len(sample_inputs),
                samples_accepted=accepted_count,
                rejection_reasons=rejection_reasons,
                quality_metadata={"samples": [m.model_dump() for m in sample_metadatas]},
                message=(
                    f"Enrollment failed. {accepted_count} valid samples accepted "
                    f"(Minimum required: {self.min_accepted_samples})."
                ),
            )

        # 5. Embedding Aggregation & Normalization
        aggregated_vec = EmbeddingAggregator.aggregate(valid_embeddings)

        logger.info(
            f"Enrollment SUCCESSFUL for '{full_name}' ({student_code}). "
            f"Accepted {accepted_count}/{len(sample_inputs)} samples."
        )

        return EnrollmentResult(
            success=True,
            student_code=student_code,
            full_name=full_name,
            samples_processed=len(sample_inputs),
            samples_accepted=accepted_count,
            rejection_reasons=rejection_reasons,
            aggregated_embedding=aggregated_vec,
            model_version=settings.MODEL_VERSION,
            quality_metadata={"samples": [m.model_dump() for m in sample_metadatas]},
            message="Student face embedding successfully enrolled.",
        )
