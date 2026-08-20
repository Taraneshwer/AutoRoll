"""
Dataset Metadata and Preprocessing Manifest Reporter.
Records all processed items, split distributions, and failed image reasons.
"""

import json
import os

from pydantic import BaseModel, Field

from autoroll.common.logger import get_logger

logger = get_logger("metadata_manager")


class FailedItemRecord(BaseModel):
    image_path: str
    identity_id: str
    failure_reason: str


class PreprocessingReport(BaseModel):
    dataset_name: str
    dataset_version: str
    preprocessing_version: str
    total_raw_images: int = 0
    total_processed_images: int = 0
    total_failed_images: int = 0
    total_identities: int = 0
    splits_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    failed_items: list[FailedItemRecord] = Field(default_factory=list)


class MetadataManager:
    """
    Manages generation, update, and persistence of dataset metadata and manifest logs.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_version: str = "1.0.0",
        preprocessing_version: str = "1.0.0",
    ):
        self.report = PreprocessingReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            preprocessing_version=preprocessing_version,
        )

    def record_failure(self, image_path: str, identity_id: str, reason: str) -> None:
        """
        Ensures failed images are never silently discarded.
        """
        record = FailedItemRecord(
            image_path=image_path, identity_id=identity_id, failure_reason=reason
        )
        self.report.failed_items.append(record)
        self.report.total_failed_images += 1
        logger.warning(f"Discarded item recorded: [{identity_id}] {image_path} -> Reason: {reason}")

    def save_report(self, output_dir: str) -> str:
        """
        Saves preprocessing summary report and failed manifest into output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "dataset_report.json")

        report_dict = self.report.model_dump()
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Dataset Preprocessing Report saved to '{report_path}'")
        return report_path
