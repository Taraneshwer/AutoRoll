"""
AutoRoll Dataset Preparation Pipeline Script.
Usage: python scripts/prepare_dataset.py [--config configs/dataset_config.yaml] [--synthetic]
"""

import argparse
import os
import sys

from autoroll.common.logger import get_logger
from autoroll.ml.preprocessing.aligner import DatasetFaceAligner
from autoroll.ml.preprocessing.dataset_loader import (
    DatasetConfig,
    DirectoryDatasetLoader,
    SyntheticDatasetLoader,
)
from autoroll.ml.preprocessing.detector import DatasetFaceDetector
from autoroll.ml.preprocessing.metadata import MetadataManager
from autoroll.ml.preprocessing.quality import FaceQualityFilter
from autoroll.ml.preprocessing.splitter import IdentityDisjointSplitter

logger = get_logger("prepare_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Dataset Preparation Pipeline")
    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to YAML dataset configuration file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and process synthetic test dataset",
    )
    return parser.parse_args()


def run_pipeline(config: DatasetConfig, use_synthetic: bool = False):
    logger.info(f"Starting Preprocessing Pipeline for dataset '{config.dataset_name}'...")

    # Step 1: Load raw dataset
    if use_synthetic:
        logger.info("Using SyntheticDatasetLoader for pipeline run...")
        loader = SyntheticDatasetLoader(
            target_dir=config.raw_data_dir, num_identities=10, images_per_id=4
        )
    else:
        loader = DirectoryDatasetLoader(raw_data_dir=config.raw_data_dir)

    raw_identity_map = loader.load_dataset()
    if not raw_identity_map:
        logger.error(
            f"No raw dataset images found in '{config.raw_data_dir}'. "
            "Pass --synthetic to generate sample dataset."
        )
        sys.exit(1)

    total_raw_count = sum(len(records) for records in raw_identity_map.values())

    # Step 2: Perform Identity-Disjoint Splitting FIRST
    splitter = IdentityDisjointSplitter(
        train_ratio=config.split_ratios.get("train", 0.8),
        val_ratio=config.split_ratios.get("val", 0.1),
        test_ratio=config.split_ratios.get("test", 0.1),
    )
    split_identities = splitter.split_identities(raw_identity_map)

    # Step 3: Initialize Detector, Quality Filter, Aligner, & Metadata Manager
    detector = DatasetFaceDetector(min_confidence=config.min_detection_confidence)
    quality_filter = FaceQualityFilter(
        min_face_size=config.min_face_size,
        min_blur_score=config.min_blur_score,
        min_confidence=config.min_detection_confidence,
    )
    aligner = DatasetFaceAligner(target_size=(112, 112), resumable=config.resumable)
    meta_mgr = MetadataManager(
        dataset_name=config.dataset_name,
        dataset_version=config.dataset_version,
        preprocessing_version=config.preprocessing_version,
    )

    meta_mgr.report.total_raw_images = total_raw_count
    meta_mgr.report.total_identities = len(raw_identity_map)

    processed_count = 0
    splits_summary = {}

    # Step 4: Process images split by split
    for split_name, id_map in split_identities.items():
        split_img_count = 0
        split_id_count = len(id_map)

        for identity_id, records in id_map.items():
            for record in records:
                img_arr, det_result, err = detector.detect_face(record.full_path)
                if err is not None:
                    meta_mgr.record_failure(
                        image_path=record.full_path,
                        identity_id=identity_id,
                        reason=err,
                    )
                    continue

                q_res = quality_filter.evaluate(img_arr, det_result)
                if not q_res.passed:
                    meta_mgr.record_failure(
                        image_path=record.full_path,
                        identity_id=identity_id,
                        reason=f"quality_failed: {q_res.reason}",
                    )
                    continue

                # Align and save image
                aligner.process_and_save(
                    image=img_arr,
                    detection=det_result,
                    output_dir=config.output_dir,
                    split_name=split_name,
                    identity_id=identity_id,
                    image_name=record.image_name,
                )

                processed_count += 1
                split_img_count += 1

        splits_summary[split_name] = {
            "identities": split_id_count,
            "processed_images": split_img_count,
        }

    meta_mgr.report.total_processed_images = processed_count
    meta_mgr.report.splits_summary = splits_summary

    # Save detailed report JSON
    report_file = meta_mgr.save_report(config.output_dir)

    print("\n" + "=" * 60)
    print("        AUTOROLL PREPROCESSING PIPELINE SUMMARY        ")
    print("=" * 60)
    print(f"Dataset Name       : {config.dataset_name}")
    print(f"Total Raw Images   : {total_raw_count}")
    print(f"Total Identities   : {len(raw_identity_map)}")
    print(f"Processed Images   : {processed_count}")
    print(f"Failed/Rejected    : {meta_mgr.report.total_failed_images}")
    print(f"Output Directory   : {config.output_dir}")
    print(f"Report File        : {report_file}")
    print("-" * 60)
    print("Splits Distribution:")
    for s_name, s_info in splits_summary.items():
        print(
            f"  [{s_name.upper():5s}] Identities: {s_info['identities']:3d} | "
            f"Images: {s_info['processed_images']:4d}"
        )
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    if os.path.exists(args.config):
        config = DatasetConfig.from_yaml(args.config)
    else:
        logger.warning(f"Config '{args.config}' not found. Using default dataset config.")
        config = DatasetConfig(
            dataset_name="autoroll_default",
            raw_data_dir="./data/raw_datasets/sample_subset",
            output_dir="./data/processed_datasets/sample_subset",
        )
    run_pipeline(config, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
