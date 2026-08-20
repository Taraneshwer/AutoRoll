"""
AutoRoll Dataset Statistics & Quality Report Script.
Usage: python scripts/dataset_stats.py [--dir data/processed_datasets/sample_subset]
"""

import argparse
import json
import os

from autoroll.common.logger import get_logger

logger = get_logger("dataset_stats")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Dataset Statistics Utility")
    parser.add_argument(
        "--dir",
        default="./data/processed_datasets/sample_subset",
        help="Path to processed dataset directory containing dataset_report.json",
    )
    return parser.parse_args()


def display_stats(processed_dir: str):
    report_file = os.path.join(processed_dir, "dataset_report.json")
    if not os.path.exists(report_file):
        logger.error(f"Report file '{report_file}' not found.")
        return

    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 65)
    print("                AUTOROLL DATASET STATISTICS                ")
    print("=" * 65)
    print(f"Dataset Name          : {data.get('dataset_name')}")
    print(f"Dataset Version       : {data.get('dataset_version')}")
    print(f"Preprocessing Version : {data.get('preprocessing_version')}")
    print(f"Total Raw Images      : {data.get('total_raw_images')}")
    print(f"Total Processed Images: {data.get('total_processed_images')}")
    print(f"Total Discarded/Failed: {data.get('total_failed_images')}")
    print(f"Total Identities      : {data.get('total_identities')}")
    print("-" * 65)
    print("Splits Distribution:")
    splits = data.get("splits_summary", {})
    for split_name, info in splits.items():
        print(
            f"  - {split_name.upper():5s} : {info.get('identities', 0):3d} Identities | "
            f"{info.get('processed_images', 0):4d} Aligned Images"
        )

    failed_items = data.get("failed_items", [])
    if failed_items:
        print("-" * 65)
        top_count = min(5, len(failed_items))
        print(f"Rejection Manifest Sample (Showing top {top_count} of {len(failed_items)}):")
        for item in failed_items[:5]:
            print(f"  - [{item.get('identity_id')}] Reason: {item.get('failure_reason')}")
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    display_stats(args.dir)


if __name__ == "__main__":
    main()
