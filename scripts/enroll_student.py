"""
AutoRoll Privacy-Preserving Student Enrollment CLI Tool.
Usage: python scripts/enroll_student.py --student-code STU1001 --full-name "Jane Doe"
       --images path1.jpg path2.jpg [--delete-temp]
"""

import argparse

from autoroll.common.logger import get_logger
from autoroll.ml.enrollment.pipeline import PrivacyPreservingEnrollmentPipeline

logger = get_logger("enroll_student_cli")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Privacy-Preserving Face Enrollment CLI")
    parser.add_argument("--student-code", required=True, help="Unique student identification code")
    parser.add_argument("--full-name", required=True, help="Student full name")
    parser.add_argument(
        "--images", nargs="+", required=True, help="List of sample image paths for enrollment"
    )
    parser.add_argument(
        "--delete-temp",
        action="store_true",
        default=False,
        help="Delete temporary raw sample images from disk after enrollment",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device selection"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = PrivacyPreservingEnrollmentPipeline(device=args.device)

    result = pipeline.enroll(
        student_code=args.student_code,
        full_name=args.full_name,
        sample_inputs=args.images,
        delete_raw_images=args.delete_temp,
    )

    print("\n" + "=" * 65)
    print("         AUTOROLL PRIVACY-PRESERVING ENROLLMENT RESULT        ")
    print("=" * 65)
    print(f"Student Code     : {result.student_code}")
    print(f"Full Name        : {result.full_name}")
    print(f"Status           : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Samples Processed: {result.samples_processed}")
    print(f"Samples Accepted : {result.samples_accepted}")
    print(f"Model Version    : {result.model_version}")

    if result.success and result.aggregated_embedding:
        print(f"Embedding Dim    : {len(result.aggregated_embedding)}")
        print(f"Embedding Prefix : {result.aggregated_embedding[:5]}...")

    if result.rejection_reasons:
        print("\nRejection Reasons:")
        for r in result.rejection_reasons:
            print(f"   |- {r}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
