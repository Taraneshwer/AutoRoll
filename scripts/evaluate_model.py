"""
AutoRoll Model Evaluation & Threshold Calibration Command CLI.
Usage: python scripts/evaluate_model.py [--dataset data/processed_datasets/sample_subset]
"""

import argparse
import os
import sys

from autoroll.common.logger import get_logger
from autoroll.ml.evaluation.metrics import VerificationMetricsCalculator
from autoroll.ml.evaluation.report import EvaluationReportBuilder
from autoroll.ml.evaluation.threshold import ThresholdCalibrator
from autoroll.ml.evaluation.verification import VerificationEvaluator
from autoroll.ml.recognition.arcface_iresnet import ArcFaceRecognizer

logger = get_logger("evaluate_model")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoRoll Face Verification Evaluation")
    parser.add_argument(
        "--dataset",
        default="./data/processed_datasets/sample_subset",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--out-dir",
        default="./reports/eval_results",
        help="Directory to save evaluation report artifacts",
    )
    parser.add_argument(
        "--threshold-config",
        default="./configs/calibrated_threshold.yaml",
        help="Output path for versioned calibrated threshold configuration",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    val_dir = os.path.join(args.dataset, "val")
    test_dir = os.path.join(args.dataset, "test")

    if not os.path.exists(val_dir):
        logger.error(
            f"Validation dataset directory '{val_dir}' not found. "
            "Run 'python scripts/prepare_dataset.py --synthetic' first."
        )
        sys.exit(1)

    logger.info("Initializing Pretrained Baseline Recognizer...")
    baseline_rec = ArcFaceRecognizer(device="auto")
    evaluator = VerificationEvaluator(recognizer=baseline_rec, max_pairs=100)

    # Step 1: Calibrate Threshold on Validation Set
    logger.info(f"Generating verification pairs from Validation Set '{val_dir}'...")
    val_pairs = evaluator.generate_pairs(val_dir)
    if not val_pairs:
        logger.error("Failed to generate validation face pairs.")
        sys.exit(1)

    val_gen, val_imp, val_lat = evaluator.evaluate_pairs(val_pairs)

    calibrator = ThresholdCalibrator(criterion="eer")
    calibrated = calibrator.calibrate(val_gen, val_imp, model_version=baseline_rec.model_version)
    calibrator.save_calibration_yaml(calibrated, args.threshold_config)

    # Step 2: Evaluate Model on Unseen Test Identities using Calibrated Threshold
    logger.info(f"Evaluating Model on Unseen Test Identities in '{test_dir}'...")
    test_pairs = evaluator.generate_pairs(test_dir) if os.path.exists(test_dir) else val_pairs
    test_gen, test_imp, test_lat = evaluator.evaluate_pairs(test_pairs)

    test_metrics = VerificationMetricsCalculator.compute_metrics(
        genuine_scores=test_gen,
        impostor_scores=test_imp,
        threshold=calibrated.threshold,
        model_version=baseline_rec.model_version,
        avg_latency_ms=test_lat,
    )

    # Step 3: Export Evaluation Reports
    builder = EvaluationReportBuilder(experiment_id="eval_phase5")
    paths = builder.export_report(
        output_dir=args.out_dir,
        pretrained_metrics=test_metrics,
        finetuned_metrics=test_metrics,  # Evaluated baseline
        calibrated_threshold=calibrated.threshold,
    )

    print("\n" + "=" * 65)
    print("         AUTOROLL OPEN-SET FACE VERIFICATION EVALUATION        ")
    print("=" * 65)
    print(f"Calibrated Threshold   : {calibrated.threshold} (Criterion: {calibrated.criterion})")
    print(
        f"Equal Error Rate (EER) : {test_metrics.eer:.4f} "
        f"(at threshold {test_metrics.eer_threshold:.4f})"
    )
    print(f"Verification Accuracy : {test_metrics.accuracy * 100.0:.2f}%")
    print(
        f"Precision / Recall / F1: {test_metrics.precision:.4f} / "
        f"{test_metrics.recall:.4f} / {test_metrics.f1_score:.4f}"
    )
    print(f"FAR / FRR              : {test_metrics.far:.4f} / {test_metrics.frr:.4f}")
    print(f"Average Latency        : {test_metrics.avg_latency_ms:.2f} ms")
    print("-" * 65)
    print(f"Threshold Config Saved : {args.threshold_config}")
    print(f"Markdown Report Saved  : {paths['md']}")
    print(f"JSON Report Saved      : {paths['json']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
