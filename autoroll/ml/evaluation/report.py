"""
Evaluation Report Builder and Exporter.
Generates Markdown and JSON evaluation reports.
"""

import json
import os

from autoroll.common.logger import get_logger
from autoroll.ml.evaluation.metrics import VerificationMetrics

logger = get_logger("evaluation_report")


class EvaluationReportBuilder:
    """
    Builds comprehensive face verification report comparing baseline vs fine-tuned models.
    """

    def __init__(self, experiment_id: str = "eval_exp"):
        self.experiment_id = experiment_id

    def generate_markdown_report(
        self,
        pretrained_metrics: VerificationMetrics | None,
        finetuned_metrics: VerificationMetrics | None,
        calibrated_threshold: float,
    ) -> str:
        md = []
        md.append("# AutoRoll Face Verification Evaluation Report\n")
        md.append(f"**Experiment ID**: `{self.experiment_id}`  ")
        md.append(f"**Calibrated Operating Threshold**: `{calibrated_threshold}`\n")
        md.append("---\n")
        md.append("## Verification Metrics Summary\n")
        md.append("| Metric | Pretrained Model | Fine-Tuned Model | Improvement |")
        md.append("| :--- | :---: | :---: | :---: |")

        def fmt(val: float | None) -> str:
            return f"{val:.4f}" if val is not None else "N/A"

        p_acc = pretrained_metrics.accuracy if pretrained_metrics else None
        f_acc = finetuned_metrics.accuracy if finetuned_metrics else None
        acc_diff = (f_acc - p_acc) if (p_acc is not None and f_acc is not None) else 0.0

        p_eer = pretrained_metrics.eer if pretrained_metrics else None
        f_eer = finetuned_metrics.eer if finetuned_metrics else None
        eer_diff = (p_eer - f_eer) if (p_eer is not None and f_eer is not None) else 0.0

        p_f1 = pretrained_metrics.f1_score if pretrained_metrics else None
        f_f1 = finetuned_metrics.f1_score if finetuned_metrics else None
        f1_diff = (f_f1 - p_f1) if (p_f1 is not None and f_f1 is not None) else 0.0

        md.append(f"| Accuracy | {fmt(p_acc)} | {fmt(f_acc)} | {acc_diff:+.4f} |")
        md.append(f"| Equal Error Rate (EER) | {fmt(p_eer)} | {fmt(f_eer)} | {eer_diff:+.4f} |")
        md.append(f"| F1 Score | {fmt(p_f1)} | {fmt(f_f1)} | {f1_diff:+.4f} |")

        if finetuned_metrics:
            p_far = pretrained_metrics.far if pretrained_metrics else None
            p_frr = pretrained_metrics.frr if pretrained_metrics else None
            p_lat = pretrained_metrics.avg_latency_ms if pretrained_metrics else None
            far_val = fmt(finetuned_metrics.far)
            frr_val = fmt(finetuned_metrics.frr)
            md.append(f"| FAR (False Accept Rate) | {fmt(p_far)} | {far_val} | - |")
            md.append(f"| FRR (False Reject Rate) | {fmt(p_frr)} | {frr_val} | - |")
            md.append(
                f"| Avg Latency (ms) | {fmt(p_lat)} | {fmt(finetuned_metrics.avg_latency_ms)} | - |"
            )

            md.append("\n## TAR at Target FAR Values (Fine-Tuned Model)\n")
            md.append("| Target FAR | True Accept Rate (TAR) | Operating Threshold |")
            md.append("| :--- | :---: | :---: |")
            for item in finetuned_metrics.tar_at_far:
                md.append(
                    f"| FAR = {item.far_target:.4f} | TAR = {item.tar:.4f} | "
                    f"Threshold = {item.threshold:.4f} |"
                )

            md.append("\n## Cosine Similarity Score Distributions\n")
            md.append(
                f"- **Genuine Pairs**: Mean = `{finetuned_metrics.genuine_stats.mean:.4f}`, "
                f"Std = `{finetuned_metrics.genuine_stats.std:.4f}`"
            )
            md.append(
                f"- **Impostor Pairs**: Mean = `{finetuned_metrics.impostor_stats.mean:.4f}`, "
                f"Std = `{finetuned_metrics.impostor_stats.std:.4f}`"
            )

        md.append("\n---\n*Report generated automatically by AutoRoll Evaluation Pipeline.*\n")
        return "\n".join(md)

    def export_report(
        self,
        output_dir: str,
        pretrained_metrics: VerificationMetrics | None,
        finetuned_metrics: VerificationMetrics | None,
        calibrated_threshold: float,
    ) -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        md_content = self.generate_markdown_report(
            pretrained_metrics, finetuned_metrics, calibrated_threshold
        )
        md_path = os.path.join(output_dir, "evaluation_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        json_data = {
            "experiment_id": self.experiment_id,
            "calibrated_threshold": calibrated_threshold,
            "pretrained_metrics": pretrained_metrics.model_dump() if pretrained_metrics else None,
            "finetuned_metrics": finetuned_metrics.model_dump() if finetuned_metrics else None,
        }
        json_path = os.path.join(output_dir, "evaluation_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Evaluation report saved to '{md_path}' and '{json_path}'")
        return {"md": md_path, "json": json_path}
