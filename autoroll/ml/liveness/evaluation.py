"""
Presentation Attack Detection (PAD) Evaluation Module.
Calculates APCER, BPCER, ACER across threat categories (Printed, Photo Replay, Video Replay).
"""


import numpy as np
from pydantic import BaseModel, Field

from autoroll.common.logger import get_logger

logger = get_logger("pad_evaluation")


class AttackCategoryMetrics(BaseModel):
    attack_name: str
    total_samples: int
    failed_attacks: int  # Spoofs misclassified as real
    apcer: float  # APCER for this specific attack category


class PADEvaluationReport(BaseModel):
    total_real_samples: int
    total_spoof_samples: int
    bpcer: float  # Bona Fide Presentation Classification Error Rate
    apcer: float  # Attack Presentation Classification Error Rate
    acer: float  # Average Classification Error Rate = (APCER + BPCER) / 2
    category_breakdown: list[AttackCategoryMetrics] = Field(default_factory=list)
    avg_latency_ms: float


class PADEvaluator:
    """
    Evaluates Presentation Attack Detection (PAD) models against genuine faces
    and specific attack categories (printed_attack, photo_replay, video_replay).
    """

    @staticmethod
    def evaluate(
        real_scores: list[float],
        attack_scores_dict: dict[str, list[float]],
        threshold: float = 0.90,
        avg_latency_ms: float = 0.0,
    ) -> PADEvaluationReport:
        real_arr = np.array(real_scores, dtype=np.float32)
        n_real = len(real_arr)

        if n_real == 0:
            raise ValueError("Must provide at least 1 real face score for PAD evaluation.")

        # BPCER: Fraction of real faces incorrectly classified as spoof (score < threshold)
        real_rejections = int(np.sum(real_arr < threshold))
        bpcer = float(real_rejections / n_real)

        category_metrics: list[AttackCategoryMetrics] = []
        total_spoof_samples = 0
        total_spoof_accepts = 0

        for attack_name, scores in attack_scores_dict.items():
            if not scores:
                continue
            atk_arr = np.array(scores, dtype=np.float32)
            n_atk = len(atk_arr)
            # APCER per category: Fraction of spoofs incorrectly classified as real
            spoof_accepts = int(np.sum(atk_arr >= threshold))
            cat_apcer = float(spoof_accepts / n_atk)

            total_spoof_samples += n_atk
            total_spoof_accepts += spoof_accepts

            category_metrics.append(
                AttackCategoryMetrics(
                    attack_name=attack_name,
                    total_samples=n_atk,
                    failed_attacks=spoof_accepts,
                    apcer=round(cat_apcer, 4),
                )
            )

        overall_apcer = float(total_spoof_accepts / max(1, total_spoof_samples))
        acer = float((bpcer + overall_apcer) / 2.0)

        report = PADEvaluationReport(
            total_real_samples=n_real,
            total_spoof_samples=total_spoof_samples,
            bpcer=round(bpcer, 4),
            apcer=round(overall_apcer, 4),
            acer=round(acer, 4),
            category_breakdown=category_metrics,
            avg_latency_ms=round(avg_latency_ms, 2),
        )

        logger.info(
            f"PAD Evaluation Complete: BPCER={report.bpcer:.4f} | APCER={report.apcer:.4f} | "
            f"ACER={report.acer:.4f} | Avg Latency={report.avg_latency_ms}ms"
        )
        return report
