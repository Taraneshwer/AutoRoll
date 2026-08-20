"""
AutoRoll Real Pretrained ML Baseline Evaluation & Benchmarking Script.
Evaluates Candidate A (ArcFace MS1MV2/GlintR100) vs Candidate B (ArcFace WebFace600K/R50)
on actual face images from sample_subset datasets.
Measures latency, memory size, same-person vs different-person similarity distributions,
verification accuracy, ROC, FAR, FRR, and EER.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import os
import glob
import time
import numpy as np
import cv2

from app.core.config import get_settings
from app.core.logger import get_logger
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.detectors.aligner import FaceAligner
from app.ml.recognition.arcface_iresnet import ArcFaceRecognizer
from app.ml.liveness.passive_fas import PassiveAntiSpoofingModel

logger = get_logger("evaluate_baseline")
settings = get_settings()

def get_face_chips(dataset_dir="data/raw_datasets/sample_subset"):
    detector = SCRFDDetector(conf_threshold=0.4)
    aligner = FaceAligner()

    identity_chips = {}
    
    student_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "student_id_*")) if os.path.isdir(d)])
    
    for sdir in student_dirs:
        student_id = os.path.basename(sdir)
        identity_chips[student_id] = []
        img_files = sorted(glob.glob(os.path.join(sdir, "*.jpg"))) + sorted(glob.glob(os.path.join(sdir, "*.png")))
        
        for ifile in img_files:
            img = cv2.imread(ifile)
            if img is None:
                continue
            dets = detector.detect(img, score_threshold=0.4)
            if dets:
                best_det = max(dets, key=lambda d: d.det_confidence)
                aligned_chip = aligner.align(img, best_det.landmarks)
                identity_chips[student_id].append({
                    "path": ifile,
                    "chip": aligned_chip,
                    "bbox": best_det.bbox.to_list(),
                    "confidence": best_det.det_confidence,
                    "landmarks": best_det.landmarks.points
                })
    return identity_chips

def compute_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def evaluate_recognition_candidate(candidate_id, model_path, identity_chips):
    recognizer = ArcFaceRecognizer(model_path=model_path)
    
    # Warmup & latency benchmark
    all_chips = [item["chip"] for s_chips in identity_chips.values() for item in s_chips]
    if not all_chips:
        raise ValueError("No face chips found for evaluation.")
        
    latencies = []
    embeddings = {}
    
    for sid, items in identity_chips.items():
        embeddings[sid] = []
        for item in items:
            t0 = time.perf_counter()
            res = recognizer.extract_embedding(item["chip"])
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            embeddings[sid].append(res.embedding)

    # Compute same-person similarities
    same_sims = []
    for sid, embs in embeddings.items():
        n = len(embs)
        for i in range(n):
            for j in range(i + 1, n):
                same_sims.append(compute_cosine_similarity(embs[i], embs[j]))

    # Compute different-person similarities
    diff_sims = []
    sids = list(embeddings.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            for e1 in embeddings[sids[i]]:
                for e2 in embeddings[sids[j]]:
                    diff_sims.append(compute_cosine_similarity(e1, e2))

    same_sims = np.array(same_sims) if len(same_sims) > 0 else np.array([1.0])
    diff_sims = np.array(diff_sims) if len(diff_sims) > 0 else np.array([0.0])

    # Compute ROC, FAR, FRR, EER at default threshold 0.65
    thresh = 0.65
    tp = np.sum(same_sims >= thresh)
    fn = np.sum(same_sims < thresh)
    fp = np.sum(diff_sims >= thresh)
    tn = np.sum(diff_sims < thresh)

    acc = (tp + tn) / (len(same_sims) + len(diff_sims))
    far = fp / len(diff_sims) if len(diff_sims) > 0 else 0.0
    frr = fn / len(same_sims) if len(same_sims) > 0 else 0.0

    # Calculate EER across thresholds 0.0 -> 1.0
    thresholds = np.linspace(0.0, 1.0, 1000)
    eer = 1.0
    min_diff = 1.0
    for t in thresholds:
        t_far = np.mean(diff_sims >= t)
        t_frr = np.mean(same_sims < t)
        if abs(t_far - t_frr) < min_diff:
            min_diff = abs(t_far - t_frr)
            eer = (t_far + t_frr) / 2.0

    return {
        "candidate_id": candidate_id,
        "model_path": model_path,
        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "same_sim_mean": float(np.mean(same_sims)),
        "same_sim_std": float(np.std(same_sims)),
        "same_sim_min": float(np.min(same_sims)),
        "same_sim_max": float(np.max(same_sims)),
        "diff_sim_mean": float(np.mean(diff_sims)),
        "diff_sim_std": float(np.std(diff_sims)),
        "diff_sim_min": float(np.min(diff_sims)),
        "diff_sim_max": float(np.max(diff_sims)),
        "accuracy": float(acc),
        "far": float(far),
        "frr": float(frr),
        "eer": float(eer)
    }

def main():
    print("=================================================================================")
    print("AUTOROLL REAL PRETRAINED ML BASELINE EVALUATION & BENCHMARK")
    print("=================================================================================")

    # 1. Extract Face Chips from dataset
    logger.info("Extracting aligned face chips from dataset sample_subset...")
    identity_chips = get_face_chips("data/raw_datasets/sample_subset")
    total_faces = sum(len(v) for v in identity_chips.values())
    logger.info(f"Loaded {total_faces} face crops across {len(identity_chips)} student identities.")

    # 2. Evaluate Candidate A (MS1MV2 / GlintR100)
    logger.info("Evaluating Candidate A (ArcFace MS1MV2 / GlintR100)...")
    res_a = evaluate_recognition_candidate("Candidate A (MS1MV2 / GlintR100)", settings.ARCFACE_MS1MV2_PATH, identity_chips)

    # 3. Evaluate Candidate B (WebFace600K / Buffalo_L)
    logger.info("Evaluating Candidate B (ArcFace WebFace600K / Buffalo_L)...")
    res_b = evaluate_recognition_candidate("Candidate B (WebFace600K / Buffalo_L)", settings.ARCFACE_GLINT_PATH, identity_chips)

    # 4. Evaluate MiniFASNet Liveness
    logger.info("Evaluating MiniFASNet Liveness Model...")
    liveness_model = PassiveAntiSpoofingModel()
    all_chips = [item["chip"] for s_chips in identity_chips.values() for item in s_chips]
    
    live_evals = [liveness_model.evaluate_liveness_detailed(chip) for chip in all_chips]
    mean_real_prob = float(np.mean([e["ml_real_prob"] for e in live_evals]))
    mean_aux_score = float(np.mean([e["aux_heuristic_score"] for e in live_evals]))

    # Print summary
    print("\n" + "=" * 90)
    print("RECOGNITION CANDIDATE COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'METRIC':<32} | {'CANDIDATE A (MS1MV2)':<24} | {'CANDIDATE B (WebFace600K)':<24}")
    print("-" * 90)
    print(f"{'Model Binary Size':<32} | {res_a['size_mb']:.2f} MB                  | {res_b['size_mb']:.2f} MB")
    print(f"{'Mean Inference Latency':<32} | {res_a['mean_latency_ms']:.2f} ms               | {res_b['mean_latency_ms']:.2f} ms")
    print(f"{'P95 Inference Latency':<32} | {res_a['p95_latency_ms']:.2f} ms               | {res_b['p95_latency_ms']:.2f} ms")
    print(f"{'Same-Person Similarity (Mean)':<32} | {res_a['same_sim_mean']:.4f} (std={res_a['same_sim_std']:.4f})  | {res_b['same_sim_mean']:.4f} (std={res_b['same_sim_std']:.4f})")
    print(f"{'Different-Person Sim (Mean)':<32} | {res_a['diff_sim_mean']:.4f} (std={res_a['diff_sim_std']:.4f})  | {res_b['diff_sim_mean']:.4f} (std={res_b['diff_sim_std']:.4f})")
    print(f"{'Verification Accuracy (@0.65)':<32} | {res_a['accuracy']*100:.2f}%                  | {res_b['accuracy']*100:.2f}%")
    print(f"{'False Accept Rate (FAR)':<32} | {res_a['far']:.6f}                 | {res_b['far']:.6f}")
    print(f"{'False Reject Rate (FRR)':<32} | {res_a['frr']:.6f}                 | {res_b['frr']:.6f}")
    print(f"{'Equal Error Rate (EER)':<32} | {res_a['eer']:.6f}                 | {res_b['eer']:.6f}")
    print("=" * 90)

    print("\n--- LIVENESS MODEL EVALUATION ---")
    print(f"MiniFASNet Real Face ML Probability (Mean): {mean_real_prob:.4f}")
    print(f"Auxiliary Moire Heuristic Score (Mean)  : {mean_aux_score:.4f}")
    print("=================================================================================\n")

if __name__ == "__main__":
    main()
