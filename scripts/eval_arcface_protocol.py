"""
AutoRoll Fixed Evaluation Protocol for ArcFace Model Verification.
Evaluates baseline pretrained model or fine-tuned checkpoints on identity-disjoint Validation and Test splits.
Selects optimal threshold on Validation split and evaluates Test split strictly at that Validation threshold.

Calculates:
  - Genuine & Impostor cosine similarity distributions
  - Global & per-dimension embedding variance
  - ROC-AUC, EER, FAR, FRR, TAR at FAR=1e-3 & 1e-4
  - Validation-selected threshold and Test metrics
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoroll.common.logger import get_logger
from autoroll.ml.recognition.iresnet_torch import iresnet50
from autoroll.ml.utils import get_execution_device

logger = get_logger("eval_arcface_protocol")


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def generate_pairs(split_dir: str, num_genuine: int = 3000, num_impostor: int = 3000, seed: int = 42) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generates deterministic genuine (same identity) and impostor (different identity) image path pairs.
    """
    np.random.seed(seed)
    id_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    images_by_id = {}
    for id_folder in id_folders:
        id_path = os.path.join(split_dir, id_folder)
        imgs = [os.path.join(id_path, f) for f in os.listdir(id_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(imgs) >= 2:
            images_by_id[id_folder] = imgs

    valid_ids = sorted(list(images_by_id.keys()))
    if len(valid_ids) < 2:
        raise ValueError(f"Insufficient identities with >=2 images in {split_dir}")

    # Generate genuine pairs
    genuine_pairs = []
    seen_gen = set()
    attempts = 0
    while len(genuine_pairs) < num_genuine and attempts < num_genuine * 10:
        attempts += 1
        chosen_id = np.random.choice(valid_ids)
        imgs = images_by_id[chosen_id]
        idx1, idx2 = np.random.choice(len(imgs), size=2, replace=False)
        pair = (imgs[idx1], imgs[idx2])
        if pair not in seen_gen:
            seen_gen.add(pair)
            genuine_pairs.append(pair)

    # Generate impostor pairs
    impostor_pairs = []
    seen_imp = set()
    attempts = 0
    while len(impostor_pairs) < num_impostor and attempts < num_impostor * 10:
        attempts += 1
        id1, id2 = np.random.choice(valid_ids, size=2, replace=False)
        img1 = np.random.choice(images_by_id[id1])
        img2 = np.random.choice(images_by_id[id2])
        pair = (img1, img2)
        if pair not in seen_imp:
            seen_imp.add(pair)
            impostor_pairs.append(pair)

    return genuine_pairs[:num_genuine], impostor_pairs[:num_impostor]


from torch.utils.data import Dataset, DataLoader

class ImageListDataset(Dataset):
    def __init__(self, img_paths: List[str]):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = cv2.imread(p)
        if img is None:
            return p, torch.zeros(3, 112, 112, dtype=torch.float32), False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 127.5
        blob_nchw = torch.from_numpy(np.transpose(blob, (2, 0, 1))).float()
        return p, blob_nchw, True


class ONNXEmbedder:
    def __init__(self, model_path: str, batch_size: int = 128, num_workers: int = 0):
        device, providers = get_execution_device("auto")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Suppress warnings
        self.sess = ort.InferenceSession(model_path, opts, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract(self, img_path: str) -> np.ndarray:
        return self.extract_batch([img_path])[img_path]

    def extract_batch(self, img_paths: List[str]) -> Dict[str, np.ndarray]:
        ds = ImageListDataset(img_paths)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        results = {}
        for paths, blobs, valids in dl:
            valid_idx = [i for i, v in enumerate(valids) if v]
            for i, v in enumerate(valids):
                if not v:
                    results[paths[i]] = np.zeros(512, dtype=np.float32)

            if not valid_idx:
                continue

            valid_blobs = blobs[valid_idx].numpy()
            outs = self.sess.run(None, {self.in_name: valid_blobs})[0]
            valid_paths = [paths[i] for i in valid_idx]
            for p, out in zip(valid_paths, outs):
                norm = np.linalg.norm(out)
                results[p] = (out / norm if norm > 0 else out).astype(np.float32)

        return results


class PyTorchEmbedder:
    def __init__(self, backbone: torch.nn.Module, device: torch.device, batch_size: int = 128, num_workers: int = 0):
        self.model = backbone
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model.eval()

    def extract(self, img_path: str) -> np.ndarray:
        return self.extract_batch([img_path])[img_path]

    def extract_batch(self, img_paths: List[str]) -> Dict[str, np.ndarray]:
        ds = ImageListDataset(img_paths)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        results = {}
        with torch.no_grad():
            for paths, blobs, valids in dl:
                valid_idx = [i for i, v in enumerate(valids) if v]
                for i, v in enumerate(valids):
                    if not v:
                        results[paths[i]] = np.zeros(512, dtype=np.float32)

                if not valid_idx:
                    continue

                valid_blobs = blobs[valid_idx].to(self.device)
                outs = self.model(valid_blobs).cpu().numpy()
                valid_paths = [paths[i] for i in valid_idx]
                for p, out in zip(valid_paths, outs):
                    norm = np.linalg.norm(out)
                    results[p] = (out / norm if norm > 0 else out).astype(np.float32)

        return results


def evaluate_split(
    embedder,
    split_dir: str,
    split_name: str,
    num_pairs: int = 1000,
    seed: int = 42,
) -> Tuple[dict, Dict[str, np.ndarray]]:
    """
    Evaluates embedder on genuine and impostor pairs for a given split.
    """
    logger.info(f"Evaluating split '{split_name}' at '{split_dir}' ({num_pairs} genuine / {num_pairs} impostor pairs)...")
    t0 = time.time()
    
    genuine_pairs, impostor_pairs = generate_pairs(split_dir, num_genuine=num_pairs, num_impostor=num_pairs, seed=seed)

    # Collect unique image paths for batch extraction
    unique_paths = list(set([p for pair in genuine_pairs + impostor_pairs for p in pair]))
    logger.info(f"Extracting embeddings for {len(unique_paths)} unique images in '{split_name}'...")
    
    emb_cache = embedder.extract_batch(unique_paths)

    elapsed_extract = time.time() - t0
    logger.info(f"Extraction complete in {elapsed_extract:.2f}s ({len(unique_paths)/elapsed_extract:.1f} img/s).")

    # Compute pairwise similarities
    gen_sims = np.array([cos_sim(emb_cache[p1], emb_cache[p2]) for p1, p2 in genuine_pairs], dtype=np.float32)
    imp_sims = np.array([cos_sim(emb_cache[p1], emb_cache[p2]) for p1, p2 in impostor_pairs], dtype=np.float32)

    # Embedding variance statistics
    all_embs = np.array(list(emb_cache.values()), dtype=np.float32)
    global_var = float(np.var(all_embs))
    per_dim_var = np.var(all_embs, axis=0)

    # Calculate EER and optimal threshold
    thresholds = np.linspace(-1.0, 1.0, 2001)
    eers = []
    fars = []
    frrs = []
    accs = []

    for th in thresholds:
        far = float(np.mean(imp_sims >= th))
        frr = float(np.mean(gen_sims < th))
        acc = float(np.mean(np.concatenate([gen_sims >= th, imp_sims < th])))
        fars.append(far)
        frrs.append(frr)
        accs.append(acc)
        eers.append(abs(far - frr))

    best_idx = int(np.argmin(eers))
    optimal_th = float(thresholds[best_idx])
    eer_val = float((fars[best_idx] + frrs[best_idx]) / 2.0)
    best_acc = float(accs[best_idx])

    # TAR at FAR = 1e-3 and 1e-4
    far_1e3_idx = np.argmin(np.abs(np.array(fars) - 1e-3))
    far_1e4_idx = np.argmin(np.abs(np.array(fars) - 1e-4))
    tar_at_1e3 = float(1.0 - frrs[far_1e3_idx])
    tar_at_1e4 = float(1.0 - frrs[far_1e4_idx])

    # ROC AUC (trapezoidal integration)
    sorted_idx = np.argsort(fars)
    auc_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(auc_func(1.0 - np.array(frrs)[sorted_idx], np.array(fars)[sorted_idx]))

    metrics = {
        "split_name": split_name,
        "num_genuine_pairs": len(gen_sims),
        "num_impostor_pairs": len(imp_sims),
        "genuine_sim_mean": float(np.mean(gen_sims)),
        "genuine_sim_std": float(np.std(gen_sims)),
        "genuine_sim_min": float(np.min(gen_sims)),
        "genuine_sim_max": float(np.max(gen_sims)),
        "impostor_sim_mean": float(np.mean(imp_sims)),
        "impostor_sim_std": float(np.std(imp_sims)),
        "impostor_sim_min": float(np.min(imp_sims)),
        "impostor_sim_max": float(np.max(imp_sims)),
        "global_embedding_variance": global_var,
        "min_per_dim_var": float(np.min(per_dim_var)),
        "max_per_dim_var": float(np.max(per_dim_var)),
        "mean_per_dim_var": float(np.mean(per_dim_var)),
        "optimal_threshold": optimal_th,
        "eer": eer_val,
        "best_accuracy_pct": best_acc * 100.0,
        "tar_at_far_1e3": tar_at_1e3,
        "tar_at_far_1e4": tar_at_1e4,
        "roc_auc": abs(auc),
    }

    raw_data = {
        "gen_sims": gen_sims,
        "imp_sims": imp_sims,
        "thresholds": thresholds,
        "fars": np.array(fars),
        "frrs": np.array(frrs),
    }

    return metrics, raw_data


def evaluate_protocol(
    embedder,
    val_dir: str = "data/face_recognition/splits/val",
    test_dir: str = "data/face_recognition/splits/test",
    num_pairs: int = 3000,
    seed: int = 42,
) -> dict:
    """
    Evaluates Validation split to pick threshold, then evaluates Test split at that Validation threshold.
    """
    val_metrics, val_raw = evaluate_split(embedder, val_dir, "validation", num_pairs=num_pairs, seed=seed)
    test_metrics, test_raw = evaluate_split(embedder, test_dir, "test", num_pairs=num_pairs, seed=seed + 1)

    val_threshold = val_metrics["optimal_threshold"]

    # Evaluate Test split strictly at Validation-selected threshold
    test_gen_sims = test_raw["gen_sims"]
    test_imp_sims = test_raw["imp_sims"]

    test_far_at_val_th = float(np.mean(test_imp_sims >= val_threshold))
    test_frr_at_val_th = float(np.mean(test_gen_sims < val_threshold))
    test_acc_at_val_th = float(np.mean(np.concatenate([test_gen_sims >= val_threshold, test_imp_sims < val_threshold]))) * 100.0
    test_tar_at_val_th = float(1.0 - test_frr_at_val_th)

    protocol_report = {
        "validation": val_metrics,
        "test_self_tuned": test_metrics,
        "test_at_val_threshold": {
            "validation_selected_threshold": val_threshold,
            "test_accuracy_pct": test_acc_at_val_th,
            "test_far": test_far_at_val_th,
            "test_frr": test_frr_at_val_th,
            "test_tar": test_tar_at_val_th,
        },
    }

    return protocol_report


def main():
    parser = argparse.ArgumentParser(description="AutoRoll ArcFace Evaluation Protocol")
    parser.add_argument("--model-path", type=str, default="models/pretrained/arcface_r50_webface_or_glint/model.onnx", help="Path to ONNX or PyTorch model")
    parser.add_argument("--is-pytorch", action="store_true", help="Flag if model is PyTorch .pt checkpoint")
    parser.add_argument("--val-dir", type=str, default="data/face_recognition/splits/val", help="Validation split path")
    parser.add_argument("--test-dir", type=str, default="data/face_recognition/splits/test", help="Test split path")
    parser.add_argument("--num-pairs", type=int, default=3000, help="Number of genuine/impostor pairs per split")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON report path")

    args = parser.parse_args()

    if args.is_pytorch:
        device = torch.device("cpu")
        logger.info(f"Loading PyTorch backbone onto '{device}'...")
        if args.model_path.endswith(".onnx"):
            from autoroll.ml.recognition.iresnet_torch import get_iresnet50
            backbone = get_iresnet50(args.model_path).to(device)
        else:
            from autoroll.ml.recognition.iresnet_torch import get_iresnet50
            backbone = get_iresnet50().to(device)
            state = torch.load(args.model_path, map_location=device)
            backbone.load_state_dict(state.get("backbone_state", state))
        embedder = PyTorchEmbedder(backbone, device)
    else:
        logger.info(f"Loading ONNX model from '{args.model_path}'...")
        embedder = ONNXEmbedder(args.model_path)

    report = evaluate_protocol(embedder, val_dir=args.val_dir, test_dir=args.test_dir, num_pairs=args.num_pairs)

    print("\n" + "=" * 80)
    print("                    AUTOROLL ARCFACE EVALUATION PROTOCOL                    ")
    print("=" * 80)
    print(f"Validation Selected Threshold : {report['test_at_val_threshold']['validation_selected_threshold']:.4f}")
    print(f"Validation EER                : {report['validation']['eer']*100:.2f}% (AUC: {report['validation']['roc_auc']:.4f})")
    print(f"Validation Genuine Sim Mean   : {report['validation']['genuine_sim_mean']:.4f} ± {report['validation']['genuine_sim_std']:.4f}")
    print(f"Validation Impostor Sim Mean  : {report['validation']['impostor_sim_mean']:.4f} ± {report['validation']['impostor_sim_std']:.4f}")
    print(f"Validation Global Variance    : {report['validation']['global_embedding_variance']:.6f}")
    print("-" * 80)
    print(f"Test Accuracy @ Val Threshold : {report['test_at_val_threshold']['test_accuracy_pct']:.2f}%")
    print(f"Test FAR @ Val Threshold      : {report['test_at_val_threshold']['test_far']*100:.4f}%")
    print(f"Test FRR @ Val Threshold      : {report['test_at_val_threshold']['test_frr']*100:.4f}%")
    print(f"Test TAR @ Val Threshold      : {report['test_at_val_threshold']['test_tar']*100:.4f}%")
    print("=" * 80 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved evaluation protocol report to '{args.output_json}'")


if __name__ == "__main__":
    main()
