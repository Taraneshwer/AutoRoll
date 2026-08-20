"""
AutoRoll GPU Memory & Training Throughput Profiler.
Safely probes VRAM consumption across batch sizes (16, 32, 64, 128) using actual ArcFace model.
Reports peak VRAM, GPU utilization, batch throughput, and max stable batch size on NVIDIA RTX 5060.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


import sys
import time
import torch
import onnxruntime as ort

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger("profile_gpu_training")


def profile_gpu():
    settings = get_settings()
    device_name = settings.resolve_device()

    print("=================================================================================")
    print("AUTOROLL GPU MEMORY & TRAINING THROUGHPUT PROFILER")
    print("=================================================================================")
    print(f"Target Hardware Execution Device : {device_name.upper()}")

    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch CUDA Available           : {cuda_avail}")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"GPU Hardware Model               : {gpu_name}")
        print(f"Total VRAM Capacity              : {total_mem / (1024**3):.2f} GB")
        print(f"Initial Available VRAM           : {free_mem / (1024**3):.2f} GB")
    else:
        print("WARNING: Running in CPU Execution Mode (CUDA not active).")

    # Load ArcFace Model ONNX Session
    model_path = settings.ARCFACE_GLINT_PATH
    logger.info(f"Loading ArcFace Recognition Model from '{model_path}'...")
    
    vram_before_load = torch.cuda.memory_allocated() if cuda_avail else 0
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_avail else ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        active_providers = session.get_providers()
        print(f"Active ONNXRuntime Providers     : {active_providers}")
    except Exception as e:
        logger.error(f"Failed to load ONNX model on CUDAExecutionProvider: {e}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    vram_after_load = torch.cuda.memory_allocated() if cuda_avail else 0
    print(f"VRAM Allocated After Model Load  : {(vram_after_load - vram_before_load) / (1024**2):.2f} MB")

    input_name = session.get_inputs()[0].name
    candidate_batch_sizes = [16, 32, 64, 128]
    
    print("\n---------------------------------------------------------------------------------")
    print("BATCH SIZE PROBING & MEMORY PROFILING RESULTS")
    print("---------------------------------------------------------------------------------")
    print(f"{'BATCH SIZE':<12} | {'FORWARD LATENCY (ms)':<22} | {'THROUGHPUT (img/s)':<20} | {'PEAK VRAM (MB)':<15} | {'STATUS':<8}")
    print("-" * 85)

    max_stable_batch_size = 0

    for batch_size in candidate_batch_sizes:
        try:
            # Create synthetic 112x112 batch tensor
            dummy_batch = torch.randn(batch_size, 3, 112, 112, dtype=torch.float32)
            if cuda_avail:
                torch.cuda.reset_peak_memory_stats()
                start_vram = torch.cuda.memory_allocated()

            # Warmup run
            np_batch = dummy_batch.numpy()
            _ = session.run(None, {input_name: np_batch})

            # Timed benchmark run
            iterations = 5
            t0 = time.time()
            for _ in range(iterations):
                _ = session.run(None, {input_name: np_batch})
            elapsed = time.time() - t0

            avg_batch_latency_ms = (elapsed / iterations) * 1000
            fps = (batch_size * iterations) / elapsed

            peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if cuda_avail else 0.0

            print(f"{batch_size:<12} | {avg_batch_latency_ms:<22.2f} | {fps:<20.2f} | {peak_vram_mb:<15.2f} | STABLE")
            max_stable_batch_size = batch_size

        except Exception as e:
            print(f"{batch_size:<12} | {'N/A':<22} | {'N/A':<20} | {'OOM / Error':<15} | FAILED ({e})")
            break

    print("=================================================================================")
    print(f"PROFILING COMPLETE | Recommended Maximum Stable Batch Size: {max_stable_batch_size}")
    print("=================================================================================\n")


if __name__ == "__main__":
    profile_gpu()
