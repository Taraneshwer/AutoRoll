"""
Worker Metrics Collector — AutoRoll Phase 14
Gathers local system CPU, memory, GPU utilization, and VRAM memory telemetry.
"""

import os
import psutil
import time
from typing import Dict, Any


class WorkerMetricsCollector:
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Collect CPU, RAM, and GPU telemetry."""
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        gpu_name = "CPU"
        gpu_util = 0.0
        vram_used = 0.0
        vram_total = 0.0

        # Attempt PyTorch CUDA GPU metrics collection if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_used = torch.cuda.memory_allocated(0) / (1024 * 1024)  # MB
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
                gpu_util = min(100.0, (vram_used / vram_total) * 100.0) if vram_total > 0 else 0.0
        except Exception:
            pass

        return {
            "cpu_percent": cpu_percent,
            "ram_used_mb": mem.used / (1024 * 1024),
            "ram_total_mb": mem.total / (1024 * 1024),
            "gpu_name": gpu_name,
            "gpu_utilization": gpu_util,
            "gpu_memory_used": vram_used,
            "gpu_memory_total": vram_total,
            "timestamp": time.time(),
        }
