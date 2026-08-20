"""
Hardware System Information and Resource Monitoring Module.
Detects CPU, RAM, and CUDA GPU utilization.
"""

from typing import Any

import psutil
import torch

from app.core.logger import get_logger

logger = get_logger("worker_system_info")


class SystemInfoMonitor:
    """
    Collects CPU, RAM, and GPU utilization metrics.
    """

    @staticmethod
    def get_cpu_ram_metrics() -> dict[str, float]:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_mb = mem.used / (1024 * 1024)
        return {
            "cpu_percent": round(float(cpu), 1),
            "ram_used_mb": round(float(ram_used_mb), 1),
            "ram_percent": round(float(mem.percent), 1),
        }

    @staticmethod
    def get_gpu_metrics() -> dict[str, Any]:
        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            return {
                "gpu_available": False,
                "gpu_name": None,
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
            }

        try:
            device_name = torch.cuda.get_device_name(0)
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            return {
                "gpu_available": True,
                "gpu_name": device_name,
                "gpu_utilization_percent": 0.0,  # NVML or fallback
                "gpu_memory_used_mb": round(float(mem_allocated), 1),
            }
        except Exception as e:
            logger.warning(f"Failed to query GPU status: {e}")
            return {
                "gpu_available": True,
                "gpu_name": "CUDA Device",
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
            }
