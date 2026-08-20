"""
Device selection and ONNX Runtime execution provider helper.
"""


from autoroll.common.logger import get_logger

logger = get_logger("ml_utils")


def is_cuda_functional() -> bool:
    """
    Checks if PyTorch CUDA is not only reported as available,
    but can actually execute tensor operations on the hardware device.
    Prevents crashes on newer GPU architectures (e.g. sm_120 / Blackwell)
    where torch.cuda.is_available() is True but CUBIN kernels are missing.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        t = torch.zeros(1, device="cuda")
        _ = t + 1.0
        torch.cuda.synchronize()
        return True
    except Exception as e:
        logger.warning(
            f"CUDA hardware reported by PyTorch, but kernel execution failed: {e}. Falling back to CPU."
        )
        return False


def get_execution_device(device_preference: str = "auto") -> tuple[str, list[str]]:
    """
    Determines execution device ('cuda' or 'cpu') and ONNX Runtime providers list.
    device_preference can be 'auto', 'cuda', or 'cpu'.
    Primary choice is GPU ('cuda'); falls back to CPU seamlessly if GPU is unavailable or non-functional.
    """
    pref = device_preference.lower().strip()
    cuda_functional = is_cuda_functional()

    ort_providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail and cuda_functional:
            ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in avail:
            ort_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "TensorrtExecutionProvider" in avail and cuda_functional:
            ort_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass

    if pref == "cpu":
        return "cpu", ["CPUExecutionProvider"]

    if pref == "cuda":
        if cuda_functional or ort_providers[0] != "CPUExecutionProvider":
            logger.info(f"GPU execution explicitly selected ({ort_providers[0]}).")
            return "cuda", ort_providers
        logger.warning("CUDA requested, but GPU execution is non-functional on this system. Falling back to CPU.")
        return "cpu", ["CPUExecutionProvider"]

    # 'auto' preference
    if cuda_functional or ort_providers[0] != "CPUExecutionProvider":
        logger.info(f"Auto-detected functional GPU hardware. Selecting primary provider: {ort_providers[0]}")
        return "cuda", ort_providers

    logger.info("Auto-selected CPU execution device (CPU fallback).")
    return "cpu", ["CPUExecutionProvider"]


def create_onnx_session(model_path: str, device_preference: str = "auto"):
    """
    Creates an ONNX Runtime InferenceSession attempting GPU execution primarily,
    with automatic seamless fallback to CPU if GPU provider fails.
    """
    import onnxruntime as ort

    device, providers = get_execution_device(device_preference)
    opts = ort.SessionOptions()

    try:
        session = ort.InferenceSession(model_path, opts, providers=providers)
        return session, device, providers
    except Exception as e:
        logger.warning(
            f"Failed to create ONNX session with primary providers {providers}: {e}. Falling back to CPUExecutionProvider."
        )
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, opts, providers=providers)
        return session, "cpu", providers

