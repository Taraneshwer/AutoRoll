"""
Independent SCRFD Face Detector Module supporting ONNX Runtime (CPU/CUDA)
with fallback for unweighted testing environments.
"""

import os

import cv2
import numpy as np

from autoroll.common.logger import get_logger
from autoroll.common.schemas import BoundingBox, DetectionResult, FaceLandmarks
from autoroll.ml.detectors.base import BaseFaceDetector
from autoroll.ml.utils import get_execution_device

logger = get_logger("scrfd_detector")


class SCRFDDetector(BaseFaceDetector):
    """
    SCRFD Face Detector implementation using ONNX Runtime.
    Supports CUDA / CPU devices with automatic hardware detection.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_path = model_path
        self.session = None
        from autoroll.common.config import get_settings
        settings = get_settings()
        self.ml_mode = settings.AUTOROLL_ML_MODE.lower()
        self.model_name = "SCRFD_10G_KPS"
        self.model_version = "1.0.0"

        path_to_use = self.model_path or settings.SCRFD_MODEL_PATH
        self.model_path = path_to_use

        if os.path.exists(self.model_path):
            try:
                from autoroll.ml.utils import create_onnx_session
                self.session, self.device, self.providers = create_onnx_session(
                    self.model_path, device_preference=device
                )
                logger.info(
                    f"ML MODEL LOADED | Name: '{self.model_name}' | "
                    f"Version: '{self.model_version}' | Path: '{self.model_path}' | "
                    f"Backend: ONNXRuntime | Device: '{self.device}' | "
                    "Precision: FP32 | Status: READY"
                )
            except Exception as e:
                err_msg = (
                    f"PRODUCTION ML ERROR: Failed to load SCRFD ONNX model from "
                    f"'{self.model_path}': {e}"
                )
                if self.ml_mode == "production":
                    logger.error(err_msg)
                    raise RuntimeError(err_msg) from e
                logger.warning(f"{err_msg}. Operating in TEST fallback mode.")
        else:
            err_msg = f"PRODUCTION ML ERROR: SCRFD model weights not found at '{self.model_path}'."
            if self.ml_mode == "production":
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            logger.info(f"{err_msg} Operating in TEST fallback mode.")

    def detect(self, image: np.ndarray, score_threshold: float = 0.5) -> list[DetectionResult]:
        """
        Detects faces in input image.
        Returns list of DetectionResult objects.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid or empty input image provided for face detection.")

        threshold = score_threshold or self.conf_threshold

        if self.session is not None:
            return self._detect_onnx(image, threshold)
        else:
            return self._detect_fallback(image, threshold)

    def _get_centers(self, feat_h: int, feat_w: int, stride: int) -> np.ndarray:
        key = (feat_h, feat_w, stride)
        if hasattr(self, "_center_cache") and key in self._center_cache:
            return self._center_cache[key]
        if not hasattr(self, "_center_cache"):
            self._center_cache = {}
        anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape((-1, 2))
        self._center_cache[key] = anchor_centers
        return anchor_centers

    def _detect_onnx(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        """
        Runs ONNX Runtime inference for SCRFD.
        """
        input_name = self.session.get_inputs()[0].name
        h_orig, w_orig = image.shape[:2]

        img_resized = cv2.resize(image, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = np.transpose(img_rgb, (2, 0, 1))[None, ...]

        net_outs = self.session.run(None, {input_name: blob})

        detections: list[DetectionResult] = []

        # Single concatenated output fallback
        if len(net_outs) == 1 or (len(net_outs) > 0 and net_outs[0].ndim == 2 and net_outs[0].shape[1] >= 15):
            for out in net_outs:
                if out.ndim == 2 and out.shape[1] >= 15:
                    for row in out:
                        score = float(row[4])
                        if score >= threshold:
                            x1 = float(row[0] * w_orig / 640.0)
                            y1 = float(row[1] * h_orig / 640.0)
                            x2 = float(row[2] * w_orig / 640.0)
                            y2 = float(row[3] * h_orig / 640.0)
                            landmarks_pts = [
                                (float(row[5 + i * 2] * w_orig / 640.0), float(row[6 + i * 2] * h_orig / 640.0))
                                for i in range(5)
                            ]
                            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score)
                            landmarks = FaceLandmarks(points=landmarks_pts)
                            detections.append(
                                DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=score)
                            )
            return detections

        # Multi-stride SCRFD decoding (9 output tensors: scores, bboxes, keypoints for strides 8, 16, 32)
        strides = [8, 16, 32]
        scores_list = net_outs[0:3]
        bboxes_list = net_outs[3:6]
        kps_list = net_outs[6:9] if len(net_outs) >= 9 else [None, None, None]

        all_boxes = []
        all_scores = []
        all_kps = []

        for idx, stride in enumerate(strides):
            scores = scores_list[idx]
            bboxes = bboxes_list[idx]
            kpss = kps_list[idx]

            feat_h = 640 // stride
            feat_w = 640 // stride
            anchor_centers = self._get_centers(feat_h, feat_w, stride)

            scores_flat = scores.flatten()
            pos_inds = np.where(scores_flat >= threshold)[0]

            for pos in pos_inds:
                score = float(scores_flat[pos])
                center = anchor_centers[pos]
                dx, dy, dw, dh = bboxes[pos] * stride

                x1 = float((center[0] - dx) * w_orig / 640.0)
                y1 = float((center[1] - dy) * h_orig / 640.0)
                x2 = float((center[0] + dw) * w_orig / 640.0)
                y2 = float((center[1] + dh) * h_orig / 640.0)

                kp_pts = []
                if kpss is not None:
                    kp_raw = kpss[pos].reshape((5, 2)) * stride
                    for k in range(5):
                        kx = float((center[0] + kp_raw[k, 0]) * w_orig / 640.0)
                        ky = float((center[1] + kp_raw[k, 1]) * h_orig / 640.0)
                        kp_pts.append((kx, ky))
                else:
                    # Fallback keypoints based on bbox box bounds
                    fw, fh = x2 - x1, y2 - y1
                    kp_pts = [
                        (x1 + fw * 0.30, y1 + fh * 0.35),
                        (x1 + fw * 0.70, y1 + fh * 0.35),
                        (x1 + fw * 0.50, y1 + fh * 0.55),
                        (x1 + fw * 0.35, y1 + fh * 0.75),
                        (x1 + fw * 0.65, y1 + fh * 0.75),
                    ]

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_kps.append(kp_pts)

        if not all_boxes:
            return []

        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_arr.tolist(), scores_arr.tolist(), threshold, self.nms_threshold
        )

        if len(indices) > 0:
            for i in indices.flatten():
                bx = boxes_arr[i]
                bbox = BoundingBox(x1=float(bx[0]), y1=float(bx[1]), x2=float(bx[2]), y2=float(bx[3]), confidence=float(scores_arr[i]))
                landmarks = FaceLandmarks(points=all_kps[i])
                detections.append(
                    DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=float(scores_arr[i]))
                )

        return detections

    def _detect_fallback(self, image: np.ndarray, threshold: float) -> list[DetectionResult]:
        """
        Pure algorithmic fallback face detector for test and unweighted local runs.
        Produces deterministic face bounding box and 5 facial landmark points.
        """
        h, w = image.shape[:2]
        detections: list[DetectionResult] = []

        # Synthetic/Fallback box centered in frame
        x1 = float(w * 0.25)
        y1 = float(h * 0.20)
        x2 = float(w * 0.75)
        y2 = float(h * 0.80)
        fw = x2 - x1
        fh = y2 - y1
        score = 0.95

        landmarks_pts = [
            (x1 + fw * 0.30, y1 + fh * 0.35),  # Left eye
            (x1 + fw * 0.70, y1 + fh * 0.35),  # Right eye
            (x1 + fw * 0.50, y1 + fh * 0.55),  # Nose tip
            (x1 + fw * 0.35, y1 + fh * 0.75),  # Left mouth corner
            (x1 + fw * 0.65, y1 + fh * 0.75),  # Right mouth corner
        ]

        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score)
        landmarks = FaceLandmarks(points=landmarks_pts)
        detections.append(
            DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=score)
        )

        return detections
