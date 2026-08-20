"""
Multi-Face Tracker with IoU Association and Adaptive Recognition Scheduling.
"""


import numpy as np

from autoroll.common.logger import get_logger
from autoroll.common.schemas import BoundingBox

logger = get_logger("multi_face_tracker")


class FaceTrack:
    """
    Represents an active face track across consecutive video frames.
    """

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        landmarks: list[tuple[float, float]],
        confidence: float,
        frame_index: int,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        self.start_frame = frame_index
        self.last_frame = frame_index
        self.frames_tracked = 1
        self.disappeared_count = 0

        # Cached Recognition & Liveness states
        self.embedding: list[float] | None = None
        self.is_live: bool = False
        self.liveness_score: float = 0.0
        self.liveness_decision: str = "PENDING"
        self.recognition_status: str = "PENDING"
        self.last_recognition_frame: int = -1

    def update(
        self,
        bbox: BoundingBox,
        landmarks: list[tuple[float, float]],
        confidence: float,
        frame_index: int,
    ) -> None:
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        self.last_frame = frame_index
        self.frames_tracked += 1
        self.disappeared_count = 0

    def should_trigger_recognition(
        self,
        current_frame: int,
        recognition_interval: int = 10,
        min_confidence: float = 0.60,
    ) -> bool:
        """
        Determines whether expensive recognition embedding extraction should run on this frame.
        """
        # 1. First time recognition
        if self.embedding is None or self.last_recognition_frame < 0:
            return True

        # 2. Configurable frame interval expired
        if (current_frame - self.last_recognition_frame) >= recognition_interval:
            return True

        # 3. Low tracking confidence
        if self.confidence < min_confidence:
            return True

        return False


class MultiFaceTracker:
    """
    Tracks multiple faces across frames using IoU (Intersection over Union) association.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_disappeared: int = 5,
        recognition_interval: int = 10,
    ):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.recognition_interval = recognition_interval

        self.next_track_id = 1
        self.active_tracks: dict[int, FaceTrack] = {}

    @staticmethod
    def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        x_a = max(box_a.x1, box_b.x1)
        y_a = max(box_a.y1, box_b.y1)
        x_b = min(box_a.x2, box_b.x2)
        y_b = min(box_a.y2, box_b.y2)

        inter_width = max(0.0, x_b - x_a)
        inter_height = max(0.0, y_b - y_a)
        inter_area = inter_width * inter_height

        area_a = box_a.area
        area_b = box_b.area

        union_area = float(area_a + area_b - inter_area)
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def update(
        self,
        detections: list[tuple[BoundingBox, list[tuple[float, float]], float]],
        frame_index: int,
    ) -> list[tuple[FaceTrack, bool]]:
        """
        Updates active face tracks with new frame detections.
        Returns List of Tuples: (FaceTrack, trigger_recognition_flag)
        """
        updated_tracks: list[tuple[FaceTrack, bool]] = []

        if not detections:
            # Increment disappeared count for all active tracks
            to_remove = []
            for track_id, track in self.active_tracks.items():
                track.disappeared_count += 1
                if track.disappeared_count > self.max_disappeared:
                    to_remove.append(track_id)
            for tid in to_remove:
                del self.active_tracks[tid]
            return []

        # If no active tracks exist, assign all detections to new tracks
        if not self.active_tracks:
            for bbox, lms, conf in detections:
                track = FaceTrack(self.next_track_id, bbox, lms, conf, frame_index)
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1
                updated_tracks.append((track, True))
            return updated_tracks

        # Compute IoU cost matrix between active tracks and new detections
        track_ids = list(self.active_tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            for j, (det_bbox, _, _) in enumerate(detections):
                iou_matrix[i, j] = self.compute_iou(self.active_tracks[tid].bbox, det_bbox)

        # Greedy matching: match pairs with highest IoU >= iou_threshold
        matched_tracks = set()
        matched_detections = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_val = iou_matrix[max_iou_idx]

            if max_val < self.iou_threshold:
                break

            t_idx, d_idx = max_iou_idx
            if t_idx not in matched_tracks and d_idx not in matched_detections:
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)

                tid = track_ids[t_idx]
                det_bbox, det_lms, det_conf = detections[d_idx]
                track = self.active_tracks[tid]
                track.update(det_bbox, det_lms, det_conf, frame_index)

                trigger_rec = track.should_trigger_recognition(
                    frame_index, self.recognition_interval
                )
                updated_tracks.append((track, trigger_rec))

            # Mask matched row and column
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        # Unmatched detections -> create new tracks
        for j, (det_bbox, det_lms, det_conf) in enumerate(detections):
            if j not in matched_detections:
                track = FaceTrack(self.next_track_id, det_bbox, det_lms, det_conf, frame_index)
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1
                updated_tracks.append((track, True))

        # Unmatched tracks -> increment disappeared
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                track = self.active_tracks[tid]
                track.disappeared_count += 1

        # Remove stale tracks
        to_remove = [
            tid
            for tid, trk in self.active_tracks.items()
            if trk.disappeared_count > self.max_disappeared
        ]
        for tid in to_remove:
            del self.active_tracks[tid]

        return updated_tracks
