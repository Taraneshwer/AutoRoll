"""
Unit tests for MultiFaceTracker module.
"""

from app.schemas.common import BoundingBox
from app.ml.inference.tracker import MultiFaceTracker


def test_compute_iou():
    box_a = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
    box_b = BoundingBox(x1=50, y1=0, x2=150, y2=100, confidence=0.9)

    iou = MultiFaceTracker.compute_iou(box_a, box_b)
    assert 0.30 <= iou <= 0.35


def test_tracker_association():
    tracker = MultiFaceTracker(iou_threshold=0.30, recognition_interval=10)

    # Frame 1: 2 face detections
    det_f1 = [
        (BoundingBox(x1=10, y1=10, x2=60, y2=60, confidence=0.95), [], 0.95),
        (BoundingBox(x1=100, y1=10, x2=150, y2=60, confidence=0.92), [], 0.92),
    ]

    tracks_f1 = tracker.update(det_f1, frame_index=1)
    assert len(tracks_f1) == 2
    assert tracks_f1[0][0].track_id == 1
    assert tracks_f1[1][0].track_id == 2
    # New faces must trigger recognition
    assert tracks_f1[0][1] is True
    assert tracks_f1[1][1] is True

    # Frame 2: Same faces slightly shifted
    det_f2 = [
        (BoundingBox(x1=12, y1=11, x2=62, y2=61, confidence=0.94), [], 0.94),
        (BoundingBox(x1=102, y1=11, x2=152, y2=61, confidence=0.91), [], 0.91),
    ]

    tracks_f2 = tracker.update(det_f2, frame_index=2)
    assert len(tracks_f2) == 2
    assert tracks_f2[0][0].track_id == 1
    assert tracks_f2[1][0].track_id == 2
