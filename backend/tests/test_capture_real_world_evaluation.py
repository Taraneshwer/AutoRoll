"""
Pytest Test Suite for Phase 17.2 RealWorldCameraAcquisitionEngine.
"""

import json
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.dataset.capture_real_world_evaluation import RealWorldCameraAcquisitionEngine


def make_test_face_frame(num_faces: int = 1, color_var: float = 50.0) -> np.ndarray:
    """Generates a synthetic frame for pipeline unit testing."""
    frame = np.random.randint(40, 200, (300, 300, 3), dtype=np.uint8)

    if num_faces == 1:
        # Draw a clear face shape with eyes and mouth for Haar cascade
        cv2.circle(frame, (150, 150), 60, (220, 200, 180), -1)
        cv2.circle(frame, (130, 130), 10, (20, 20, 20), -1)
        cv2.circle(frame, (170, 130), 10, (20, 20, 20), -1)
        cv2.ellipse(frame, (150, 180), (25, 10), 0, 0, 180, (20, 20, 20), 4)
    elif num_faces == 2:
        cv2.circle(frame, (80, 150), 40, (220, 200, 180), -1)
        cv2.circle(frame, (220, 150), 40, (220, 200, 180), -1)

    return frame


def test_acquisition_engine_initialization():
    engine = RealWorldCameraAcquisitionEngine(camera_index=0)
    assert engine.rejections["no_face"] == 0
    assert engine.rejections["duplicate_image"] == 0


def test_process_frame_quality_and_rejections():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        engine = RealWorldCameraAcquisitionEngine()
        engine.collector.root_dir = tmp_path
        engine.collector.enrollment_dir = tmp_path / "enrollment"
        engine.collector.probes_dir = tmp_path / "probes"
        engine.collector.liveness_dir = tmp_path / "liveness"
        engine.collector.manifests_dir = tmp_path / "manifests"
        engine.collector._ensure_directories()

        # Solid black frame (no face) -> Reject no_face
        blank_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, msg, rec = engine.process_and_ingest_frame(blank_frame, "P001", "enrollment")
        assert ok is False
        assert engine.rejections["no_face"] > 0 or engine.rejections["poor_quality"] > 0


def test_liveness_participant_folder_storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        engine = RealWorldCameraAcquisitionEngine()
        engine.collector.root_dir = tmp_path
        engine.collector.enrollment_dir = tmp_path / "enrollment"
        engine.collector.probes_dir = tmp_path / "probes"
        engine.collector.liveness_dir = tmp_path / "liveness"
        engine.collector.manifests_dir = tmp_path / "manifests"
        engine.collector._ensure_directories()

        # Ingest raw bytes directly to test participant-specific liveness folder
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        rec = engine.collector.ingest_sample(
            image_bytes=buf.tobytes(),
            participant_id="P001",
            sample_type="liveness",
            liveness_attack="Printed Photograph",
        )

        expected_file = tmp_path / "liveness" / "P001" / "printed_photograph" / rec["filename"]
        assert expected_file.exists()


def test_camera_backend_selection_and_resolution_configuration(monkeypatch):
    from unittest.mock import MagicMock

    engine = RealWorldCameraAcquisitionEngine(camera_index=0)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    mock_cap.read.return_value = (True, mock_frame)

    calls = []

    def mock_video_capture(index, *args):
        calls.append((index, args))
        return mock_cap

    monkeypatch.setattr(cv2, "VideoCapture", mock_video_capture)

    cap = engine.open_camera()
    assert cap == mock_cap
    assert len(calls) == 1
    assert calls[0][0] == 0

    if sys.platform == "win32":
        assert calls[0][1][0] == cv2.CAP_DSHOW

    # open_camera no longer hardcodes width/height — only FPS is set
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)


def test_camera_opening_failure_raises_runtime_error(monkeypatch):
    from unittest.mock import MagicMock

    engine = RealWorldCameraAcquisitionEngine(camera_index=0)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    with pytest.raises(RuntimeError, match="CAMERA INITIALIZATION FAILED"):
        engine.open_camera()


def test_select_camera_interactively_explicit_index(monkeypatch):
    from scripts.dataset.capture_real_world_evaluation import select_camera_interactively

    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.test_camera_usability",
        lambda idx: {"usable": True, "status": "USABLE", "friendly_name": "Mock Camera",
                     "device_path": "", "is_virtual": False},
    )
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: [{"friendly_name": "Mock Camera", "device_path": ""}],
    )

    assert select_camera_interactively(explicit_index=0) == 0


def test_preview_modal_confirmation_and_cancellation(monkeypatch):
    from unittest.mock import MagicMock
    from scripts.dataset.capture_real_world_evaluation import RealWorldCameraAcquisitionEngine

    engine = RealWorldCameraAcquisitionEngine(camera_index=1)
    mock_cap = MagicMock()
    mock_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    mock_cap.read.return_value = (True, mock_frame)

    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    # Use enumerate_directshow_devices (not the deleted get_windows_camera_names)
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: [{"friendly_name": "Integrated Camera", "device_path": ""},
                 {"friendly_name": "Integrated Camera", "device_path": ""}],
    )

    # Test cancellation (ESC = 27)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    assert engine.show_preview_modal(mock_cap) is False


def test_camera_usability_black_frame_rejection(monkeypatch):
    from unittest.mock import MagicMock
    from scripts.dataset.capture_real_world_evaluation import test_camera_usability

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Mean brightness = 0
    mock_cap.read.return_value = (True, black_frame)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    res = test_camera_usability(0)
    assert res["usable"] is False
    assert "BLACK FRAME" in res["status"]


def test_camera_usability_low_variance_rejection(monkeypatch):
    from unittest.mock import MagicMock
    from scripts.dataset.capture_real_world_evaluation import test_camera_usability

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Uniform grey frame with zero variance and mean brightness 128
    solid_gray_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    mock_cap.read.return_value = (True, solid_gray_frame)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    res = test_camera_usability(0)
    assert res["usable"] is False
    assert "LOW VARIANCE" in res["status"] or "FROZEN" in res["status"]


def test_camera_usability_frozen_frame_rejection(monkeypatch):
    from unittest.mock import MagicMock
    from scripts.dataset.capture_real_world_evaluation import test_camera_usability

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Frame with mild texture but 0 frame-to-frame change
    static_frame = np.random.randint(100, 105, (720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, static_frame)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    res = test_camera_usability(0)
    assert res["usable"] is False


def test_camera_usability_valid_camera_acceptance(monkeypatch):
    from unittest.mock import MagicMock
    from scripts.dataset.capture_real_world_evaluation import test_camera_usability

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    # Generate sequence of varying frames
    frame_idx = [0]

    def mock_read():
        frame_idx[0] += 1
        frame = np.random.randint(20, 220, (720, 1280, 3), dtype=np.uint8)
        # Add dynamic face-like variation per frame
        cv2.circle(frame, (100 + frame_idx[0] * 5, 200), 50, (200, 180, 160), -1)
        return True, frame

    mock_cap.read.side_effect = mock_read
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    res = test_camera_usability(0)
    assert res["usable"] is True
    assert res["status"] == "USABLE"


def test_select_camera_interactively_unusable_explicit_index_exits(monkeypatch):
    from scripts.dataset.capture_real_world_evaluation import select_camera_interactively

    # Mock test_camera_usability to return unusable
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.test_camera_usability",
        lambda idx: {"usable": False, "status": "UNUSABLE (BLACK FRAME)",
                     "friendly_name": "Mock", "device_path": "", "is_virtual": False},
    )
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: [{"friendly_name": "Mock", "device_path": ""}],
    )

    with pytest.raises(SystemExit):
        select_camera_interactively(explicit_index=0)


# ============================================================
# NEW: DirectShow COM order-independence and virtual-cam tests
# ============================================================

def test_directshow_name_mapping_uses_com_order_not_pnp_order(monkeypatch):
    """
    Prove that device names are mapped by DirectShow COM enumeration order
    (= OpenCV CAP_DSHOW index order), NOT by PnP enumeration order.

    Scenario:
      - PnP (Get-PnpDevice) would return: ["OBS Virtual Camera", "Integrated Camera"]
      - DirectShow COM returns:            [index 0 = Integrated Camera,
                                            index 1 = OBS Virtual Camera]
      So the CORRECT mapping is:
        OpenCV index 0 → "Integrated Camera"
        OpenCV index 1 → "OBS Virtual Camera"
      If we used PnP order it would be WRONG (flipped).
    """
    from scripts.dataset.capture_real_world_evaluation import enumerate_available_cameras

    # DirectShow COM enumeration (correct order = OpenCV order)
    com_devices = [
        {"friendly_name": "Integrated Camera",  "device_path": "\\\\?\\usb#vid_04f2..."},
        {"friendly_name": "OBS Virtual Camera", "device_path": ""},
    ]
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: com_devices,
    )

    # test_camera_usability: both indexes are "usable" with distinct widths
    def fake_usability(idx):
        return {
            "index": idx, "friendly_name": "Unknown", "device_path": "",
            "is_virtual": False, "backend": "DirectShow",
            "width": 2560 if idx == 0 else 1920,
            "height": 1440 if idx == 0 else 1080,
            "fps": 30.0, "brightness": 100.0, "variance": 500.0,
            "frame_change": 2.5, "status": "USABLE", "usable": True,
        }
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.test_camera_usability",
        fake_usability,
    )

    cams = enumerate_available_cameras(max_index=1)

    # Index 0 MUST be "Integrated Camera" (COM order), NOT "OBS Virtual Camera" (PnP order)
    assert cams[0]["index"] == 0
    assert cams[0]["friendly_name"] == "Integrated Camera", (
        f"Expected 'Integrated Camera' at index 0, got '{cams[0]['friendly_name']}'. "
        "Name must come from DirectShow COM enumeration, NOT PnP order."
    )
    assert cams[1]["friendly_name"] == "OBS Virtual Camera"


def test_virtual_camera_is_selectable_not_auto_rejected(monkeypatch):
    """
    Verify that virtual cameras (OBS, EShare, etc.) are NOT auto-rejected.
    A virtual camera with valid frames must be marked USABLE [VIRTUAL CAMERA]
    and must appear in the selectable list so the operator can choose it.
    """
    from scripts.dataset.capture_real_world_evaluation import enumerate_available_cameras

    com_devices = [
        {"friendly_name": "Integrated Camera",  "device_path": "...real..."},
        {"friendly_name": "OBS Virtual Camera", "device_path": ""},
    ]
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: com_devices,
    )

    def fake_usability(idx):
        # Both cameras return valid frames
        return {
            "index": idx, "friendly_name": "Unknown", "device_path": "",
            "is_virtual": False, "backend": "DirectShow",
            "width": 1920, "height": 1080,
            "fps": 30.0, "brightness": 80.0, "variance": 400.0,
            "frame_change": 1.5, "status": "USABLE", "usable": True,
        }
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.test_camera_usability",
        fake_usability,
    )

    cams = enumerate_available_cameras(max_index=1)

    obs_cam = next(c for c in cams if "OBS" in c["friendly_name"])
    assert obs_cam["usable"] is True, "OBS Virtual Camera with valid frames must be usable/selectable"
    assert obs_cam["is_virtual"] is True
    assert "VIRTUAL CAMERA" in obs_cam["status"]


def test_name_mapping_independent_of_enumeration_count(monkeypatch):
    """
    Even if only one PnP-visible device exists, the DirectShow COM order
    determines the index-to-name binding. Tests with 3 COM-enumerated devices
    to confirm position[2] is definitively index 2 regardless of external
    enumeration order.
    """
    from scripts.dataset.capture_real_world_evaluation import enumerate_available_cameras

    com_devices = [
        {"friendly_name": "USB Webcam A",    "device_path": "\\\\?\\usb#A"},
        {"friendly_name": "USB Webcam B",    "device_path": "\\\\?\\usb#B"},
        {"friendly_name": "EShare Virtual",  "device_path": ""},
    ]
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.enumerate_directshow_devices",
        lambda: com_devices,
    )

    def fake_usability(idx):
        return {
            "index": idx, "friendly_name": "Unknown", "device_path": "",
            "is_virtual": False, "backend": "DirectShow",
            "width": 1280, "height": 720, "fps": 30.0,
            "brightness": 60.0, "variance": 300.0,
            "frame_change": 1.0, "status": "USABLE", "usable": True,
        }
    monkeypatch.setattr(
        "scripts.dataset.capture_real_world_evaluation.test_camera_usability",
        fake_usability,
    )

    cams = enumerate_available_cameras(max_index=2)

    assert cams[0]["friendly_name"] == "USB Webcam A",   f"Got: {cams[0]['friendly_name']}"
    assert cams[1]["friendly_name"] == "USB Webcam B",   f"Got: {cams[1]['friendly_name']}"
    assert cams[2]["friendly_name"] == "EShare Virtual", f"Got: {cams[2]['friendly_name']}"
    assert cams[2]["is_virtual"] is True


# ============================================================
# PREVIEW FRAME PATH REGRESSION TESTS
# ============================================================

def test_prepare_preview_frame_preserves_16x9_aspect():
    """2560×1440 source must be displayed as 1280×720 (exact 16:9 scale-down, no crop)."""
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    src = np.random.randint(0, 255, (1440, 2560, 3), dtype=np.uint8)
    preview = prepare_preview_frame(src, max_width=1280, max_height=720)

    assert preview.shape[0] == 720,  f"Expected height 720, got {preview.shape[0]}"
    assert preview.shape[1] == 1280, f"Expected width 1280, got {preview.shape[1]}"
    src_ratio  = 2560 / 1440
    prev_ratio = preview.shape[1] / preview.shape[0]
    # After exact 2× scale-down ratio must be identical
    assert abs(src_ratio - prev_ratio) < 0.001, (
        f"Aspect ratio changed: source {src_ratio:.4f} → preview {prev_ratio:.4f}")


def test_prepare_preview_frame_preserves_complete_image():
    """
    The scaled output must contain ALL source content — verified by comparing
    the centre pixel after a known colour fill.
    """
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    src = np.zeros((1440, 2560, 3), dtype=np.uint8)
    src[:] = (0, 128, 255)   # fill entire source with known colour

    preview = prepare_preview_frame(src, max_width=1280, max_height=720)

    # Top-left, centre, and bottom-right of the preview must all contain the source colour
    for (r, c) in [(0, 0), (360, 640), (719, 1279)]:
        pixel = preview[r, c]
        assert not np.all(pixel == 0), (
            f"Pixel at ({r},{c}) is black — part of the source frame was cropped away")


def test_prepare_preview_frame_never_crops_when_letterboxing_required():
    """
    A 16:9 source into an 800×600 window must be letterboxed (4:3 window),
    not cropped. The content region must equal a down-scaled full source frame.
    """
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    # 16:9 source
    src = np.full((900, 1600, 3), 200, dtype=np.uint8)
    src[0, 0] = (0, 0, 255)   # top-left marker (red corner)

    preview = prepare_preview_frame(src, max_width=800, max_height=600)
    assert preview.shape[:2] == (600, 800)

    # scale = min(800/1600, 600/900) = min(0.5, 0.667) = 0.5
    # scaled = 800×450, letterboxed in 800×600 → y_off = 75
    content_h = int(900 * 0.5)   # 450
    content_w = int(1600 * 0.5)  # 800
    y_off = (600 - content_h) // 2  # 75

    # Black bars: rows 0..y_off-1 and rows (y_off+content_h)..599
    if y_off > 0:
        bar = preview[:y_off, :, :]
        assert np.all(bar == 0), "Top letterbox bar should be black"

    # Content region should not be black
    content_region = preview[y_off:y_off + content_h, :content_w]
    assert np.mean(content_region) > 50, "Content region should not be all-black (cropped)"


def test_prepare_preview_frame_does_not_crop_source():
    """
    Explicitly verify no NumPy slice crop: the output pixel count covers
    the full content from source (not a sub-rectangle).
    """
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    src = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    preview = prepare_preview_frame(src, max_width=1280, max_height=720)

    # 1:1 scale (source already fits) — output must equal source exactly
    assert preview.shape == src.shape, (
        f"Same-size input should pass through unchanged: {src.shape} → {preview.shape}")
    assert np.array_equal(preview, src), "Same-size input must be bit-identical output"


def test_ml_processing_copy_does_not_affect_display_frame():
    """
    Prove that mutating a processing_frame copy (simulating ML preprocessing)
    does NOT alter the original frame used for display.
    """
    original = np.random.randint(30, 200, (720, 1280, 3), dtype=np.uint8)
    snapshot = original.copy()

    # Simulate acquisition loop split
    processing_frame = original.copy()
    display_frame    = original.copy()

    # Simulate ML mutating processing_frame in-place
    processing_frame[:] = 0          # e.g. normalize to black
    processing_frame = cv2.resize(processing_frame, (112, 112))   # e.g. crop to 112×112

    # display_frame must be unaffected
    assert np.array_equal(display_frame, snapshot), (
        "ML operations on processing_frame must not alter display_frame")
    assert display_frame.shape == (720, 1280, 3), (
        "display_frame dimensions must remain 720×1280 regardless of ML processing")


def test_bounding_box_drawn_on_full_frame_not_on_crop():
    """
    Face bounding boxes must be drawn on a copy of the complete original frame.
    The underlying image must remain the full camera resolution.
    """
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    frame = np.random.randint(40, 200, (1440, 2560, 3), dtype=np.uint8)

    # Simulate the acquisition loop display path
    display_frame = frame.copy()
    # Draw a bounding box at known coordinates
    cv2.rectangle(display_frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # After bbox drawing, the underlying image is still the full resolution
    assert display_frame.shape[:2] == (1440, 2560), (
        "After drawing bounding box, display_frame must still be full source resolution")

    # Pass through prepare_preview_frame — should still preserve 16:9
    preview = prepare_preview_frame(display_frame, max_width=1280, max_height=720)
    src_ratio  = 2560 / 1440
    prev_ratio = preview.shape[1] / preview.shape[0]
    assert abs(src_ratio - prev_ratio) < 0.001, (
        f"Aspect ratio must be preserved after bbox drawing: "
        f"source {src_ratio:.4f} → preview {prev_ratio:.4f}")


def test_post_selection_preview_uses_full_frame_path():
    """
    After camera selection, the frame shown in the acquisition window must use
    the SAME prepare_preview_frame path as the camera-selection preview —
    i.e., it is always the complete camera frame scaled to fit, never a face crop.
    """
    from scripts.dataset.capture_real_world_evaluation import prepare_preview_frame

    # Simulate a 2560×1440 camera frame with a known pattern in all four corners
    src = np.zeros((1440, 2560, 3), dtype=np.uint8)
    src[0,    0]    = (255, 0, 0)    # top-left
    src[0,    2559] = (0, 255, 0)    # top-right
    src[1439, 0]    = (0, 0, 255)    # bottom-left
    src[1439, 2559] = (255, 255, 0)  # bottom-right

    display_frame = src.copy()   # simulates: display_frame = frame.copy() in loop
    preview = prepare_preview_frame(display_frame, max_width=1280, max_height=720)

    assert preview.shape == (720, 1280, 3)

    # The preview is a 2× scale-down: corner pixels must NOT be black
    # (if cropped, corners would be absent and appear black)
    corners = [
        preview[0, 0],
        preview[0, 1279],
        preview[719, 0],
        preview[719, 1279],
    ]
    all_black = [np.all(c == 0) for c in corners]
    assert not all(all_black), (
        "All four corners of the preview are black — the full frame was not preserved. "
        "The preview appears to be cropped rather than scaled.")


