"""
Unit tests for WebSocket Real-Time Telemetry Protocol and Event Manager.
"""

from server.app.websockets.manager import ConnectionManager
from server.app.websockets.protocol import WebSocketEventEnvelope, WebSocketEventType


def test_websocket_event_envelope():
    mgr = ConnectionManager()
    data = {"camera_id": "cam_01", "status": "ONLINE"}

    envelope = mgr.wrap_event(WebSocketEventType.CAMERA_STATUS_CHANGED, data)

    assert isinstance(envelope, WebSocketEventEnvelope)
    assert envelope.event_type == WebSocketEventType.CAMERA_STATUS_CHANGED
    assert envelope.event_id.startswith("evt_")
    assert envelope.sequence_number == 1
    assert envelope.timestamp_ms > 0
    assert envelope.data["camera_id"] == "cam_01"

    # Verify sequence increment
    envelope2 = mgr.wrap_event(WebSocketEventType.ATTENDANCE_CONFIRMED, {"student_id": "s1"})
    assert envelope2.sequence_number == 2


def test_all_websocket_event_types():
    assert WebSocketEventType.CAMERA_STATUS_CHANGED == "CAMERA_STATUS_CHANGED"
    assert WebSocketEventType.WORKER_STATUS_CHANGED == "WORKER_STATUS_CHANGED"
    assert WebSocketEventType.FACE_DETECTED == "FACE_DETECTED"
    assert WebSocketEventType.RECOGNITION_RESULT == "RECOGNITION_RESULT"
    assert WebSocketEventType.LIVENESS_RESULT == "LIVENESS_RESULT"
    assert WebSocketEventType.ATTENDANCE_CONFIRMED == "ATTENDANCE_CONFIRMED"
    assert WebSocketEventType.SPOOF_ATTEMPT == "SPOOF_ATTEMPT"
    assert WebSocketEventType.SYSTEM_METRICS_UPDATED == "SYSTEM_METRICS_UPDATED"
