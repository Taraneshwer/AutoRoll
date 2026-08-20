"""
Unit tests for CompleteSystemProfiler.
"""

from autoroll.common.profiler import CompleteSystemProfiler, SystemStageLatencies


def test_profiler_sample_recording():
    profiler = CompleteSystemProfiler(window_size=10)

    for i in range(5):
        sample = SystemStageLatencies(
            camera_decode_ms=1.2,
            frame_acquisition_ms=0.5,
            detection_ms=5.0,
            tracking_ms=0.8,
            recognition_ms=4.0,
            liveness_ms=2.0,
            result_transmission_ms=0.4,
            server_processing_ms=0.5,
            websocket_delivery_ms=0.3,
            end_to_end_ms=14.7,
        )
        profiler.record_sample(sample)

    rep = profiler.get_benchmark_report()
    assert rep["sample_count"] == 5
    assert rep["latencies"]["detection_ms"]["mean"] == 5.0
    assert rep["latencies"]["end_to_end_ms"]["mean"] == 14.7
