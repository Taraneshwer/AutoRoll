"""
Unit tests for DistributedScalingBenchmark.
"""

from autoroll.common.scaling_benchmark import DistributedScalingBenchmark


def test_distributed_scaling_benchmark():
    bench = DistributedScalingBenchmark(num_cameras=4, frames_per_camera=10)
    results = bench.run_full_suite()

    assert len(results) == 4
    assert results[0].num_workers == 1
    assert results[3].num_workers == 4

    for r in results:
        assert r.total_throughput_fps > 0
        assert r.avg_latency_ms > 0
        assert r.p95_latency_ms >= r.avg_latency_ms
        assert r.scaling_efficiency_percent > 0
