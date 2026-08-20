"""
Unit tests for Temporal Liveness Aggregator module.
"""

from autoroll.ml.liveness.temporal_analyzer import TemporalLivenessAggregator


def test_temporal_aggregator_sliding_window():
    aggregator = TemporalLivenessAggregator(window_size=5)

    # Push 5 spatial scores
    scores = [0.9, 0.85, 0.95, 0.88, 0.92]
    final_score = 0.0
    for s in scores:
        final_score = aggregator.update(s)

    assert 0.8 <= final_score <= 1.0
    assert len(aggregator.spatial_scores_buffer) == 5


def test_temporal_aggregator_sequence_eval():
    aggregator = TemporalLivenessAggregator()

    # Sequence with low spoof score
    spoof_seq = [0.9, 0.85, 0.20, 0.88, 0.90]
    score = aggregator.aggregate_sequence(spoof_seq)

    # Minimum penalty should drop overall score
    assert score < 0.80
