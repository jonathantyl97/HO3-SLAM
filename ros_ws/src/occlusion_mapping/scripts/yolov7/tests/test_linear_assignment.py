"""
Tests for min_cost_matching, matching_cascade, and gate_cost_matrix
in deep_sort/sort/linear_assignment.py.
"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.linear_assignment import (
    min_cost_matching, matching_cascade, gate_cost_matrix, INFTY_COST
)


def _dummy_metric(tracks, detections, track_indices, detection_indices):
    """Returns an identity-like cost matrix: 0 on diagonal, 1 elsewhere."""
    n = len(track_indices)
    m = len(detection_indices)
    cost = np.ones((n, m))
    for i in range(min(n, m)):
        cost[i, i] = 0.0
    return cost


def _make_track(time_since_update=1):
    t = MagicMock()
    t.time_since_update = time_since_update
    return t


class TestMinCostMatching:
    def test_empty_tracks_returns_empty_matches(self):
        matches, unmatched_tracks, unmatched_dets = min_cost_matching(
            _dummy_metric, 0.5, [], [MagicMock()])
        assert matches == []
        assert len(unmatched_dets) == 1

    def test_empty_detections_returns_empty_matches(self):
        matches, unmatched_tracks, unmatched_dets = min_cost_matching(
            _dummy_metric, 0.5, [MagicMock()], [])
        assert matches == []
        assert len(unmatched_tracks) == 1

    def test_perfect_one_to_one_match(self):
        tracks = [MagicMock(), MagicMock()]
        dets = [MagicMock(), MagicMock()]
        matches, unmatched_t, unmatched_d = min_cost_matching(
            _dummy_metric, 0.5, tracks, dets)
        assert len(matches) == 2
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_high_cost_prevents_match(self):
        def all_high_metric(tracks, dets, t_idx, d_idx):
            n, m = len(t_idx), len(d_idx)
            return np.full((n, m), INFTY_COST)

        tracks = [MagicMock()]
        dets = [MagicMock()]
        matches, unmatched_t, unmatched_d = min_cost_matching(
            all_high_metric, 0.5, tracks, dets)
        assert len(matches) == 0
        assert len(unmatched_t) == 1
        assert len(unmatched_d) == 1

    def test_output_indices_partition_inputs(self):
        tracks = [MagicMock()] * 3
        dets = [MagicMock()] * 2
        matches, unmatched_t, unmatched_d = min_cost_matching(
            _dummy_metric, 0.5, tracks, dets)
        all_track_indices = set(t for t, _ in matches) | set(unmatched_t)
        all_det_indices = set(d for _, d in matches) | set(unmatched_d)
        assert all_track_indices == {0, 1, 2}
        assert all_det_indices == {0, 1}

    def test_custom_track_detection_indices(self):
        tracks = [MagicMock()] * 5
        dets = [MagicMock()] * 5
        matches, unmatched_t, unmatched_d = min_cost_matching(
            _dummy_metric, 0.5, tracks, dets,
            track_indices=[1, 3],
            detection_indices=[2, 4])
        assert all(t in [1, 3] for t, _ in matches)
        assert all(d in [2, 4] for _, d in matches)

    def test_max_distance_threshold_respected(self):
        def cost_metric(tracks, dets, t_idx, d_idx):
            n, m = len(t_idx), len(d_idx)
            cost = np.zeros((n, m))
            cost[0, 0] = 0.3  # below threshold
            cost[0, 1] = 0.8  # above threshold
            return cost

        tracks = [MagicMock()]
        dets = [MagicMock(), MagicMock()]
        matches, _, unmatched_d = min_cost_matching(
            cost_metric, 0.5, tracks, dets)
        matched_dets = {d for _, d in matches}
        assert 0 in matched_dets       # cost 0.3 < 0.5 → matched
        assert 1 in unmatched_d        # cost 0.8 > 0.5 → not matched


class TestMatchingCascade:
    def test_empty_detections_returns_empty_matches(self):
        tracks = [_make_track(time_since_update=1)]
        matches, unmatched_t, unmatched_d = matching_cascade(
            _dummy_metric, 0.5, cascade_depth=5, tracks=tracks, detections=[])
        assert matches == []
        assert 0 in unmatched_t

    def test_empty_tracks_returns_all_unmatched_detections(self):
        dets = [MagicMock(), MagicMock()]
        matches, unmatched_t, unmatched_d = matching_cascade(
            _dummy_metric, 0.5, cascade_depth=5, tracks=[], detections=dets)
        assert matches == []
        assert len(unmatched_d) == 2

    def test_matches_track_at_correct_cascade_level(self):
        # Track with time_since_update=1 should be matched at level 0
        tracks = [_make_track(time_since_update=1), _make_track(time_since_update=3)]
        dets = [MagicMock(), MagicMock()]
        matches, unmatched_t, unmatched_d = matching_cascade(
            _dummy_metric, 0.5, cascade_depth=5, tracks=tracks, detections=dets)
        matched_track_ids = {t for t, _ in matches}
        assert 0 in matched_track_ids  # time_since_update=1 → matched at level 0

    def test_output_partitions_all_inputs(self):
        tracks = [_make_track(1), _make_track(2)]
        dets = [MagicMock(), MagicMock()]
        matches, unmatched_t, unmatched_d = matching_cascade(
            _dummy_metric, 0.5, cascade_depth=5, tracks=tracks, detections=dets)
        all_t = {t for t, _ in matches} | set(unmatched_t)
        all_d = {d for _, d in matches} | set(unmatched_d)
        assert all_t == {0, 1}
        assert all_d == {0, 1}

    def test_cascade_stops_when_no_detections_remain(self):
        # Two tracks both at level 1; only one detection.
        # After first match, no detections remain and cascade should stop early.
        tracks = [_make_track(1), _make_track(1)]
        dets = [MagicMock()]
        matches, unmatched_t, unmatched_d = matching_cascade(
            _dummy_metric, 0.5, cascade_depth=10, tracks=tracks, detections=dets)
        assert len(matches) == 1
        assert len(unmatched_d) == 0


class TestGateCostMatrix:
    def _make_kf_and_detection(self, near=True):
        from deep_sort.sort.kalman_filter import KalmanFilter, chi2inv95
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100., 200., 1.0, 50.]))
        track = MagicMock()
        track.mean = mean
        track.covariance = cov
        if near:
            det = MagicMock()
            det.to_xyah.return_value = np.array([100., 200., 1.0, 50.])
        else:
            det = MagicMock()
            det.to_xyah.return_value = np.array([5000., 5000., 1.0, 50.])
        return kf, track, det, mean, cov

    def test_nearby_detection_not_gated(self):
        from deep_sort.sort.kalman_filter import KalmanFilter, chi2inv95
        kf, track, det, _, _ = self._make_kf_and_detection(near=True)
        cost_matrix = np.zeros((1, 1))
        result = gate_cost_matrix(kf, cost_matrix, [track], [det], [0], [0])
        # Near detection should NOT be set to INFTY_COST
        assert result[0, 0] < INFTY_COST

    def test_distant_detection_gets_gated(self):
        from deep_sort.sort.kalman_filter import KalmanFilter
        kf, track, det, _, _ = self._make_kf_and_detection(near=False)
        cost_matrix = np.zeros((1, 1))
        result = gate_cost_matrix(kf, cost_matrix, [track], [det], [0], [0])
        assert result[0, 0] == INFTY_COST

    def test_matrix_shape_preserved(self):
        from deep_sort.sort.kalman_filter import KalmanFilter
        kf, track, det, _, _ = self._make_kf_and_detection(near=True)
        cost_matrix = np.zeros((1, 2))
        det2 = MagicMock()
        det2.to_xyah.return_value = np.array([100., 200., 1.0, 50.])
        result = gate_cost_matrix(
            kf, cost_matrix, [track], [det, det2], [0], [0, 1])
        assert result.shape == (1, 2)
