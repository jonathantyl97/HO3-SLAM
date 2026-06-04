"""
Tests for IoU computation and cost matrix building in deep_sort/sort/iou_matching.py.
"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.iou_matching import iou, iou_cost
from deep_sort.sort.linear_assignment import INFTY_COST


def _make_detection(tlwh):
    det = MagicMock()
    det.tlwh = np.asarray(tlwh, dtype=float)
    return det


def _make_track(tlwh, time_since_update=0):
    track = MagicMock()
    track.to_tlwh.return_value = np.asarray(tlwh, dtype=float)
    track.time_since_update = time_since_update
    return track


class TestIoU:
    def test_identical_boxes_return_one(self):
        bbox = np.array([0., 0., 10., 10.])
        candidates = np.array([[0., 0., 10., 10.]])
        result = iou(bbox, candidates)
        np.testing.assert_allclose(result, [1.0])

    def test_non_overlapping_boxes_return_zero(self):
        bbox = np.array([0., 0., 10., 10.])
        candidates = np.array([[20., 20., 10., 10.]])
        result = iou(bbox, candidates)
        np.testing.assert_allclose(result, [0.0])

    def test_half_overlap_returns_correct_value(self):
        # bbox: [0,0,10,10] (area 100), candidate: [5,0,10,10] (area 100)
        # intersection: [5,0,5,10] = area 50, union = 150
        bbox = np.array([0., 0., 10., 10.])
        candidates = np.array([[5., 0., 10., 10.]])
        result = iou(bbox, candidates)
        np.testing.assert_allclose(result, [50. / 150.], atol=1e-6)

    def test_contained_box_returns_correct_value(self):
        # Small box entirely within large box
        # large: [0,0,10,10] area=100, small: [2,2,6,6] area=36
        # intersection = 36, union = 100
        bbox = np.array([0., 0., 10., 10.])
        candidates = np.array([[2., 2., 6., 6.]])
        result = iou(bbox, candidates)
        np.testing.assert_allclose(result, [36. / 100.], atol=1e-6)

    def test_multiple_candidates(self):
        bbox = np.array([0., 0., 10., 10.])
        candidates = np.array([
            [0., 0., 10., 10.],   # identical → 1.0
            [20., 20., 10., 10.], # no overlap → 0.0
        ])
        result = iou(bbox, candidates)
        assert result.shape == (2,)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.0)

    def test_iou_output_in_unit_interval(self):
        rng = np.random.default_rng(42)
        bbox = np.array([50., 50., 20., 20.])
        candidates = rng.uniform(0, 100, size=(20, 4))
        candidates[:, 2:] = np.abs(candidates[:, 2:]) + 1  # positive w, h
        result = iou(bbox, candidates)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetry(self):
        # iou(A, [B]) should equal iou(B, [A])
        a = np.array([10., 10., 20., 20.])
        b = np.array([[20., 10., 20., 20.]])
        iou_ab = iou(a, b)[0]
        iou_ba = iou(b[0], np.array([a]))[0]
        np.testing.assert_allclose(iou_ab, iou_ba)


class TestIouCost:
    def test_returns_matrix_of_correct_shape(self):
        tracks = [_make_track([0, 0, 10, 10])] * 3
        detections = [_make_detection([5, 5, 10, 10])] * 2
        cost = iou_cost(tracks, detections)
        assert cost.shape == (3, 2)

    def test_stale_track_gets_infty_cost(self):
        tracks = [_make_track([0, 0, 10, 10], time_since_update=2)]
        detections = [_make_detection([0, 0, 10, 10])]
        cost = iou_cost(tracks, detections)
        assert cost[0, 0] == INFTY_COST

    def test_fresh_track_gets_real_iou_cost(self):
        tracks = [_make_track([0, 0, 10, 10], time_since_update=0)]
        detections = [_make_detection([0, 0, 10, 10])]
        cost = iou_cost(tracks, detections)
        # identical boxes: cost = 1 - 1.0 = 0.0
        np.testing.assert_allclose(cost[0, 0], 0.0)

    def test_costs_are_in_zero_one_range_for_fresh_tracks(self):
        tracks = [_make_track([0, 0, 10, 10], time_since_update=0)]
        detections = [
            _make_detection([0, 0, 10, 10]),   # perfect overlap → cost 0
            _make_detection([50, 50, 10, 10]),  # no overlap → cost 1
        ]
        cost = iou_cost(tracks, detections)
        np.testing.assert_allclose(cost[0, 0], 0.0)
        np.testing.assert_allclose(cost[0, 1], 1.0)

    def test_subset_of_track_and_detection_indices(self):
        tracks = [_make_track([0, 0, 10, 10])] * 4
        detections = [_make_detection([0, 0, 10, 10])] * 4
        cost = iou_cost(tracks, detections,
                        track_indices=[0, 2],
                        detection_indices=[1, 3])
        assert cost.shape == (2, 2)
