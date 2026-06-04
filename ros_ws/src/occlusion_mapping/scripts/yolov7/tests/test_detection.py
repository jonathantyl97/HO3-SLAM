"""
Tests for Detection class and preprocessing.non_max_suppression.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.detection import Detection
from deep_sort.sort.preprocessing import non_max_suppression


class TestDetection:
    def _det(self, tlwh=(10., 20., 30., 40.), conf=0.9):
        return Detection(tlwh, conf, np.zeros(128))

    def test_to_tlbr_adds_wh_to_xy(self):
        d = self._det(tlwh=(10., 20., 30., 40.))
        tlbr = d.to_tlbr()
        np.testing.assert_allclose(tlbr, [10., 20., 40., 60.])

    def test_to_xyah_center_x(self):
        d = self._det(tlwh=(10., 20., 30., 40.))
        xyah = d.to_xyah()
        # center_x = 10 + 30/2 = 25
        np.testing.assert_allclose(xyah[0], 25., atol=1e-6)

    def test_to_xyah_center_y(self):
        d = self._det(tlwh=(10., 20., 30., 40.))
        xyah = d.to_xyah()
        # center_y = 20 + 40/2 = 40
        np.testing.assert_allclose(xyah[1], 40., atol=1e-6)

    def test_to_xyah_aspect_ratio(self):
        d = self._det(tlwh=(0., 0., 30., 40.))
        xyah = d.to_xyah()
        # aspect ratio = width/height = 30/40 = 0.75
        np.testing.assert_allclose(xyah[2], 0.75, atol=1e-6)

    def test_to_xyah_height_unchanged(self):
        d = self._det(tlwh=(0., 0., 30., 40.))
        xyah = d.to_xyah()
        np.testing.assert_allclose(xyah[3], 40., atol=1e-6)

    def test_confidence_stored_as_float(self):
        d = Detection([0, 0, 10, 10], 0.75, np.zeros(8))
        assert isinstance(d.confidence, float)
        assert abs(d.confidence - 0.75) < 1e-6

    def test_feature_stored_as_float32(self):
        feat = np.array([1., 2., 3.], dtype=np.float64)
        d = Detection([0, 0, 10, 10], 0.9, feat)
        assert d.feature.dtype == np.float32

    def test_square_box_aspect_ratio_is_one(self):
        d = Detection([0., 0., 50., 50.], 0.8, np.zeros(4))
        xyah = d.to_xyah()
        np.testing.assert_allclose(xyah[2], 1.0, atol=1e-6)


class TestNonMaxSuppression:
    def test_empty_boxes_return_empty(self):
        result = non_max_suppression(np.zeros((0, 4)), max_bbox_overlap=0.5)
        assert result == []

    def test_single_box_always_kept(self):
        boxes = np.array([[0., 0., 10., 10.]])
        result = non_max_suppression(boxes, max_bbox_overlap=0.5)
        assert len(result) == 1
        assert result[0] == 0

    def test_non_overlapping_boxes_all_kept(self):
        boxes = np.array([
            [0., 0., 10., 10.],
            [20., 0., 10., 10.],
            [40., 0., 10., 10.],
        ])
        result = non_max_suppression(boxes, max_bbox_overlap=0.5)
        assert len(result) == 3

    def test_heavily_overlapping_boxes_suppressed(self):
        # Two nearly identical boxes; only one should survive
        boxes = np.array([
            [0., 0., 10., 10.],
            [1., 1., 10., 10.],
        ])
        result = non_max_suppression(boxes, max_bbox_overlap=0.3)
        assert len(result) == 1

    def test_confidence_scores_respected(self):
        boxes = np.array([
            [0., 0., 10., 10.],
            [1., 1., 10., 10.],
        ])
        scores = np.array([0.3, 0.9])
        result = non_max_suppression(boxes, max_bbox_overlap=0.3, scores=scores)
        # Higher-confidence box (index 1) should be kept
        assert 1 in result

    def test_zero_overlap_threshold_keeps_one(self):
        boxes = np.array([
            [0., 0., 10., 10.],
            [5., 5., 10., 10.],  # overlaps significantly
        ])
        result = non_max_suppression(boxes, max_bbox_overlap=0.0)
        assert len(result) == 1

    def test_result_indices_are_valid(self):
        boxes = np.random.rand(10, 4)
        boxes[:, 2:] = np.abs(boxes[:, 2:]) + 1
        result = non_max_suppression(boxes, max_bbox_overlap=0.5)
        assert all(0 <= i < len(boxes) for i in result)
