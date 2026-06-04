"""
Tests for nearest-neighbor distance metric utilities in deep_sort/sort/nn_matching.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.nn_matching import (
    _pdist, _cosine_distance, _nn_euclidean_distance,
    _nn_cosine_distance, NearestNeighborDistanceMetric
)


class TestPdist:
    def test_self_distance_is_zero(self):
        a = np.array([[1., 2., 3.]])
        result = _pdist(a, a)
        np.testing.assert_allclose(result, [[0.]], atol=1e-10)

    def test_known_squared_distance(self):
        a = np.array([[0., 0.]])
        b = np.array([[3., 4.]])
        result = _pdist(a, b)
        np.testing.assert_allclose(result, [[25.]], atol=1e-10)

    def test_output_shape(self):
        a = np.random.rand(5, 4)
        b = np.random.rand(7, 4)
        result = _pdist(a, b)
        assert result.shape == (5, 7)

    def test_empty_a_returns_empty(self):
        result = _pdist(np.zeros((0, 3)), np.ones((4, 3)))
        assert result.shape == (0, 4)

    def test_empty_b_returns_empty(self):
        result = _pdist(np.ones((4, 3)), np.zeros((0, 3)))
        assert result.shape == (4, 0)

    def test_non_negative_distances(self):
        a = np.random.rand(10, 8)
        b = np.random.rand(10, 8)
        result = _pdist(a, b)
        assert np.all(result >= 0.)


class TestCosineDistance:
    def test_identical_vectors_have_zero_distance(self):
        v = np.array([[1., 0., 0.]])
        result = _cosine_distance(v, v)
        np.testing.assert_allclose(result, [[0.]], atol=1e-6)

    def test_orthogonal_vectors_have_distance_one(self):
        a = np.array([[1., 0.]])
        b = np.array([[0., 1.]])
        result = _cosine_distance(a, b)
        np.testing.assert_allclose(result, [[1.]], atol=1e-6)

    def test_opposite_vectors_have_distance_two(self):
        a = np.array([[1., 0.]])
        b = np.array([[-1., 0.]])
        result = _cosine_distance(a, b)
        np.testing.assert_allclose(result, [[2.]], atol=1e-6)

    def test_normalized_flag_gives_same_result(self):
        a = np.array([[3., 4.]])
        b = np.array([[5., 0.]])
        result_unnorm = _cosine_distance(a, b, data_is_normalized=False)
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        result_norm = _cosine_distance(a_norm, b_norm, data_is_normalized=True)
        np.testing.assert_allclose(result_unnorm, result_norm, atol=1e-6)

    def test_output_shape(self):
        a = np.random.rand(3, 5)
        b = np.random.rand(7, 5)
        result = _cosine_distance(a, b)
        assert result.shape == (3, 7)


class TestNnEuclideanDistance:
    def test_exact_match_returns_zero(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[1., 2., 3.]])
        result = _nn_euclidean_distance(x, y)
        np.testing.assert_allclose(result, [0.], atol=1e-10)

    def test_returns_minimum_distance(self):
        x = np.array([[0., 0.], [10., 10.]])
        y = np.array([[1., 1.]])  # closer to origin
        result = _nn_euclidean_distance(x, y)
        # Squared distance to origin = 2, to (10,10) = 162
        np.testing.assert_allclose(result, [2.], atol=1e-6)

    def test_non_negative_output(self):
        x = np.random.rand(5, 4)
        y = np.random.rand(8, 4)
        result = _nn_euclidean_distance(x, y)
        assert np.all(result >= 0.)


class TestNnCosineDistance:
    def test_exact_match_returns_near_zero(self):
        v = np.array([[1., 0., 0.]])
        result = _nn_cosine_distance(v, v)
        np.testing.assert_allclose(result, [0.], atol=1e-6)

    def test_returns_minimum_across_gallery(self):
        x = np.array([[1., 0.], [0., 1.]])
        y = np.array([[1., 0.]])  # identical to first gallery entry
        result = _nn_cosine_distance(x, y)
        np.testing.assert_allclose(result, [0.], atol=1e-6)


class TestNearestNeighborDistanceMetric:
    def test_invalid_metric_raises_value_error(self):
        with pytest.raises(ValueError):
            NearestNeighborDistanceMetric("manhattan", 0.5)

    def test_euclidean_metric_created(self):
        m = NearestNeighborDistanceMetric("euclidean", 0.5)
        assert m._metric is _nn_euclidean_distance

    def test_cosine_metric_created(self):
        m = NearestNeighborDistanceMetric("cosine", 0.3)
        assert m._metric is _nn_cosine_distance

    def test_partial_fit_adds_features(self):
        m = NearestNeighborDistanceMetric("cosine", 0.3)
        features = np.random.rand(3, 8)
        targets = np.array([1, 2, 3])
        m.partial_fit(features, targets, active_targets=[1, 2, 3])
        assert set(m.samples.keys()) == {1, 2, 3}

    def test_partial_fit_removes_inactive_targets(self):
        m = NearestNeighborDistanceMetric("cosine", 0.3)
        features = np.random.rand(3, 8)
        targets = np.array([1, 2, 3])
        m.partial_fit(features, targets, active_targets=[1, 2, 3])
        # Now only target 1 is active
        m.partial_fit(np.random.rand(1, 8), np.array([1]), active_targets=[1])
        assert set(m.samples.keys()) == {1}

    def test_budget_limits_stored_samples(self):
        budget = 3
        m = NearestNeighborDistanceMetric("cosine", 0.3, budget=budget)
        for i in range(10):
            m.partial_fit(np.random.rand(1, 8), np.array([42]),
                          active_targets=[42])
        assert len(m.samples[42]) <= budget

    def test_distance_returns_correct_shape(self):
        m = NearestNeighborDistanceMetric("euclidean", 0.5)
        features = np.random.rand(4, 8)
        targets = np.array([10, 20, 10, 20])
        m.partial_fit(features, targets, active_targets=[10, 20])
        query = np.random.rand(5, 8)
        cost = m.distance(query, targets=[10, 20])
        assert cost.shape == (2, 5)

    def test_distance_zero_for_stored_feature(self):
        m = NearestNeighborDistanceMetric("euclidean", 0.5)
        feat = np.array([[1., 0., 0., 0.]])
        m.partial_fit(feat, np.array([1]), active_targets=[1])
        cost = m.distance(feat, targets=[1])
        np.testing.assert_allclose(cost[0, 0], 0., atol=1e-6)
