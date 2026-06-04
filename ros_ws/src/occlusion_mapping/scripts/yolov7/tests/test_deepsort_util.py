"""
Tests for softmax, softmin, and draw_bbox in deepsort_util.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsort_util import softmax, softmin


class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.array([1., 2., 3., 4.])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_higher_input_gets_higher_probability(self):
        x = np.array([1., 2., 3.])
        result = softmax(x)
        assert result[2] > result[1] > result[0]

    def test_uniform_input_gives_uniform_output(self):
        x = np.ones(5)
        result = softmax(x)
        np.testing.assert_allclose(result, np.full(5, 0.2), atol=1e-6)

    def test_output_all_positive(self):
        x = np.array([-2., -1., 0., 1., 2.])
        result = softmax(x)
        assert np.all(result > 0.)

    def test_raises_on_non_array(self):
        with pytest.raises(AssertionError):
            softmax([1., 2., 3.])

    def test_single_element_returns_one(self):
        x = np.array([42.])
        result = softmax(x)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)


class TestSoftmin:
    def test_output_sums_to_one(self):
        x = np.array([1., 2., 3., 4.])
        result = softmin(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_lower_input_gets_higher_weight(self):
        x = np.array([1., 2., 3.])
        result = softmin(x)
        assert result[0] > result[1] > result[2]

    def test_uniform_input_gives_uniform_output(self):
        x = np.ones(4)
        result = softmin(x)
        np.testing.assert_allclose(result, np.full(4, 0.25), atol=1e-6)

    def test_output_all_positive(self):
        x = np.array([0., 1., 2., 3.])
        result = softmin(x)
        assert np.all(result > 0.)

    def test_raises_on_non_array(self):
        with pytest.raises(AssertionError):
            softmin([1., 2., 3.])

    def test_softmax_and_softmin_complement_each_other(self):
        # softmin(x) == softmax(-x) conceptually (with different temperature)
        # They should reverse the ordering of a monotone input
        x = np.array([1., 2., 3., 4., 5.])
        sm_max = softmax(x)
        sm_min = softmin(x)
        # Largest element in softmax corresponds to smallest in softmin
        assert np.argmax(sm_max) == np.argmin(sm_min)
        assert np.argmin(sm_max) == np.argmax(sm_min)
