"""
Tests for the Track class state machine and coordinate conversions.

State transitions:
  Tentative → Confirmed  after n_init consecutive updates
  Tentative → Deleted    on the first missed detection
  Confirmed → Deleted    after max_age frames without an update
"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.track import Track, TrackState
from deep_sort.sort.kalman_filter import KalmanFilter


def _make_track(x=100., y=200., a=1.0, h=50., n_init=3, max_age=5,
                feature=None):
    kf = KalmanFilter()
    mean, covariance = kf.initiate(np.array([x, y, a, h]))
    return Track(mean, covariance, track_id=1,
                 n_init=n_init, max_age=max_age, feature=feature)


def _make_detection(x=100., y=200., a=1.0, h=50.):
    det = MagicMock()
    det.to_xyah.return_value = np.array([x, y, a, h])
    det.feature = np.zeros(128)
    return det


class TestTrackInitialState:
    def test_initial_state_is_tentative(self):
        t = _make_track()
        assert t.is_tentative()
        assert not t.is_confirmed()
        assert not t.is_deleted()

    def test_initial_hits_is_one(self):
        t = _make_track()
        assert t.hits == 1

    def test_initial_age_is_one(self):
        t = _make_track()
        assert t.age == 1

    def test_initial_time_since_update_is_zero(self):
        t = _make_track()
        assert t.time_since_update == 0

    def test_feature_appended_when_provided(self):
        feat = np.ones(128)
        t = _make_track(feature=feat)
        assert len(t.features) == 1
        np.testing.assert_array_equal(t.features[0], feat)

    def test_no_feature_stored_when_none(self):
        t = _make_track(feature=None)
        assert len(t.features) == 0


class TestTrackStateTransitions:
    def test_tentative_to_deleted_on_miss(self):
        t = _make_track(n_init=3)
        t.mark_missed()
        assert t.is_deleted()

    def test_tentative_to_confirmed_after_n_init_updates(self):
        kf = KalmanFilter()
        t = _make_track(n_init=3)
        for _ in range(2):  # 1 initial hit + 2 updates = 3 total
            t.update(kf, _make_detection())
        assert t.is_confirmed()

    def test_not_confirmed_before_n_init_updates(self):
        kf = KalmanFilter()
        t = _make_track(n_init=3)
        t.update(kf, _make_detection())  # now hits = 2, still tentative
        assert t.is_tentative()

    def test_confirmed_track_not_deleted_before_max_age(self):
        kf = KalmanFilter()
        t = _make_track(n_init=2, max_age=5)
        t.update(kf, _make_detection())  # confirmed (hits=2 >= n_init=2)
        assert t.is_confirmed()
        for _ in range(4):
            t.predict(kf)
            t.mark_missed()
        assert not t.is_deleted()

    def test_confirmed_track_deleted_after_max_age(self):
        kf = KalmanFilter()
        t = _make_track(n_init=2, max_age=3)
        t.update(kf, _make_detection())  # confirmed
        for _ in range(4):
            t.predict(kf)
            t.mark_missed()
        assert t.is_deleted()

    def test_update_resets_time_since_update(self):
        kf = KalmanFilter()
        t = _make_track()
        t.predict(kf)
        assert t.time_since_update == 1
        t.update(kf, _make_detection())
        assert t.time_since_update == 0

    def test_predict_increments_age_and_time_since_update(self):
        kf = KalmanFilter()
        t = _make_track()
        t.predict(kf)
        assert t.age == 2
        assert t.time_since_update == 1

    def test_update_increments_hits(self):
        kf = KalmanFilter()
        t = _make_track()
        initial_hits = t.hits
        t.update(kf, _make_detection())
        assert t.hits == initial_hits + 1

    def test_update_appends_feature(self):
        kf = KalmanFilter()
        t = _make_track()
        t.update(kf, _make_detection())
        assert len(t.features) == 1


class TestTrackCoordinateConversions:
    def test_to_tlwh_returns_four_values(self):
        t = _make_track(x=100., y=200., a=1.0, h=50.)
        ret = t.to_tlwh()
        assert ret.shape == (4,)

    def test_to_tlwh_width_equals_aspect_ratio_times_height(self):
        # When a=2.0, w should equal 2 * h
        t = _make_track(x=100., y=200., a=2.0, h=50.)
        ret = t.to_tlwh()
        w, h = ret[2], ret[3]
        np.testing.assert_allclose(w, 2.0 * h, atol=1e-6)

    def test_to_tlwh_top_left_from_center(self):
        # center (x=100, y=200), w=50, h=50 → tl = (75, 175)
        t = _make_track(x=100., y=200., a=1.0, h=50.)
        ret = t.to_tlwh()
        np.testing.assert_allclose(ret[0], 100. - 50. / 2, atol=1e-6)
        np.testing.assert_allclose(ret[1], 200. - 50. / 2, atol=1e-6)

    def test_to_tlbr_bottom_right_from_top_left_and_size(self):
        t = _make_track(x=100., y=200., a=1.0, h=50.)
        tlwh = t.to_tlwh()
        tlbr = t.to_tlbr()
        np.testing.assert_allclose(tlbr[2], tlwh[0] + tlwh[2], atol=1e-6)
        np.testing.assert_allclose(tlbr[3], tlwh[1] + tlwh[3], atol=1e-6)

    def test_tlbr_br_is_always_geq_tl(self):
        t = _make_track(x=100., y=200., a=1.5, h=60.)
        tlbr = t.to_tlbr()
        assert tlbr[2] >= tlbr[0]
        assert tlbr[3] >= tlbr[1]
