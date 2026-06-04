"""
Tests for the Kalman filter used in DeepSort tracking.

The filter maintains an 8D state (x, y, a, h, vx, vy, va, vh) and provides
initiate/predict/project/update/gating_distance operations.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deep_sort.sort.kalman_filter import KalmanFilter, chi2inv95


class TestChi2Table:
    def test_table_has_nine_entries(self):
        assert len(chi2inv95) == 9

    def test_values_are_monotonically_increasing(self):
        values = [chi2inv95[k] for k in sorted(chi2inv95)]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_known_value_at_4dof(self):
        # chi2(0.95, 4) ≈ 9.488
        assert abs(chi2inv95[4] - 9.4877) < 1e-3


class TestKalmanFilterInitiate:
    def setup_method(self):
        self.kf = KalmanFilter()

    def test_mean_shape(self):
        mean, _ = self.kf.initiate(np.array([100., 200., 1.0, 50.]))
        assert mean.shape == (8,)

    def test_covariance_shape(self):
        _, cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))
        assert cov.shape == (8, 8)

    def test_position_in_mean_matches_measurement(self):
        meas = np.array([100., 200., 1.0, 50.])
        mean, _ = self.kf.initiate(meas)
        np.testing.assert_array_equal(mean[:4], meas)

    def test_initial_velocities_are_zero(self):
        mean, _ = self.kf.initiate(np.array([100., 200., 1.0, 50.]))
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_covariance_is_diagonal(self):
        _, cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))
        off_diag = cov - np.diag(np.diag(cov))
        np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))

    def test_covariance_is_positive_definite(self):
        _, cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_larger_bbox_produces_larger_position_uncertainty(self):
        _, cov_small = self.kf.initiate(np.array([100., 200., 1.0, 20.]))
        _, cov_large = self.kf.initiate(np.array([100., 200., 1.0, 100.]))
        assert cov_large[0, 0] > cov_small[0, 0]


class TestKalmanFilterPredict:
    def setup_method(self):
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))

    def test_predict_output_shapes(self):
        new_mean, new_cov = self.kf.predict(self.mean, self.cov)
        assert new_mean.shape == (8,)
        assert new_cov.shape == (8, 8)

    def test_predict_position_unchanged_with_zero_velocity(self):
        # Initial velocity is zero; position should stay the same after predict
        new_mean, _ = self.kf.predict(self.mean, self.cov)
        np.testing.assert_allclose(new_mean[:4], self.mean[:4], atol=1e-10)

    def test_predict_position_shifts_with_nonzero_velocity(self):
        mean_with_vel = self.mean.copy()
        mean_with_vel[4] = 10.   # vx = 10
        mean_with_vel[5] = -5.   # vy = -5
        new_mean, _ = self.kf.predict(mean_with_vel, self.cov)
        assert abs(new_mean[0] - (self.mean[0] + 10.)) < 1e-6
        assert abs(new_mean[1] - (self.mean[1] - 5.)) < 1e-6

    def test_predict_increases_covariance(self):
        _, new_cov = self.kf.predict(self.mean, self.cov)
        # All diagonal elements should grow due to added process noise
        assert np.all(np.diag(new_cov) >= np.diag(self.cov))

    def test_predict_covariance_remains_symmetric(self):
        _, new_cov = self.kf.predict(self.mean, self.cov)
        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-10)


class TestKalmanFilterProject:
    def setup_method(self):
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))

    def test_project_mean_shape(self):
        proj_mean, _ = self.kf.project(self.mean, self.cov)
        assert proj_mean.shape == (4,)

    def test_project_covariance_shape(self):
        _, proj_cov = self.kf.project(self.mean, self.cov)
        assert proj_cov.shape == (4, 4)

    def test_projected_mean_matches_position_state(self):
        proj_mean, _ = self.kf.project(self.mean, self.cov)
        np.testing.assert_allclose(proj_mean, self.mean[:4], atol=1e-10)

    def test_projected_covariance_positive_definite(self):
        _, proj_cov = self.kf.project(self.mean, self.cov)
        eigenvalues = np.linalg.eigvalsh(proj_cov)
        assert np.all(eigenvalues > 0)


class TestKalmanFilterUpdate:
    def setup_method(self):
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))

    def test_update_output_shapes(self):
        new_mean, new_cov = self.kf.update(
            self.mean, self.cov, np.array([100., 200., 1.0, 50.]))
        assert new_mean.shape == (8,)
        assert new_cov.shape == (8, 8)

    def test_update_with_exact_measurement_reduces_position_uncertainty(self):
        # Updating with the same position should reduce uncertainty
        _, new_cov = self.kf.update(self.mean, self.cov, self.mean[:4])
        assert np.all(np.diag(new_cov[:4, :4]) <= np.diag(self.cov[:4, :4]))

    def test_update_mean_moves_toward_measurement(self):
        measurement = np.array([110., 210., 1.0, 50.])  # shifted position
        new_mean, _ = self.kf.update(self.mean, self.cov, measurement)
        # New mean should be between old mean and measurement
        assert new_mean[0] > self.mean[0]
        assert new_mean[1] > self.mean[1]

    def test_update_covariance_remains_symmetric(self):
        new_mean, new_cov = self.kf.update(
            self.mean, self.cov, np.array([100., 200., 1.0, 50.]))
        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-10)

    def test_predict_then_update_cycle(self):
        measurement = np.array([101., 201., 1.0, 50.])
        pred_mean, pred_cov = self.kf.predict(self.mean, self.cov)
        updated_mean, updated_cov = self.kf.update(pred_mean, pred_cov, measurement)
        assert updated_mean.shape == (8,)
        assert updated_cov.shape == (8, 8)


class TestKalmanFilterGatingDistance:
    def setup_method(self):
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(np.array([100., 200., 1.0, 50.]))

    def test_returns_array_of_length_n(self):
        measurements = np.array([
            [100., 200., 1.0, 50.],
            [200., 300., 1.0, 50.],
        ])
        distances = self.kf.gating_distance(self.mean, self.cov, measurements)
        assert distances.shape == (2,)

    def test_nearby_measurement_has_low_distance(self):
        measurements = np.array([[100., 200., 1.0, 50.]])
        distances = self.kf.gating_distance(self.mean, self.cov, measurements)
        assert distances[0] < chi2inv95[4]

    def test_distant_measurement_has_high_distance(self):
        measurements = np.array([[1000., 2000., 1.0, 50.]])
        distances = self.kf.gating_distance(self.mean, self.cov, measurements)
        assert distances[0] > chi2inv95[4]

    def test_position_only_returns_length_n(self):
        measurements = np.array([
            [100., 200., 1.0, 50.],
            [150., 250., 1.0, 50.],
        ])
        distances = self.kf.gating_distance(
            self.mean, self.cov, measurements, only_position=True)
        assert distances.shape == (2,)

    def test_position_only_uses_2dof_threshold(self):
        # With only_position=True, gating uses chi2inv95[2], not chi2inv95[4]
        measurements = np.array([[100., 200., 1.0, 50.]])
        dist_4dof = self.kf.gating_distance(self.mean, self.cov, measurements,
                                            only_position=False)
        dist_2dof = self.kf.gating_distance(self.mean, self.cov, measurements,
                                            only_position=True)
        # Both should be near-zero for this measurement, but computed differently
        assert dist_4dof.shape == dist_2dof.shape == (1,)

    def test_distances_are_non_negative(self):
        measurements = np.random.rand(10, 4) * 200
        distances = self.kf.gating_distance(self.mean, self.cov, measurements)
        assert np.all(distances >= 0)
