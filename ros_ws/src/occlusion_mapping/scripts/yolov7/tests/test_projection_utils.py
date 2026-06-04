"""
Tests for pure-math utility functions in projection.py.

ROS-dependent functions (publish_mappoints, publish_path) and file I/O
(read_mappoints, read_trajectory) are out of scope here; they should be
covered with integration tests once a ROS environment is available.
"""
import sys
import os
import tempfile
import numpy as np
import pytest
import importlib.util
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load projection.py by file path to avoid triggering other package __init__
_proj_path = os.path.join(os.path.dirname(__file__), '..', 'projection.py')
_spec = importlib.util.spec_from_file_location("projection", _proj_path)
projection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(projection)

get_rotation_vector = projection.get_rotation_vector
invert_pose = projection.invert_pose
project_points = projection.project_points
find_features_to_project = projection.find_features_to_project
read_mappoints = projection.read_mappoints
read_trajectory = projection.read_trajectory

# Grab the cv2 mock that conftest installed so we can configure side effects
_cv2 = sys.modules["cv2"]


class TestGetRotationVector:
    def test_identity_quaternion_calls_rodrigues(self):
        _cv2.Rodrigues.return_value = (np.zeros((3, 1)), None)
        q = np.array([0., 0., 0., 1.])
        get_rotation_vector(q)
        assert _cv2.Rodrigues.called

    def test_handles_unnormalized_quaternion(self):
        _cv2.Rodrigues.return_value = (np.zeros((3, 1)), None)
        q = np.array([0., 0., 0., 2.])
        get_rotation_vector(q)

    def test_output_shape(self):
        fake_rvec = np.array([[0.1], [0.2], [0.3]])
        _cv2.Rodrigues.return_value = (fake_rvec, None)
        q = np.array([0., 0., 0., 1.])
        result = get_rotation_vector(q)
        assert result.shape == (3, 1)


def _real_cv2_or_skip():
    """Return the real cv2 module, or skip the test if it isn't installed."""
    # Our conftest installed a MagicMock under 'cv2'. Try to find the real
    # package by probing for its binary extension directly.
    import importlib.util as _ilu
    import importlib
    # Temporarily remove our mock so find_spec can look for the real thing
    _saved = sys.modules.pop("cv2", None)
    try:
        spec = _ilu.find_spec("cv2")
        if spec is None:
            pytest.skip("real cv2 not available")
        real = importlib.import_module("cv2")
        return real
    except (ValueError, ModuleNotFoundError):
        pytest.skip("real cv2 not available")
    finally:
        if _saved is not None:
            sys.modules["cv2"] = _saved


class TestInvertPose:
    def test_double_inversion_recovers_original_translation(self):
        real_cv2 = _real_cv2_or_skip()
        _cv2.Rodrigues.side_effect = real_cv2.Rodrigues
        rvec = np.array([[0.1], [0.2], [0.3]])
        tvec = np.array([[1.], [2.], [3.]])
        inv_rvec, inv_tvec = invert_pose(rvec, tvec)
        re_rvec, re_tvec = invert_pose(inv_rvec, inv_tvec)
        np.testing.assert_allclose(re_tvec, tvec, atol=1e-6)

    def test_zero_translation_inverts_to_zero(self):
        real_cv2 = _real_cv2_or_skip()
        _cv2.Rodrigues.side_effect = real_cv2.Rodrigues
        rvec = np.array([[0.], [0.], [0.]])
        tvec = np.array([[0.], [0.], [0.]])
        _, inv_tvec = invert_pose(rvec, tvec)
        np.testing.assert_allclose(inv_tvec, tvec, atol=1e-10)


class TestProjectPoints:
    def test_point_in_front_of_camera_is_valid(self):
        real_cv2 = _real_cv2_or_skip()
        _cv2.projectPoints.side_effect = real_cv2.projectPoints
        _cv2.Rodrigues.side_effect = real_cv2.Rodrigues
        K = np.array([[500., 0., 320.],
                      [0., 500., 240.],
                      [0., 0., 1.]], dtype=np.float64)
        D = np.zeros(5)
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        points = np.array([[0., 0., 5.]])
        _, valid_mask = project_points(points, rvec, tvec, K, D)
        assert valid_mask[0] == 1

    def test_point_behind_camera_is_invalid(self):
        real_cv2 = _real_cv2_or_skip()
        _cv2.projectPoints.side_effect = real_cv2.projectPoints
        _cv2.Rodrigues.side_effect = real_cv2.Rodrigues
        K = np.array([[500., 0., 320.],
                      [0., 500., 240.],
                      [0., 0., 1.]], dtype=np.float64)
        D = np.zeros(5)
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        points = np.array([[0., 0., -5.]])
        _, valid_mask = project_points(points, rvec, tvec, K, D)
        assert valid_mask[0] == 0


class TestReadMappoints:
    def test_reads_all_points(self):
        content = "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            pts = read_mappoints(fname, scale=1.0, percentage=100)
            assert len(pts) == 3
        finally:
            os.unlink(fname)

    def test_scale_is_applied(self):
        content = "1.0 2.0 3.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            pts = read_mappoints(fname, scale=2.0, percentage=100)
            np.testing.assert_allclose(pts[0], [2.0, 4.0, 6.0])
        finally:
            os.unlink(fname)

    def test_percentage_zero_returns_empty(self):
        content = "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            pts = read_mappoints(fname, scale=1.0, percentage=0)
            assert len(pts) == 0
        finally:
            os.unlink(fname)

    def test_percentage_clamped_below_zero(self):
        content = "1.0 2.0 3.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            pts = read_mappoints(fname, scale=1.0, percentage=-10)
            assert len(pts) == 0
        finally:
            os.unlink(fname)

    def test_points_sorted_by_z(self):
        content = "0.0 0.0 5.0\n0.0 0.0 1.0\n0.0 0.0 3.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            pts = read_mappoints(fname, scale=1.0, percentage=100)
            z_values = pts[:, 2]
            assert list(z_values) == sorted(z_values)
        finally:
            os.unlink(fname)


class TestReadTrajectory:
    def test_reads_correct_fields(self):
        content = "0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(content)
            fname = f.name
        try:
            traj = read_trajectory(fname)
            assert len(traj) == 1
            kf_id, t, q = traj[0]
            assert kf_id == 0
            np.testing.assert_allclose(t, [1., 2., 3.])
            np.testing.assert_allclose(q, [0., 0., 0., 1.])
        finally:
            os.unlink(fname)

    def test_multiple_frames(self):
        lines = [
            "0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n",
            "1 4.0 5.0 6.0 0.1 0.2 0.3 0.9\n",
        ]
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.writelines(lines)
            fname = f.name
        try:
            traj = read_trajectory(fname)
            assert len(traj) == 2
            assert traj[0][0] == 0
            assert traj[1][0] == 1
        finally:
            os.unlink(fname)
