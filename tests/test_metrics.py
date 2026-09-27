import pytest

from graph_slam.metrics import final_pose_error, landmark_rmse, pose_rmse


def test_pose_rmse_zero_when_identical():
    poses = [(1.0, 2.0), (3.0, 4.0)]
    assert pose_rmse(poses, poses) == 0.0


def test_pose_rmse_known_value():
    estimated = [(0.0, 0.0), (0.0, 0.0)]
    true = [(3.0, 4.0), (3.0, 4.0)]  # each point is distance 5 away
    assert pose_rmse(estimated, true) == pytest.approx(5.0)


def test_landmark_rmse_known_value():
    estimated = [(1.0, 1.0)]
    true = [(4.0, 5.0)]  # distance 5
    assert landmark_rmse(estimated, true) == pytest.approx(5.0)


def test_final_pose_error_known_value():
    assert final_pose_error((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError):
        pose_rmse([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])


def test_empty_lists_return_zero():
    assert pose_rmse([], []) == 0.0
    assert landmark_rmse([], []) == 0.0
