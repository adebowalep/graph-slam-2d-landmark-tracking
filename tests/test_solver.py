"""Solver tests, including a fully hand-derived analytic example.

The analytic example (`test_two_pose_one_landmark_matches_hand_derivation`)
sets up 2 poses and 1 landmark with unit noise weights and solves the 6x6
linear system by hand (see the derivation in the docstring below) to get
an exact expected `mu`, independent of the implementation under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_slam.solver import UnderconstrainedSystemError, extract_poses_and_landmarks, slam


def test_two_pose_one_landmark_matches_hand_derivation():
    """2 poses, 1 landmark, world_size=10, motion_noise=measurement_noise=1.0.

    With unit weights, decoupling x and y (they never interact), and using
    a=P0, b=P1, c=L (per axis), the constraint equations reduce to:

        3a - b - c = world_center - measurement - motion
        -a + b = motion
        -a + c = measurement

    Substituting (ii) and (iii) into (i) leaves `a = world_center` exactly
    (the starting pose is always recovered exactly, since it's the
    anchored/gauge-fixed variable), then `b = world_center + motion` and
    `c = world_center + measurement`.

    With world_size=10 (center=5), motion=(4, 1), and a measurement of
    (3, -2) from pose 0 to the landmark:

        P0 = (5, 5)
        P1 = (5 + 4, 5 + 1)   = (9, 6)
        L  = (5 + 3, 5 - 2)   = (8, 3)
    """
    data = [([[0, 3.0, -2.0]], (4.0, 1.0))]

    mu = slam(
        data=data,
        num_steps=2,
        num_landmarks=1,
        world_size=10.0,
        motion_noise=1.0,
        measurement_noise=1.0,
    )
    poses, landmarks = extract_poses_and_landmarks(mu, num_steps=2, num_landmarks=1)

    assert poses[0] == pytest.approx((5.0, 5.0))
    assert poses[1] == pytest.approx((9.0, 6.0))
    assert landmarks[0] == pytest.approx((8.0, 3.0))


def test_three_pose_no_noise_recovers_exact_square_path():
    """A no-measurement-error, no-motion-error path should be recovered exactly."""
    # Robot moves (10, 0) then (0, 10); one landmark seen from every pose.
    data = [
        ([[0, 5.0, 5.0]], (10.0, 0.0)),
        ([[0, -5.0, 5.0]], (0.0, 10.0)),
    ]
    mu = slam(
        data=data,
        num_steps=3,
        num_landmarks=1,
        world_size=100.0,
        motion_noise=1.0,
        measurement_noise=1.0,
    )
    poses, landmarks = extract_poses_and_landmarks(mu, num_steps=3, num_landmarks=1)

    assert poses[0] == pytest.approx((50.0, 50.0))
    assert poses[1] == pytest.approx((60.0, 50.0))
    assert poses[2] == pytest.approx((60.0, 60.0))
    # landmark seen from pose0 at (5,5) offset -> (55, 55); from pose1 at (-5,5) -> (55, 55)
    assert landmarks[0] == pytest.approx((55.0, 55.0))


def test_wrong_data_length_raises_value_error():
    data = [([[0, 1.0, 1.0]], (1.0, 1.0))]  # length 1, but num_steps=3 needs length 2
    with pytest.raises(ValueError):
        slam(data, num_steps=3, num_landmarks=1, world_size=100.0, motion_noise=1.0, measurement_noise=1.0)


def test_out_of_range_landmark_index_raises_value_error():
    data = [([[5, 1.0, 1.0]], (1.0, 1.0))]  # landmark index 5, but num_landmarks=1
    with pytest.raises(ValueError):
        slam(data, num_steps=2, num_landmarks=1, world_size=100.0, motion_noise=1.0, measurement_noise=1.0)


def test_never_observed_landmark_raises_underconstrained_error():
    data = [
        ([], (1.0, 1.0)),
        ([], (1.0, 1.0)),
    ]  # landmark 0 is declared but never measured
    with pytest.raises(UnderconstrainedSystemError):
        slam(data, num_steps=3, num_landmarks=1, world_size=100.0, motion_noise=1.0, measurement_noise=1.0)


@pytest.mark.parametrize("motion_noise,measurement_noise", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_non_positive_noise_rejected(motion_noise, measurement_noise):
    data = [([[0, 1.0, 1.0]], (1.0, 1.0))]
    with pytest.raises(ValueError):
        slam(
            data,
            num_steps=2,
            num_landmarks=1,
            world_size=100.0,
            motion_noise=motion_noise,
            measurement_noise=measurement_noise,
        )


def test_extract_poses_and_landmarks_shapes():
    mu = np.arange(2 * (3 + 2), dtype=float)
    poses, landmarks = extract_poses_and_landmarks(mu, num_steps=3, num_landmarks=2)
    assert len(poses) == 3
    assert len(landmarks) == 2
    assert poses[0] == (0.0, 1.0)
    assert landmarks[0] == (6.0, 7.0)
