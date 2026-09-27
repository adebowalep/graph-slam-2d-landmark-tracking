import numpy as np
import pytest

from graph_slam.constraints import initialize_constraints


def test_dimensions_scale_with_steps_and_landmarks():
    omega, xi = initialize_constraints(num_steps=3, num_landmarks=2, world_size=100.0)
    expected_size = 2 * (3 + 2)
    assert omega.shape == (expected_size, expected_size)
    assert xi.shape == (expected_size, 1)


def test_only_starting_pose_entries_are_nonzero():
    omega, xi = initialize_constraints(num_steps=4, num_landmarks=3, world_size=100.0)
    assert omega[0, 0] == 1.0
    assert omega[1, 1] == 1.0
    # every other entry of omega must be exactly 0
    mask = np.ones_like(omega, dtype=bool)
    mask[0, 0] = mask[1, 1] = False
    assert np.all(omega[mask] == 0.0)

    assert xi[0, 0] == 50.0
    assert xi[1, 0] == 50.0
    assert np.all(xi[2:] == 0.0)


def test_starting_pose_is_world_center():
    _, xi = initialize_constraints(num_steps=2, num_landmarks=1, world_size=60.0)
    assert xi[0, 0] == 30.0
    assert xi[1, 0] == 30.0


def test_omega_is_symmetric():
    omega, _ = initialize_constraints(num_steps=5, num_landmarks=4, world_size=100.0)
    assert np.array_equal(omega, omega.T)


@pytest.mark.parametrize("num_steps,num_landmarks,world_size", [(0, 1, 100.0), (1, -1, 100.0), (1, 1, 0.0), (1, 1, -5.0)])
def test_invalid_inputs_rejected(num_steps, num_landmarks, world_size):
    with pytest.raises(ValueError):
        initialize_constraints(num_steps, num_landmarks, world_size)
