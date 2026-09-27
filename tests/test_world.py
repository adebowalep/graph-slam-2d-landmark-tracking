import pytest

from graph_slam.solver import extract_poses_and_landmarks, slam
from graph_slam.world import simulate


def test_simulate_output_shapes():
    result = simulate(
        num_steps=15,
        num_landmarks=4,
        world_size=100.0,
        measurement_range=50.0,
        motion_noise=2.0,
        measurement_noise=2.0,
        distance=15.0,
        seed=1,
    )
    assert len(result.data) == 14  # num_steps - 1
    assert len(result.true_poses) == 15
    assert len(result.true_landmarks) == 4
    assert result.true_poses[0] == pytest.approx((50.0, 50.0))


def test_simulate_is_reproducible_with_same_seed():
    kwargs = dict(
        num_steps=10,
        num_landmarks=3,
        world_size=100.0,
        measurement_range=50.0,
        motion_noise=2.0,
        measurement_noise=2.0,
        distance=15.0,
        seed=123,
    )
    result_a = simulate(**kwargs)
    result_b = simulate(**kwargs)
    assert result_a.true_poses == result_b.true_poses
    assert result_a.true_landmarks == result_b.true_landmarks
    assert result_a.data == result_b.data


def test_simulate_every_landmark_is_eventually_observed():
    result = simulate(
        num_steps=20,
        num_landmarks=5,
        world_size=100.0,
        measurement_range=50.0,
        motion_noise=2.0,
        measurement_noise=2.0,
        distance=20.0,
        seed=42,
    )
    observed = set()
    for measurements, _ in result.data:
        for landmark_index, _, _ in measurements:
            observed.add(landmark_index)
    assert observed == set(range(5))


def test_simulate_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        simulate(
            num_steps=0,
            num_landmarks=1,
            world_size=100.0,
            measurement_range=50.0,
            motion_noise=1.0,
            measurement_noise=1.0,
            distance=10.0,
        )
    with pytest.raises(ValueError):
        simulate(
            num_steps=5,
            num_landmarks=-1,
            world_size=100.0,
            measurement_range=50.0,
            motion_noise=1.0,
            measurement_noise=1.0,
            distance=10.0,
        )


def test_end_to_end_simulation_and_slam_within_tolerance():
    """A small seeded end-to-end run: check numerical tolerances, not exact values."""
    result = simulate(
        num_steps=20,
        num_landmarks=5,
        world_size=100.0,
        measurement_range=50.0,
        motion_noise=2.0,
        measurement_noise=2.0,
        distance=20.0,
        seed=7,
    )
    mu = slam(
        data=result.data,
        num_steps=20,
        num_landmarks=5,
        world_size=100.0,
        motion_noise=2.0,
        measurement_noise=2.0,
    )
    poses, landmarks = extract_poses_and_landmarks(mu, num_steps=20, num_landmarks=5)

    # the first pose is always recovered near-exactly (it's the anchored/gauge-fixed variable)
    assert poses[0] == pytest.approx(result.true_poses[0], abs=1e-9)

    # every other pose and every landmark should be within a generous but
    # meaningful tolerance of ground truth, given the noise scale used (2.0)
    for estimated, true in zip(poses[1:], result.true_poses[1:]):
        assert abs(estimated[0] - true[0]) < 15.0
        assert abs(estimated[1] - true[1]) < 15.0
    for estimated, true in zip(landmarks, result.true_landmarks):
        assert abs(estimated[0] - true[0]) < 15.0
        assert abs(estimated[1] - true[1]) < 15.0
