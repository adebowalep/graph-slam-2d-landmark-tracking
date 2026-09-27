import pytest

from graph_slam.robot import Robot


def test_init_starts_at_world_center():
    robot = Robot(world_size=100.0, seed=0)
    assert robot.x == 50.0
    assert robot.y == 50.0


def test_move_no_noise_updates_position_exactly():
    robot = Robot(world_size=100.0, motion_noise=0.0, seed=0)
    moved = robot.move(10.0, -5.0)
    assert moved is True
    assert robot.x == pytest.approx(60.0)
    assert robot.y == pytest.approx(45.0)


def test_move_outside_boundary_returns_false_and_does_not_move():
    robot = Robot(world_size=10.0, motion_noise=0.0, seed=0)
    moved = robot.move(100.0, 0.0)
    assert moved is False
    assert robot.x == 5.0
    assert robot.y == 5.0


@pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (-100.0, -100.0), (100.0, 100.0)])
def test_move_boundary_edge_cases(dx, dy):
    robot = Robot(world_size=10.0, motion_noise=0.0, seed=0)
    x_before, y_before = robot.x, robot.y
    moved = robot.move(dx, dy)
    if moved:
        assert 0.0 <= robot.x <= 10.0
        assert 0.0 <= robot.y <= 10.0
    else:
        assert (robot.x, robot.y) == (x_before, y_before)


def test_sense_no_noise_visible_landmark():
    robot = Robot(world_size=100.0, measurement_range=30.0, measurement_noise=0.0, seed=0)
    robot.landmarks = [[52.0, 51.0]]
    robot.num_landmarks = 1
    measurements = robot.sense()
    assert measurements == [[0, 2.0, 1.0]]


def test_sense_out_of_range_landmark_excluded():
    robot = Robot(world_size=100.0, measurement_range=5.0, measurement_noise=0.0, seed=0)
    robot.landmarks = [[90.0, 90.0]]  # far from center (50, 50)
    robot.num_landmarks = 1
    assert robot.sense() == []


def test_sense_range_minus_one_always_visible():
    robot = Robot(world_size=100.0, measurement_range=-1, measurement_noise=0.0, seed=0)
    robot.landmarks = [[0.0, 0.0], [100.0, 100.0]]
    robot.num_landmarks = 2
    measurements = robot.sense()
    assert len(measurements) == 2


def test_sense_returns_correct_format():
    robot = Robot(world_size=100.0, measurement_range=-1, measurement_noise=0.0, seed=0)
    robot.landmarks = [[10.0, 20.0]]
    robot.num_landmarks = 1
    measurements = robot.sense()
    assert len(measurements) == 1
    index, dx, dy = measurements[0]
    assert index == 0
    assert isinstance(dx, float)
    assert isinstance(dy, float)


def test_sense_noise_is_bounded_and_reproducible_with_seed():
    robot_a = Robot(world_size=100.0, measurement_range=-1, measurement_noise=2.0, seed=7)
    robot_a.landmarks = [[52.0, 51.0]]
    robot_a.num_landmarks = 1

    robot_b = Robot(world_size=100.0, measurement_range=-1, measurement_noise=2.0, seed=7)
    robot_b.landmarks = [[52.0, 51.0]]
    robot_b.num_landmarks = 1

    assert robot_a.sense() == robot_b.sense()


def test_sense_noise_magnitude_bounded():
    robot = Robot(world_size=100.0, measurement_range=-1, measurement_noise=2.0, seed=7)
    robot.landmarks = [[52.0, 51.0]]
    robot.num_landmarks = 1
    _, dx, dy = robot.sense()[0]
    # true offset is (2.0, 1.0); noise is uniform on [-1, 1) * measurement_noise
    assert abs(dx - 2.0) < 2.0
    assert abs(dy - 1.0) < 2.0


def test_make_landmarks_count_and_bounds():
    robot = Robot(world_size=50.0, seed=1)
    robot.make_landmarks(4)
    assert robot.num_landmarks == 4
    assert len(robot.landmarks) == 4
    for x, y in robot.landmarks:
        assert 0.0 <= x <= 50.0
        assert 0.0 <= y <= 50.0


def test_invalid_world_size_rejected():
    with pytest.raises(ValueError):
        Robot(world_size=0.0)


def test_invalid_negative_noise_rejected():
    with pytest.raises(ValueError):
        Robot(motion_noise=-1.0)
    with pytest.raises(ValueError):
        Robot(measurement_noise=-1.0)


def test_make_landmarks_negative_count_rejected():
    robot = Robot(seed=0)
    with pytest.raises(ValueError):
        robot.make_landmarks(-1)
