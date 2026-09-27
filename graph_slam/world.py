"""Synthetic world / trajectory generation for Graph SLAM demos and tests.

Origin: this is a rewrite of the course starter's `helpers.make_data`. The
core simulation loop (move in a straight line, pick a new heading on
collision with a wall, retry until every landmark has been observed at
least once) is preserved faithfully. What's new:

  * it returns the *true* pose at every time step (not just the final one),
    so real error metrics can be computed instead of only eyeballing a plot;
  * it takes an explicit `seed` for reproducibility instead of relying on
    the global `random` module;
  * it fails loudly (`RuntimeError`) after `max_attempts` instead of
    looping forever if a configuration can never observe every landmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from graph_slam.robot import Robot

Measurement = list[float]  # [landmark_index, dx, dy]
TimeStep = tuple[list[Measurement], tuple[float, float]]  # (measurements, motion)


@dataclass
class SimulationResult:
    """The output of `simulate`: SLAM input data plus ground truth for evaluation."""

    data: list[TimeStep]
    true_poses: list[tuple[float, float]]
    true_landmarks: list[tuple[float, float]]
    world_size: float
    num_landmarks: int
    num_steps: int
    motion_noise: float
    measurement_noise: float
    seed: int | None


def simulate(
    num_steps: int,
    num_landmarks: int,
    world_size: float,
    measurement_range: float,
    motion_noise: float,
    measurement_noise: float,
    distance: float,
    seed: int | None = None,
    max_attempts: int = 2000,
) -> SimulationResult:
    """Simulate a robot moving through a world with random landmarks.

    Args:
        num_steps: Number of robot poses (N). The robot senses and moves
            `num_steps - 1` times, so `len(result.data) == num_steps - 1`.
        num_landmarks: Number of landmarks to scatter in the world.
        world_size: Side length of the square world.
        measurement_range: See `Robot.measurement_range`.
        motion_noise: See `Robot.motion_noise`.
        measurement_noise: See `Robot.measurement_noise`.
        distance: Intended step length each time the robot moves.
        seed: Optional seed for reproducibility.
        max_attempts: Safety cap on world-generation retries (a fresh
            landmark layout is retried if it never lets every landmark be
            observed at least once, which would make the SLAM system
            underconstrained/singular).

    Returns:
        A `SimulationResult` with the SLAM input `data` and ground truth.

    Raises:
        ValueError: For non-positive `num_steps` or negative `num_landmarks`.
        RuntimeError: If `max_attempts` world layouts in a row all fail to
            observe every landmark at least once.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if num_landmarks < 0:
        raise ValueError(f"num_landmarks must be >= 0, got {num_landmarks}")

    for attempt in range(max_attempts):
        attempt_seed = None if seed is None else seed + attempt
        robot = Robot(
            world_size=world_size,
            measurement_range=measurement_range,
            motion_noise=motion_noise,
            measurement_noise=measurement_noise,
            seed=attempt_seed,
        )
        robot.make_landmarks(num_landmarks)
        seen = [False] * num_landmarks

        orientation = robot.rand() * math.pi + math.pi  # uniform-ish in [0, 2*pi)
        dx = math.cos(orientation) * distance
        dy = math.sin(orientation) * distance

        data: list[TimeStep] = []
        true_poses: list[tuple[float, float]] = [(robot.x, robot.y)]

        for _ in range(num_steps - 1):
            measurements = robot.sense()
            for landmark_index, _, _ in measurements:
                seen[int(landmark_index)] = True

            while not robot.move(dx, dy):
                orientation = robot.rand() * math.pi + math.pi
                dx = math.cos(orientation) * distance
                dy = math.sin(orientation) * distance

            data.append((measurements, (dx, dy)))
            true_poses.append((robot.x, robot.y))

        if num_landmarks == 0 or all(seen):
            return SimulationResult(
                data=data,
                true_poses=true_poses,
                true_landmarks=[tuple(landmark) for landmark in robot.landmarks],
                world_size=world_size,
                num_landmarks=num_landmarks,
                num_steps=num_steps,
                motion_noise=motion_noise,
                measurement_noise=measurement_noise,
                seed=seed,
            )

    raise RuntimeError(
        f"Could not generate a world where every landmark is observed at least once "
        f"after {max_attempts} attempts. Try a larger measurement_range, fewer "
        f"landmarks, or more num_steps."
    )
