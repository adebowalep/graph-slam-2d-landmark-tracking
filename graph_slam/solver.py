"""2D Graph SLAM solver: build constraints from motion/measurement data and solve for mu.

Origin: this is the completed version of the course starter's `slam()`
TODO. The constraint-update math (weight each motion/measurement by
`1/noise`, symmetric off-diagonal updates in Omega, signed updates in Xi)
is the standard Graph SLAM formulation and was implemented from scratch
against the course's problem statement, not copied from a reference
solution. Two changes from the literal course template:

  * `np.linalg.solve(omega, xi)` is used instead of forming
    `np.linalg.inv(omega)` explicitly and multiplying -- mathematically
    equivalent for a non-singular system, but avoids computing a full
    matrix inverse (more numerically stable, and normally faster for a
    single right-hand side).
  * A never-observed landmark is detected *before* solving and reported
    with a clear error, instead of surfacing as a cryptic
    `numpy.linalg.LinAlgError: Singular matrix` (or a silently wrong
    answer if the system happens to be only near-singular).
"""

from __future__ import annotations

import numpy as np

from graph_slam.constraints import initialize_constraints
from graph_slam.world import TimeStep


class UnderconstrainedSystemError(RuntimeError):
    """Raised when the SLAM constraint system cannot be uniquely solved.

    This happens when at least one landmark is never referenced by any
    measurement in `data`: that landmark's rows/columns in Omega stay all
    zero, leaving Omega singular.
    """


def slam(
    data: list[TimeStep],
    num_steps: int,
    num_landmarks: int,
    world_size: float,
    motion_noise: float,
    measurement_noise: float,
) -> np.ndarray:
    """Estimate the robot's full trajectory and all landmark positions.

    Args:
        data: A list of `(measurements, motion)` pairs, one per time step,
            where `measurements` is a list of `[landmark_index, dx, dy]`
            and `motion` is `(dx, dy)`. Must have length `num_steps - 1`
            (one motion between each pair of consecutive poses).
        num_steps: Number of robot poses, N.
        num_landmarks: Number of landmarks, L.
        world_size: Side length of the square world.
        motion_noise: Noise scale used to weight motion constraints
            (weight = `1 / motion_noise`).
        measurement_noise: Noise scale used to weight measurement
            constraints (weight = `1 / measurement_noise`).

    Returns:
        `mu`, a flat array of length `2*(N+L)`: `[P0x, P0y, ..., L0x, L0y, ...]`.
        Use `extract_poses_and_landmarks` to split it into poses and landmarks.

    Raises:
        ValueError: If `len(data) != num_steps - 1`, a measurement
            references a landmark index outside `[0, num_landmarks)`, or
            either noise parameter is <= 0.
        UnderconstrainedSystemError: If a landmark is never observed, or
            Omega is otherwise singular.
    """
    if len(data) != num_steps - 1:
        raise ValueError(
            f"len(data) must equal num_steps - 1 (one motion per step transition); "
            f"got len(data)={len(data)}, num_steps={num_steps}"
        )
    if motion_noise <= 0 or measurement_noise <= 0:
        raise ValueError("motion_noise and measurement_noise must be positive")

    observed = [False] * num_landmarks
    for measurements, _ in data:
        for landmark_index, _, _ in measurements:
            if not (0 <= landmark_index < num_landmarks):
                raise ValueError(
                    f"measurement references landmark index {landmark_index}, "
                    f"outside [0, {num_landmarks})"
                )
            observed[int(landmark_index)] = True

    unobserved = [i for i, seen in enumerate(observed) if not seen]
    if unobserved:
        raise UnderconstrainedSystemError(
            f"landmark(s) {unobserved} are never referenced by any measurement in `data`; "
            f"the SLAM system is underconstrained (Omega would be singular)."
        )

    omega, xi = initialize_constraints(num_steps, num_landmarks, world_size)

    for t, (measurements, motion) in enumerate(data):
        pose_index = 2 * t

        for landmark_idx, dx, dy in measurements:
            landmark_index = 2 * (num_steps + int(landmark_idx))
            for offset, d in ((0, dx), (1, dy)):
                p = pose_index + offset
                l = landmark_index + offset

                omega[p, p] += 1.0 / measurement_noise
                omega[l, l] += 1.0 / measurement_noise
                omega[p, l] += -1.0 / measurement_noise
                omega[l, p] += -1.0 / measurement_noise

                xi[p, 0] += -d / measurement_noise
                xi[l, 0] += d / measurement_noise

        dx, dy = motion
        next_pose_index = 2 * (t + 1)
        for offset, d in ((0, dx), (1, dy)):
            p = pose_index + offset
            n = next_pose_index + offset

            omega[p, p] += 1.0 / motion_noise
            omega[n, n] += 1.0 / motion_noise
            omega[p, n] += -1.0 / motion_noise
            omega[n, p] += -1.0 / motion_noise

            xi[p, 0] += -d / motion_noise
            xi[n, 0] += d / motion_noise

    try:
        mu = np.linalg.solve(omega, xi)
    except np.linalg.LinAlgError as exc:
        raise UnderconstrainedSystemError(
            "Omega is singular even though every landmark was observed at least once; "
            "the pose/landmark graph may not be fully connected."
        ) from exc

    return mu.flatten()


def extract_poses_and_landmarks(
    mu: np.ndarray, num_steps: int, num_landmarks: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split a flat `mu` vector into a list of poses and a list of landmarks."""
    poses = [(float(mu[2 * i]), float(mu[2 * i + 1])) for i in range(num_steps)]
    landmarks = [
        (float(mu[2 * (num_steps + i)]), float(mu[2 * (num_steps + i) + 1]))
        for i in range(num_landmarks)
    ]
    return poses, landmarks
