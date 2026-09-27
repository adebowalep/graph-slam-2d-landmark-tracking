"""Constraint-matrix (Omega, Xi) initialization for 2D Graph SLAM."""

from __future__ import annotations

import numpy as np


def initialize_constraints(
    num_steps: int, num_landmarks: int, world_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial information matrix Omega and information vector Xi.

    The system stacks x/y for `num_steps` robot poses followed by x/y for
    `num_landmarks` landmarks, interleaved as
    `[P0x, P0y, P1x, P1y, ..., L0x, L0y, L1x, L1y, ...]`, giving a
    `2*(num_steps + num_landmarks)` square system. All entries start at 0
    ("unknown") except the robot's starting pose, which is anchored at the
    center of the world with full confidence (weight 1) -- this is what
    makes the system solvable at all (Graph SLAM only recovers *relative*
    positions from motion/measurement constraints; anchoring one pose fixes
    the overall coordinate frame, a standard technique often called "gauge
    fixing").

    Args:
        num_steps: Number of robot poses, N.
        num_landmarks: Number of landmarks, L.
        world_size: Side length of the square world; the robot is assumed
            to start at (world_size / 2, world_size / 2).

    Returns:
        `(omega, xi)`: `omega` has shape `(2*(N+L), 2*(N+L))`, `xi` has
        shape `(2*(N+L), 1)`.

    Raises:
        ValueError: If `num_steps < 1`, `num_landmarks < 0`, or
            `world_size <= 0`.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if num_landmarks < 0:
        raise ValueError(f"num_landmarks must be >= 0, got {num_landmarks}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    size = 2 * (num_steps + num_landmarks)
    omega = np.zeros((size, size))
    xi = np.zeros((size, 1))

    omega[0, 0] = 1.0
    omega[1, 1] = 1.0
    xi[0, 0] = world_size / 2.0
    xi[1, 0] = world_size / 2.0

    return omega, xi
