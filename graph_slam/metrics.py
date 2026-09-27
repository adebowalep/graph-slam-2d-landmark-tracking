"""Error metrics comparing SLAM estimates against ground truth."""

from __future__ import annotations

import math


def _rmse(estimated: list[tuple[float, float]], true: list[tuple[float, float]]) -> float:
    if len(estimated) != len(true):
        raise ValueError(f"length mismatch: {len(estimated)} estimated vs {len(true)} true")
    if not estimated:
        return 0.0
    squared_errors = [
        (ex - tx) ** 2 + (ey - ty) ** 2 for (ex, ey), (tx, ty) in zip(estimated, true)
    ]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def pose_rmse(
    estimated_poses: list[tuple[float, float]], true_poses: list[tuple[float, float]]
) -> float:
    """Root-mean-square Euclidean error between estimated and true poses."""
    return _rmse(estimated_poses, true_poses)


def landmark_rmse(
    estimated_landmarks: list[tuple[float, float]], true_landmarks: list[tuple[float, float]]
) -> float:
    """Root-mean-square Euclidean error between estimated and true landmarks.

    Landmarks are compared by index, which is valid here because both the
    simulator and the solver preserve landmark identity/order throughout
    (no data-association problem to solve) -- this project does not
    implement landmark re-identification from unlabeled measurements.
    """
    return _rmse(estimated_landmarks, true_landmarks)


def final_pose_error(
    estimated_final_pose: tuple[float, float], true_final_pose: tuple[float, float]
) -> float:
    """Euclidean distance between the estimated and true final robot pose."""
    (ex, ey), (tx, ty) = estimated_final_pose, true_final_pose
    return math.hypot(ex - tx, ey - ty)
