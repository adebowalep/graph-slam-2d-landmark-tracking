"""2D Graph SLAM: robot simulation, constraint-based SLAM solver, metrics, and visualization."""

from graph_slam.robot import Robot
from graph_slam.world import SimulationResult, simulate
from graph_slam.constraints import initialize_constraints
from graph_slam.solver import UnderconstrainedSystemError, extract_poses_and_landmarks, slam

__all__ = [
    "Robot",
    "SimulationResult",
    "simulate",
    "initialize_constraints",
    "slam",
    "extract_poses_and_landmarks",
    "UnderconstrainedSystemError",
]

__version__ = "0.1.0"
