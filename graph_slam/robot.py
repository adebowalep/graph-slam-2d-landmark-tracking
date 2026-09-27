"""2D robot simulation: motion with noise, and landmark sensing within a range.

This is a refactor of the Udacity Computer Vision Nanodegree "Implement SLAM"
starter `robot_class.py`, plus the completed `sense()` implementation. The
mathematical behavior (uniform noise, box-shaped measurement range, hard world
boundary) is preserved exactly from the original; see the "Origin" note below
each method for what changed and why.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Robot:
    """A robot that moves in a straight line and senses (dx, dy) offsets to landmarks.

    The robot senses x/y *offsets* to landmarks directly, not range and
    bearing. This is a deliberate simplification inherited from the course
    starter code, not an oversight -- see the "Limitations" section of the
    project README.

    Attributes:
        world_size: The world is a square of this side length; valid
            coordinates are in [0, world_size].
        measurement_range: Maximum |dx| and |dy| (each independently) for a
            landmark to be visible. Pass -1 to make all landmarks always
            visible. Note this is a box/Chebyshev range, not a Euclidean
            radius -- inherited from the original course sensor model.
        motion_noise: Scale of the uniform noise added to each motion step.
        measurement_noise: Scale of the uniform noise added to each sensed
            (dx, dy).
        seed: Optional seed for a private random.Random instance, so
            simulations are reproducible without touching global random
            state. (Origin: the course starter used the global `random`
            module directly, which made reproducing an exact run
            impossible when other code also drew from `random`. Using a
            per-robot `random.Random` instance is the one behavioral
            change from the original -- the noise *distribution* is
            identical, uniform on [-1, 1) scaled by the noise parameter.)
    """

    world_size: float = 100.0
    measurement_range: float = 30.0
    motion_noise: float = 1.0
    measurement_noise: float = 1.0
    seed: int | None = None

    x: float = field(init=False)
    y: float = field(init=False)
    landmarks: list[list[float]] = field(init=False, default_factory=list)
    num_landmarks: int = field(init=False, default=0)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.motion_noise < 0 or self.measurement_noise < 0:
            raise ValueError("motion_noise and measurement_noise must be non-negative")
        self.x = self.world_size / 2.0
        self.y = self.world_size / 2.0
        self._rng = random.Random(self.seed)

    def rand(self) -> float:
        """A random float uniform on [-1.0, 1.0)."""
        return self._rng.random() * 2.0 - 1.0

    def move(self, dx: float, dy: float) -> bool:
        """Attempt to move by (dx, dy) plus motion noise.

        Returns False (and leaves the robot's position unchanged) if the
        noisy destination would fall outside the world boundary.
        """
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise

        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False

        self.x = x
        self.y = y
        return True

    def sense(self) -> list[list[float]]:
        """Measure noisy (dx, dy) offsets to every landmark within range.

        Returns:
            A list of `[landmark_index, dx, dy]` entries, one per landmark
            currently within `measurement_range` (all landmarks, if
            `measurement_range == -1`).
        """
        measurements: list[list[float]] = []

        for index, (landmark_x, landmark_y) in enumerate(self.landmarks):
            dx = landmark_x - self.x + self.rand() * self.measurement_noise
            dy = landmark_y - self.y + self.rand() * self.measurement_noise

            in_range = self.measurement_range == -1 or (
                abs(dx) <= self.measurement_range and abs(dy) <= self.measurement_range
            )
            if in_range:
                measurements.append([index, dx, dy])

        return measurements

    def make_landmarks(self, num_landmarks: int) -> None:
        """Place `num_landmarks` landmarks uniformly at random in the world."""
        if num_landmarks < 0:
            raise ValueError(f"num_landmarks must be non-negative, got {num_landmarks}")
        self.landmarks = [
            [round(self._rng.random() * self.world_size), round(self._rng.random() * self.world_size)]
            for _ in range(num_landmarks)
        ]
        self.num_landmarks = num_landmarks

    def __repr__(self) -> str:
        return f"Robot: [x={self.x:.5f} y={self.y:.5f}]"
