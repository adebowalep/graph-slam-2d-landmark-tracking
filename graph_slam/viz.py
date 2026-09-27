"""Visualization: side-by-side comparison of true vs. SLAM-estimated trajectory and landmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display required (CLI / CI use)
import matplotlib.pyplot as plt


def plot_comparison(
    true_poses: list[tuple[float, float]],
    estimated_poses: list[tuple[float, float]],
    true_landmarks: list[tuple[float, float]],
    estimated_landmarks: list[tuple[float, float]],
    world_size: float,
    save_path: str | Path,
) -> Path:
    """Plot true vs. estimated trajectory and landmarks and save to `save_path`.

    Returns the path the figure was saved to.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    tx, ty = zip(*true_poses)
    ex, ey = zip(*estimated_poses)
    ax.plot(tx, ty, "o-", color="tab:blue", alpha=0.6, label="True trajectory")
    ax.plot(ex, ey, "o--", color="tab:orange", alpha=0.8, label="Estimated trajectory")

    if true_landmarks:
        ltx, lty = zip(*true_landmarks)
        ax.scatter(ltx, lty, marker="x", s=120, color="tab:blue", label="True landmarks")
    if estimated_landmarks:
        lex, ley = zip(*estimated_landmarks)
        ax.scatter(lex, ley, marker="x", s=120, color="tab:orange", label="Estimated landmarks")

    ax.set_xlim(-5, world_size + 5)
    ax.set_ylim(-5, world_size + 5)
    ax.set_aspect("equal")
    ax.set_title("Graph SLAM: true vs. estimated trajectory and landmarks")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path
