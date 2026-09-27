"""Command-line demo: generate a synthetic world, run Graph SLAM, report metrics, save a figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph_slam.metrics import final_pose_error, landmark_rmse, pose_rmse
from graph_slam.solver import UnderconstrainedSystemError, extract_poses_and_landmarks, slam
from graph_slam.viz import plot_comparison
from graph_slam.world import simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-slam-demo",
        description=(
            "Generate a synthetic 2D world, run the Graph SLAM solver on the "
            "simulated motion/measurement data, and report how closely the "
            "estimate matches the (otherwise hidden) ground truth."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--steps", type=int, default=20, help="Number of robot poses, N (default: 20)")
    parser.add_argument(
        "--num-landmarks", type=int, default=5, help="Number of landmarks (default: 5)"
    )
    parser.add_argument(
        "--world-size", type=float, default=100.0, help="Square world side length (default: 100.0)"
    )
    parser.add_argument(
        "--measurement-range",
        type=float,
        default=50.0,
        help="Max |dx|, |dy| for a landmark to be sensed, -1 for unlimited (default: 50.0)",
    )
    parser.add_argument(
        "--motion-noise", type=float, default=2.0, help="Motion noise scale (default: 2.0)"
    )
    parser.add_argument(
        "--measurement-noise", type=float, default=2.0, help="Measurement noise scale (default: 2.0)"
    )
    parser.add_argument(
        "--distance", type=float, default=20.0, help="Intended distance per move (default: 20.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write the figure and metrics.json into (default: ./results)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    result = simulate(
        num_steps=args.steps,
        num_landmarks=args.num_landmarks,
        world_size=args.world_size,
        measurement_range=args.measurement_range,
        motion_noise=args.motion_noise,
        measurement_noise=args.measurement_noise,
        distance=args.distance,
        seed=args.seed,
    )

    try:
        mu = slam(
            data=result.data,
            num_steps=args.steps,
            num_landmarks=args.num_landmarks,
            world_size=args.world_size,
            motion_noise=args.motion_noise,
            measurement_noise=args.measurement_noise,
        )
    except UnderconstrainedSystemError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    estimated_poses, estimated_landmarks = extract_poses_and_landmarks(
        mu, args.steps, args.num_landmarks
    )

    metrics = {
        "pose_rmse": pose_rmse(estimated_poses, result.true_poses),
        "landmark_rmse": landmark_rmse(estimated_landmarks, result.true_landmarks),
        "final_pose_error": final_pose_error(estimated_poses[-1], result.true_poses[-1]),
        "seed": args.seed,
        "steps": args.steps,
        "num_landmarks": args.num_landmarks,
        "world_size": args.world_size,
        "motion_noise": args.motion_noise,
        "measurement_noise": args.measurement_noise,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plot_comparison(
        true_poses=result.true_poses,
        estimated_poses=estimated_poses,
        true_landmarks=result.true_landmarks,
        estimated_landmarks=estimated_landmarks,
        world_size=args.world_size,
        save_path=args.output_dir / "trajectory.png",
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"pose RMSE:        {metrics['pose_rmse']:.3f}")
    print(f"landmark RMSE:    {metrics['landmark_rmse']:.3f}")
    print(f"final pose error: {metrics['final_pose_error']:.3f}")
    print(f"figure saved to:  {figure_path}")
    print(f"metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
