import json

from graph_slam.cli import main


def test_cli_end_to_end_writes_figure_and_metrics(tmp_path):
    output_dir = tmp_path / "results"
    exit_code = main(
        [
            "--seed",
            "42",
            "--steps",
            "20",
            "--num-landmarks",
            "5",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    figure_path = output_dir / "trajectory.png"
    metrics_path = output_dir / "metrics.json"
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "pose_rmse" in metrics
    assert "landmark_rmse" in metrics
    assert "final_pose_error" in metrics
    assert metrics["pose_rmse"] >= 0.0
