# Archive: original Udacity submission

This folder preserves the project exactly as it was submitted for Udacity's
Computer Vision Nanodegree "Implement SLAM" project, before the portfolio
refactor. Nothing here is used by the `graph_slam` package at the repo root;
it's kept for provenance and to show the starting point.

Contents:

- `1. Robot Moving and Sensing.ipynb`, `2. Omega and Xi, Constraints.ipynb`,
  `3. Landmark Detection and Tracking.ipynb` — the three course notebooks
  (1 and 2 are Udacity's exploratory starter notebooks, unmodified except
  for being run; Notebook 3 contains the completed, graded implementation
  of `initialize_constraints` and `slam`).
- `1. Robot Moving and Sensing.html`, `3. Landmark Detection and Tracking.html`
  — HTML exports of the executed notebooks, as originally supplied.
- `robot_class.py` — the robot simulation with the completed `sense()` method.
- `helpers.py` — Udacity-provided world/data generation and visualization
  helpers (unmodified).
- `images/` — Udacity-provided diagrams referenced by the notebooks.
- `README_original.md` — the original assignment README (overview,
  rubric, submission instructions).
- `requirements_original.txt` — the original (2018-era, now outdated) pinned
  dependency list.
- `project3-workspace-export.zip` — the raw export from the Udacity online
  workspace this refactor was built from.

See the root [README.md](../README.md) for what was independently built on
top of this starting point.
