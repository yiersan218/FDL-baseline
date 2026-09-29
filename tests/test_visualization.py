import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.plot_history import (  # noqa: E402
    LOSS_KEYS,
    _phase_boundaries,
    visualize_history,
)
from visualization.main import DEFAULT_RESULTS_DIRECTORY  # noqa: E402


class VisualizationTests(unittest.TestCase):
    def test_default_experiment_results_are_inside_visualization(self):
        self.assertEqual(
            DEFAULT_RESULTS_DIRECTORY,
            PROJECT_ROOT / "visualization" / "result",
        )

    def test_visualize_history_creates_metric_and_loss_images(self):
        phases = ("pretraining", "pretraining", "clustering")
        history = []
        for round_number in range(1, 4):
            history.append(
                {
                    "round": round_number,
                    "phase": phases[round_number - 1],
                    "train": {
                        key: float(round_number) / (index + 1)
                        for index, key in enumerate(LOSS_KEYS)
                    },
                    "clustering": {
                        "acc": 0.4 + round_number * 0.01,
                        "nmi": 0.5 + round_number * 0.01,
                        "ari": 0.3 + round_number * 0.01,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "history.json"
            summary_path = root / "summary.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            summary_path.write_text(
                json.dumps({"dataset": "synthetic", "best_round": 3}),
                encoding="utf-8",
            )
            paths = visualize_history(history_path, root / "plots", summary_path)
            self.assertEqual(set(paths), {"metrics", "losses"})
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_two_stage_boundary_is_detected(self):
        history = [
            {"round": 1, "phase": "pretraining", "train": {}},
            {"round": 2, "phase": "pretraining", "train": {}},
            {"round": 3, "phase": "pretraining", "train": {}},
            {"round": 4, "phase": "clustering", "train": {}},
        ]
        boundaries = _phase_boundaries(history)
        self.assertEqual(
            [(item[0], item[1]) for item in boundaries],
            [(3.5, "Clustering stage starts")],
        )


if __name__ == "__main__":
    unittest.main()
