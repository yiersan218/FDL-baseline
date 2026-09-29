import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import (  # noqa: E402
    BACKUP_CONFIG_DIR,
    DEFAULT_CONFIG_DIR,
    INIT_CONFIG_DIR,
    apply_overrides,
    load_config,
)
from flcore.servers.servercluster import (  # noqa: E402
    FederatedMultiViewClusteringServer,
    pretraining_clustering_scale,
    progress_report_gap,
)
from flcore.trainmodel.multiview import MultiViewClusteringModel, clustering_objective  # noqa: E402
from utils.mat_data import balanced_client_indices, load_multiview_mat  # noqa: E402


class ClusteringProjectTests(unittest.TestCase):
    def test_progress_is_reported_at_ten_percent_intervals(self):
        self.assertEqual(progress_report_gap(100), 10)
        self.assertEqual(progress_report_gap(16), 2)
        self.assertEqual(progress_report_gap(10), 1)
        self.assertEqual(progress_report_gap(2), 1)

    def test_all_dataset_configs_load(self):
        configs = [load_config(path) for path in DEFAULT_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 6)
        self.assertEqual(
            {config["dataset"]["name"] for config in configs},
            {"ALOI_100", "flower17", "LandUse_21", "Mfeat", "NUSWIDE", "Scene-15"},
        )
        for config in configs:
            training = config["training"]
            self.assertGreaterEqual(training["center_init_round"], 0)
            self.assertLessEqual(
                training["center_init_round"],
                training["pretrain_rounds"],
            )
            self.assertGreater(training["pretraining_end_learning_rate"], 0)
            self.assertGreater(training["pretraining_local_epochs"], 0)
            self.assertGreater(training["clustering_local_epochs"], 0)
            self.assertGreater(training["cluster_head_learning_rate_multiplier"], 0)
            self.assertGreaterEqual(training["center_momentum"], 0)
            self.assertLess(training["center_momentum"], 1)
            for value in config["loss_weights"].values():
                self.assertGreater(float(value), 0)

    def test_backup_dataset_config_loads(self):
        configs = [load_config(path) for path in BACKUP_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["dataset"]["name"], "animal")
        for value in configs[0]["loss_weights"].values():
            self.assertGreater(float(value), 0)

    def test_scene15_two_stage_schedule(self):
        config = load_config(DEFAULT_CONFIG_DIR / "Scene-15.json")
        training = config["training"]
        self.assertEqual(training["rounds"], 30)
        self.assertEqual(training["pretrain_rounds"], 22)
        self.assertEqual(training["center_init_round"], 14)
        self.assertEqual(training["local_epochs"], 2)
        self.assertEqual(training["pretraining_local_epochs"], 2)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 5e-8)
        self.assertEqual(training["clustering_learning_rate"], 5e-8)
        self.assertEqual(training["center_momentum"], 0.99875)
        self.assertEqual(config["loss_weights"]["clustering"], 0.05)
        self.assertEqual(config["loss_weights"]["balance"], 0.05)

    def test_initial_configs_use_common_hyperparameters(self):
        configs = [load_config(path) for path in INIT_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 6)
        common_sections = []
        for config in configs:
            common_sections.append(
                {
                    "num_clients": config["dataset"]["num_clients"],
                    "normalization": config["dataset"]["normalization"],
                    "model": config["model"],
                    "training": config["training"],
                    "loss_weights": config["loss_weights"],
                    "output": config["output"],
                }
            )
        self.assertTrue(all(item == common_sections[0] for item in common_sections))

    def test_mat_loader(self):
        data = load_multiview_mat(PROJECT_ROOT / "dataset" / "LandUse_21.mat")
        self.assertEqual(data.num_samples, 2100)
        self.assertEqual(data.num_clusters, 21)
        self.assertEqual(data.view_dims, [20, 59, 40])

    def test_mat_loader_accepts_data_labels_and_transposed_views(self):
        import tempfile

        views = np.empty((1, 2), dtype=object)
        views[0, 0] = np.arange(15, dtype=np.float64).reshape(3, 5)
        views[0, 1] = np.arange(10, dtype=np.float64).reshape(2, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.mat"
            savemat(path, {"data": views, "labels": np.array([[1, 1, 2, 2, 3]])})
            data = load_multiview_mat(path)
        self.assertEqual(data.num_samples, 5)
        self.assertEqual(data.num_clusters, 3)
        self.assertEqual(data.view_dims, [3, 2])

    def test_mat_loader_accepts_gt_label_alias(self):
        import tempfile

        views = np.empty((1, 1), dtype=object)
        views[0, 0] = np.arange(20, dtype=np.float64).reshape(4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt_alias.mat"
            savemat(path, {"X": views, "gt": np.array([[1], [1], [2], [2], [3]])})
            data = load_multiview_mat(path)
        self.assertEqual(data.num_samples, 5)
        self.assertEqual(data.num_clusters, 3)
        self.assertEqual(data.view_dims, [4])

    def test_balanced_partition_is_label_free(self):
        parts = balanced_client_indices(103, 5, seed=42)
        merged = np.concatenate(parts)
        self.assertEqual(len(np.unique(merged)), 103)
        self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_model_and_unsupervised_loss(self):
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        loss, metrics = clustering_objective(
            outputs,
            views,
            {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.5, "balance": 0.05},
            clustering_enabled=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs["assignments"].shape, (12, 3))
        self.assertIn("clustering", metrics)

    def test_pretraining_scales_clustering_terms(self):
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        weights = {
            "reconstruction": 1.0,
            "consistency": 0.2,
            "clustering": 0.5,
            "balance": 0.05,
        }
        disabled_loss, _ = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=False,
        )
        zero_scale_loss, metrics = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=True,
            clustering_weight_scale=0.0,
        )
        self.assertTrue(torch.allclose(disabled_loss, zero_scale_loss))
        self.assertEqual(metrics["clustering_weight_scale"], 0.0)
        self.assertGreaterEqual(metrics["clustering"], 0.0)

    def test_two_stage_pretraining_scale(self):
        self.assertEqual(pretraining_clustering_scale(1, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(2, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(3, 2, 4), 0.5)
        self.assertEqual(pretraining_clustering_scale(4, 2, 4), 1.0)
        self.assertEqual(pretraining_clustering_scale(5, 2, 4), 1.0)

    def test_base_overrides_refresh_inherited_two_stage_parameters(self):
        import json
        import tempfile

        raw_config = json.loads(
            (DEFAULT_CONFIG_DIR / "ALOI_100.json").read_text(encoding="utf-8")
        )
        for key in (
            "pretraining_local_epochs",
            "clustering_local_epochs",
            "pretraining_end_learning_rate",
            "clustering_learning_rate",
        ):
            raw_config["training"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inherited.json"
            path.write_text(json.dumps(raw_config), encoding="utf-8")
            config = apply_overrides(
                load_config(path),
                ["training.local_epochs=3", "training.learning_rate=0.01"],
            )
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 3)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 0.001)
        self.assertEqual(training["clustering_learning_rate"], 0.01)

    def test_base_overrides_preserve_explicit_two_stage_parameters(self):
        config = apply_overrides(
            load_config(DEFAULT_CONFIG_DIR / "Scene-15.json"),
            ["training.local_epochs=1", "training.learning_rate=0.01"],
        )
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 2)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 5e-8)
        self.assertEqual(training["clustering_learning_rate"], 5e-8)

    def test_explicit_two_stage_override_stops_inheriting(self):
        import json
        import tempfile

        raw_config = json.loads(
            (DEFAULT_CONFIG_DIR / "ALOI_100.json").read_text(encoding="utf-8")
        )
        for key in ("pretraining_local_epochs", "clustering_local_epochs"):
            raw_config["training"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inherited.json"
            path.write_text(json.dumps(raw_config), encoding="utf-8")
            config = apply_overrides(
                load_config(path),
                ["training.local_epochs=1", "training.pretraining_local_epochs=3"],
            )
            config = apply_overrides(config, ["training.local_epochs=4"])
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 3)
        self.assertEqual(training["clustering_local_epochs"], 4)

    def test_fixed_dec_target_is_accepted_and_shape_checked(self):
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        weights = {
            "reconstruction": 1.0,
            "consistency": 0.2,
            "clustering": 0.5,
            "balance": 0.05,
        }
        fixed_target = torch.full_like(outputs["assignments"], 1.0 / 3.0)
        loss, _metrics = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=True,
            target_assignments=fixed_target,
        )
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            clustering_objective(
                outputs,
                views,
                weights,
                clustering_enabled=True,
                target_assignments=fixed_target[:-1],
            )

    def test_cluster_centers_use_aligned_per_cluster_counts(self):
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel([4], 2, [3], 2)
        server.device = torch.device("cpu")
        server.centers_initialized = True
        server.training = {"center_momentum": 0.0}
        with torch.no_grad():
            server.global_model.cluster_head.centers.copy_(
                torch.tensor([[0.0, 0.0], [10.0, 10.0]])
            )

        states = []
        for centers in (
            torch.tensor([[2.0, 2.0], [8.0, 8.0]]),
            torch.tensor([[1.0, 1.0], [9.0, 9.0]]),
        ):
            state = {
                key: value.detach().cpu().clone()
                for key, value in server.global_model.state_dict().items()
            }
            state["cluster_head.centers"] = centers
            states.append(state)
        updates = [
            {
                "num_samples": 10,
                "state_dict": states[0],
                "cluster_counts": torch.tensor([9.0, 1.0]),
            },
            {
                "num_samples": 10,
                "state_dict": states[1],
                "cluster_counts": torch.tensor([1.0, 9.0]),
            },
        ]

        server._aggregate(updates)

        expected = torch.tensor([[1.9, 1.9], [8.9, 8.9]])
        self.assertTrue(torch.allclose(server.global_model.cluster_head.centers, expected))


if __name__ == "__main__":
    unittest.main()
