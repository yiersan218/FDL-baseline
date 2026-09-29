import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.clients.clientcluster import FederatedClusteringClient
from flcore.trainmodel.multiview import MultiViewClusteringModel
from utils.clustering_metrics import evaluate_clustering
from utils.mat_data import MultiViewSubset, balanced_client_indices


def progress_report_gap(total_rounds):
    """Report training progress after roughly every 10% of all rounds."""
    return max(1, int(np.ceil(int(total_rounds) / 10.0)))


class FederatedMultiViewClusteringServer:
    def __init__(self, data, config, device):
        self.data = data
        self.config = config
        self.device = device
        self.training = config["training"]
        self.seed = int(self.training["seed"])
        self.rng = random.Random(self.seed)
        partitions = balanced_client_indices(
            data.num_samples,
            int(config["dataset"]["num_clients"]),
            self.seed,
        )
        if any(len(indices) < int(config["dataset"]["num_clusters"]) for indices in partitions):
            raise ValueError("Every client needs at least num_clusters samples for center initialization")
        self.clients = [
            FederatedClusteringClient(i, data, indices, config, device)
            for i, indices in enumerate(partitions)
        ]
        model = config["model"]
        self.global_model = MultiViewClusteringModel(
            view_dims=data.view_dims,
            num_clusters=int(config["dataset"]["num_clusters"]),
            hidden_dims=list(model["hidden_dims"]),
            embedding_dim=int(model["embedding_dim"]),
            dropout=float(model.get("dropout", 0.0)),
            alpha=float(model.get("student_t_alpha", 1.0)),
        ).to(device)
        self.history = []
        self.centers_initialized = False
        self.best_state = None
        self.best_round = None
        self.best_metrics = None

    def _selected_clients(self):
        count = max(1, int(np.ceil(len(self.clients) * float(self.training["join_ratio"]))))
        return self.rng.sample(self.clients, count) if count < len(self.clients) else list(self.clients)

    def _initialize_centers(self):
        summaries = [client.cluster_summary(self.global_model) for client in self.clients]
        centers = np.concatenate([item[0] for item in summaries], axis=0)
        counts = np.concatenate([item[1] for item in summaries], axis=0)
        kmeans = KMeans(
            n_clusters=int(self.config["dataset"]["num_clusters"]),
            n_init=int(self.training.get("center_init_n_init", 10)),
            random_state=self.seed,
        ).fit(centers, sample_weight=np.maximum(counts, 1.0))
        with torch.no_grad():
            value = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
            self.global_model.cluster_head.centers.copy_(value)
        self.centers_initialized = True

    @staticmethod
    def _align_centers(state_dict, reference_centers, cluster_counts=None):
        key = "cluster_head.centers"
        centers = state_dict[key].numpy()
        reference = reference_centers.numpy()
        distances = ((centers[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(distances)
        aligned = np.empty_like(centers)
        aligned[columns] = centers[rows]
        state_dict[key] = torch.from_numpy(aligned)
        if cluster_counts is None:
            return None
        counts = cluster_counts.numpy()
        aligned_counts = np.empty_like(counts)
        aligned_counts[columns] = counts[rows]
        return torch.from_numpy(aligned_counts)

    def _aggregate(self, updates):
        center_key = "cluster_head.centers"
        reference = self.global_model.state_dict()[center_key].detach().cpu()
        if self.centers_initialized:
            for update in updates:
                update["cluster_counts"] = self._align_centers(
                    update["state_dict"],
                    reference,
                    update.get("cluster_counts"),
                )
        total_samples = sum(update["num_samples"] for update in updates)
        result = {}
        for key, reference_value in self.global_model.state_dict().items():
            values = [update["state_dict"][key] for update in updates]
            if key == center_key and self.centers_initialized:
                numerator = torch.zeros_like(values[0])
                denominator = torch.zeros(values[0].shape[0], dtype=values[0].dtype)
                for update, value in zip(updates, values):
                    counts = update.get("cluster_counts")
                    if counts is None:
                        counts = torch.full_like(
                            denominator,
                            float(update["num_samples"]) / denominator.numel(),
                        )
                    counts = counts.to(dtype=values[0].dtype)
                    numerator.add_(value * counts.unsqueeze(1))
                    denominator.add_(counts)
                aggregated = numerator / denominator.clamp_min(1e-12).unsqueeze(1)
                empty_clusters = denominator <= 1e-12
                aggregated[empty_clusters] = reference[empty_clusters]
                momentum = float(self.training.get("center_momentum", 0.0))
                result[key] = momentum * reference + (1.0 - momentum) * aggregated
                continue
            if reference_value.is_floating_point():
                aggregated = torch.zeros_like(values[0])
                for update, value in zip(updates, values):
                    aggregated.add_(value, alpha=update["num_samples"] / total_samples)
                result[key] = aggregated
            else:
                result[key] = values[0]
        self.global_model.load_state_dict(result)
        self.global_model.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        self.global_model.eval()
        dataset = MultiViewSubset(self.data, np.arange(self.data.num_samples))
        loader = DataLoader(
            dataset,
            batch_size=int(self.training.get("eval_batch_size", self.training["batch_size"])),
            shuffle=False,
            num_workers=int(self.training.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
        )
        predictions, labels, confidences = [], [], []
        for views, batch_labels, _indices in loader:
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            assignments = self.global_model(views)["assignments"]
            predictions.append(assignments.argmax(dim=1).cpu().numpy())
            confidences.append(assignments.max(dim=1).values.cpu().numpy())
            labels.append(batch_labels.numpy())
        metrics = evaluate_clustering(np.concatenate(labels), np.concatenate(predictions))
        metrics["confidence"] = float(np.concatenate(confidences).mean())
        return metrics

    def train(self):
        rounds = int(self.training["rounds"])
        pretrain_rounds = int(self.training["pretrain_rounds"])
        representation_warmup_rounds = int(
            self.training.get("representation_warmup_rounds", 1)
        )
        eval_gap = int(self.training.get("eval_gap", 1))
        report_gap = progress_report_gap(rounds)
        started = time.time()
        for round_index in range(rounds):
            if round_index == representation_warmup_rounds and not self.centers_initialized:
                print("Initializing global cluster centers from client summaries ...")
                self._initialize_centers()
            formal_training = round_index >= pretrain_rounds
            clustering_enabled = self.centers_initialized
            if clustering_enabled and not formal_training:
                joint_rounds = max(pretrain_rounds - representation_warmup_rounds, 1)
                completed_joint_rounds = round_index - representation_warmup_rounds + 1
                clustering_weight_scale = min(
                    1.0,
                    completed_joint_rounds / joint_rounds,
                )
                phase = "joint_pretrain"
            elif formal_training:
                clustering_weight_scale = 1.0
                phase = "clustering"
            else:
                clustering_weight_scale = 0.0
                phase = "representation_warmup"
            selected = self._selected_clients()
            updates = [
                client.train(
                    self.global_model,
                    clustering_enabled,
                    round_index,
                    formal_training=formal_training,
                    clustering_weight_scale=clustering_weight_scale,
                )
                for client in selected
            ]
            self._aggregate(updates)
            train_metrics = self._weighted_training_metrics(updates)
            record = {
                "round": round_index + 1,
                "phase": phase,
                "clients": [client.id for client in selected],
                "train": train_metrics,
                "elapsed_seconds": time.time() - started,
            }
            if (round_index + 1) % eval_gap == 0 or round_index + 1 == rounds:
                record["clustering"] = self.evaluate()
                self._update_best(record["clustering"], round_index + 1, formal_training)
            self.history.append(record)
            round_number = round_index + 1
            if round_number % report_gap == 0 or round_number == rounds:
                clustering = record.get("clustering", {})
                print(
                    f"Round {round_number:03d}/{rounds} {record['phase']:<10} "
                    f"loss={train_metrics['loss']:.4f} "
                    f"ACC={clustering.get('acc', float('nan')):.4f} "
                    f"NMI={clustering.get('nmi', float('nan')):.4f} "
                    f"ARI={clustering.get('ari', float('nan')):.4f}"
                )
        if self.best_state is not None:
            self.global_model.load_state_dict(self.best_state)
            self.global_model.to(self.device)
        final_metrics = self.evaluate()
        return self._save(final_metrics)

    def _update_best(self, metrics, round_number, formal_training):
        if not formal_training:
            return
        selection_metric = self.training.get("selection_metric", "nmi")
        if selection_metric not in metrics:
            raise KeyError(f"Unknown selection metric: {selection_metric}")
        if self.best_metrics is None or metrics[selection_metric] > self.best_metrics[selection_metric]:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_round = int(round_number)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.global_model.state_dict().items()
            }

    @staticmethod
    def _weighted_training_metrics(updates):
        total = sum(update["num_samples"] for update in updates)
        keys = updates[0]["metrics"].keys()
        return {
            key: sum(update["metrics"][key] * update["num_samples"] for update in updates) / total
            for key in keys
        }

    def _save(self, final_metrics):
        output = self.config["output"]
        root = Path(output["directory"]) / self.data.name
        root.mkdir(parents=True, exist_ok=True)
        with (root / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2, ensure_ascii=False)
        summary = {
            "dataset": self.data.name,
            "samples": self.data.num_samples,
            "view_dims": self.data.view_dims,
            "num_clusters": self.data.num_clusters,
            "metrics": final_metrics,
            "best_round": self.best_round,
            "selection_metric": self.training.get("selection_metric", "nmi"),
            "config": self.config,
        }
        with (root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        if output.get("save_model", True):
            # torch.save(self.global_model.state_dict(), root / "model.pt")
            pass
        return summary
