import json
from pathlib import Path

import matplotlib

# The visualizer is also used on headless training servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ("acc", "nmi", "ari")
LOSS_KEYS = ("loss", "reconstruction", "consistency", "clustering", "balance")
COLORS = {
    "acc": "#2563eb",
    "nmi": "#059669",
    "ari": "#dc2626",
    "loss": "#111827",
    "reconstruction": "#2563eb",
    "consistency": "#7c3aed",
    "clustering": "#ea580c",
    "balance": "#0891b2",
}


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_history(history):
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must contain a non-empty list of round records")
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict) or "round" not in record or "train" not in record:
            raise ValueError(f"Invalid history record at position {index}")


PHASE_BOUNDARY_STYLES = (
    ("joint_pretrain", "Joint pretraining starts", "#7c3aed", "-."),
    ("clustering", "Formal clustering starts", "#6b7280", "--"),
)


def _phase_boundaries(history):
    """Return boundaries for every trainable clustering phase in the history."""
    boundaries = []
    for phase, label, color, linestyle in PHASE_BOUNDARY_STYLES:
        for record in history:
            if record.get("phase") == phase:
                boundaries.append(
                    (float(record["round"]) - 0.5, label, color, linestyle)
                )
                break
    return sorted(boundaries, key=lambda item: item[0])


def _decorate_axis(ax, history, best_round=None):
    for boundary, label, color, linestyle in _phase_boundaries(history):
        ax.axvline(
            boundary,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    if best_round is not None:
        ax.axvline(float(best_round), color="#f59e0b", linestyle=":", linewidth=1.5, label="Best round")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Communication round")


def _plot_metrics(history, output_path, dataset, best_round, dpi):
    figure, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    for key in METRIC_KEYS:
        rounds, values = [], []
        for record in history:
            value = record.get("clustering", {}).get(key)
            if value is not None:
                rounds.append(record["round"])
                values.append(value)
        if rounds:
            ax.plot(rounds, values, marker="o", markersize=4, linewidth=2, color=COLORS[key], label=key.upper())
    _decorate_axis(ax, history, best_round)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{dataset} clustering metrics")
    ax.legend(loc="best")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_losses(history, output_path, dataset, best_round, dpi):
    rounds = np.asarray([record["round"] for record in history], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True, sharex=True)
    for ax, key in zip(axes.flat, LOSS_KEYS):
        values = np.asarray([record["train"].get(key, np.nan) for record in history], dtype=float)
        ax.plot(rounds, values, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[key])
        _decorate_axis(ax, history, best_round)
        ax.set_title("Total loss" if key == "loss" else f"{key.capitalize()} loss")
        ax.set_ylabel("Loss")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center", frameon=False)
    figure.suptitle(f"{dataset} training losses", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def visualize_history(history_path, output_directory, summary_path=None, image_format="png", dpi=160):
    """Render metric and loss curves from one completed run.

    Returns a mapping containing the generated image paths.
    """
    history_path = Path(history_path)
    history = _load_json(history_path)
    _validate_history(history)

    summary = {}
    if summary_path is not None and Path(summary_path).exists():
        summary = _load_json(summary_path)
    dataset = summary.get("dataset", history_path.parent.name)
    best_round = summary.get("best_round")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = image_format.lower().lstrip(".")
    metric_path = output_directory / f"clustering_metrics.{suffix}"
    loss_path = output_directory / f"training_losses.{suffix}"
    _plot_metrics(history, metric_path, dataset, best_round, dpi)
    _plot_losses(history, loss_path, dataset, best_round, dpi)
    return {"metrics": metric_path, "losses": loss_path}
