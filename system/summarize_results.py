import argparse
import json
from pathlib import Path

from config import PROJECT_ROOT, resolve_project_path


METRIC_NAMES = ("acc", "nmi", "ari")


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_record(history, round_number):
    for record in history:
        if int(record["round"]) == int(round_number) and "clustering" in record:
            return record["clustering"]
    raise ValueError(f"No clustering metrics found for round {round_number}")


def collect_result(dataset_dir):
    summary_path = dataset_dir / "summary.json"
    history_path = dataset_dir / "history.json"
    if not summary_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"Missing summary.json or history.json in {dataset_dir}")

    summary = _load_json(summary_path)
    history = _load_json(history_path)
    if not history:
        raise ValueError(f"Empty history: {history_path}")

    best_round = int(summary["best_round"])
    configured_rounds = int(summary["config"]["training"]["rounds"])
    best_metrics = _metric_record(history, best_round)
    last_metrics = _metric_record(history, configured_rounds)
    return {
        "dataset": summary["dataset"],
        "seed": int(summary["config"]["training"]["seed"]),
        "best_round": best_round,
        "configured_rounds": configured_rounds,
        "best": {name: float(best_metrics[name]) for name in METRIC_NAMES},
        "last": {name: float(last_metrics[name]) for name in METRIC_NAMES},
        "delta": {
            name: float(last_metrics[name]) - float(best_metrics[name])
            for name in METRIC_NAMES
        },
    }


def write_summary(root, expected_datasets=None, title="Federated Multi-View Clustering Results"):
    root = resolve_project_path(root)
    dataset_names = (
        sorted(expected_datasets)
        if expected_datasets is not None
        else sorted(path.name for path in root.iterdir() if path.is_dir())
    )
    rows = [collect_result(root / name) for name in dataset_names]
    if not rows:
        raise ValueError(f"No dataset results found in {root}")

    lines = [
        f"# {title}",
        "",
        f"- Results directory: `{root}`",
        "- Selection metric: NMI",
        "- Metric changes use `last epoch - best epoch`; negative values indicate degradation.",
        "",
        "| Dataset | Seed | ACC | NMI | ARI | Best epoch | Configured epochs | ΔACC | ΔNMI | ΔARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best"]
        delta = row["delta"]
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {best['acc']:.6f} | "
            f"{best['nmi']:.6f} | {best['ari']:.6f} | {row['best_round']} | "
            f"{row['configured_rounds']} | {delta['acc']:+.6f} | "
            f"{delta['nmi']:+.6f} | {delta['ari']:+.6f} |"
        )

    output_path = root / "summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize completed clustering runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--title", default="Federated Multi-View Clustering Results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path, _rows = write_summary(args.results_dir, args.datasets, args.title)
    print(path)
