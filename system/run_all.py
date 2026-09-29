import argparse
import json

from config import DEFAULT_CONFIG_DIR, apply_overrides, load_config, resolve_project_path
from main import run
from summarize_results import write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run all federated clustering datasets")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summaries = {}
    for path in sorted(DEFAULT_CONFIG_DIR.glob("*.json")):
        overrides = list(args.override) + [f'training.device="{args.device}"']
        config = apply_overrides(load_config(path), overrides)
        dataset_path = resolve_project_path(config["dataset"]["file"])
        if not dataset_path.exists():
            print(f"Skipping {path.stem}: dataset file not found: {dataset_path}")
            continue
        print(f"\n{'=' * 20} {path.stem} {'=' * 20}")
        summaries[path.stem] = run(config)["metrics"]
    summary_path, _rows = write_summary(
        "results",
        expected_datasets=summaries,
        title="Tuned Configuration Results",
    )
    print("\nFinal metrics")
    print(json.dumps(summaries, indent=2))
    print(f"Markdown summary: {summary_path}")
