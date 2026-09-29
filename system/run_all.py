import argparse
import json

from config import BACKUP_CONFIG_DIR, DEFAULT_CONFIG_DIR, apply_overrides, load_config
from main import run
from summarize_results import write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run all federated clustering datasets")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summaries = {}
    config_paths = sorted(DEFAULT_CONFIG_DIR.glob("*.json"))
    config_paths.extend(sorted(BACKUP_CONFIG_DIR.glob("*.json")))
    for path in config_paths:
        overrides = list(args.override) + [
            f'training.device="{args.device}"',
            f'output.directory={json.dumps(args.results_dir, ensure_ascii=False)}',
        ]
        config = apply_overrides(load_config(path), overrides)
        print(f"\n{'=' * 20} {path.stem} {'=' * 20}")
        summaries[path.stem] = run(config)["metrics"]
    summary_path, _rows = write_summary(
        args.results_dir,
        expected_datasets=summaries,
        title="Two-Stage Tuned Configuration Results",
    )
    print("\nFinal metrics")
    print(json.dumps(summaries, indent=2))
    print(f"Markdown summary: {summary_path}")
