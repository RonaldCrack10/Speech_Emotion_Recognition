"""Unified CLI entrypoint for the Speech Emotion Recognition project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="app",
        choices=["app", "eval", "experiments"],
        help="Run the Gradio demo (app), checkpoint evaluation (eval), or conference ablations (experiments).",
    )
    args = parser.parse_args()

    if args.command == "app":
        from app import demo, get_predictor

        get_predictor()
        demo.launch()
    elif args.command == "eval":
        from evaluation import run_evaluation

        run_evaluation()
    elif args.command == "experiments":
        from experiments import run_all_experiments

        run_all_experiments()


if __name__ == "__main__":
    main()
