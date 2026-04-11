"""Main entry point for ML pipeline application."""

import argparse
import logging

from src import config
from src.app import app


def main() -> None:
    """Execute the main entry point of the application.

    Load configuration, set up logging, and run the ML pipeline.
    """
    parser = argparse.ArgumentParser(description="ML Pipeline Application")
    parser.add_argument(
        "--mode",
        choices=["once", "schedule"],
        default="once",
        help="Run pipeline once or start daily scheduler",
    )
    args = parser.parse_args()

    cfg = config.load_config()
    config.setup_logger(cfg)
    logging.info(f"Running with: Config( {cfg} )")

    app(cfg, mode=args.mode)


if __name__ == "__main__":
    main()
