#!/usr/bin/env python3
"""Entry point: runs the full European power fair-value forecasting pipeline."""
import sys

from cobblestone.config import Config
from cobblestone.pipeline import Pipeline


def main() -> None:
    config = Config.from_env()

    # Validate only what is strictly required
    if not config.openai_api_key:
        print(
            "WARNING: OPENAI_API_KEY not set — LLM QA step will be skipped.\n"
            "Add it to .env to enable AI-accelerated QA rule generation.",
            file=sys.stderr,
        )

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
