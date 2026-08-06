"""Pipeline configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


@dataclass
class Config:
    anthropic_api_key: str
    entsoe_api_key: str
    market: str
    start_date: str
    end_date: str
    llm_model: str = "claude-haiku-4-5"
    random_seed: int = 42
    cv_folds: int = 12
    test_months: int = 3
    force_refetch: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            entsoe_api_key=os.environ.get("ENTSOE_API_KEY", ""),
            market=os.environ.get("MARKET", "DE"),
            start_date=os.environ.get("START_DATE", "2022-01-01"),
            end_date=os.environ.get("END_DATE", "2025-12-31"),
            force_refetch=os.environ.get("FORCE_REFETCH", "false").lower() == "true",
        )

def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
