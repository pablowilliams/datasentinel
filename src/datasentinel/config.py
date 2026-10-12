"""Runtime configuration, read once from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATASENTINEL_", env_file=".env", extra="ignore")

    dsn: str = "postgresql+psycopg://localhost:5432/datasentinel"
    seed: int = 20261012
    generated_dir: Path = ROOT / "data" / "generated"
    config_dir: Path = ROOT / "config"


@lru_cache
def settings() -> Settings:
    return Settings()
