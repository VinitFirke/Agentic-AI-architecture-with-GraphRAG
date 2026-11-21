"""Load the YAML config and the secrets/hosts from .env into one settings object."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "assembly.yaml"


@dataclass(frozen=True)
class Settings:
    """Merged view of configs/assembly.yaml and the environment."""

    raw: dict[str, Any]
    ollama_host: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, relative: str) -> Path:
        """Resolve a path from the config relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(config_path: str | Path = DEFAULT_CONFIG) -> Settings:
    """Read the YAML config and .env. Raises KeyError if a required env var is missing."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(
        raw=raw,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ["NEO4J_PASSWORD"],
    )
