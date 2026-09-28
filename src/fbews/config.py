"""Configuration loading for FBEWS.

Every pipeline stage receives its settings from here. Paths are resolved
relative to the project root so nothing is hardcoded, and credentials are
read from environment variables only (never from the YAML file).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("FBEWS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
REGIONS_CONFIG = PROJECT_ROOT / "configs" / "regions.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    # -- dict-ish access -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve one of the configured directories, creating it on demand."""
        rel = self.raw["paths"][key]
        p = (self.root / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- convenience ---------------------------------------------------------
    @property
    def data_mode(self) -> str:
        return os.environ.get("FBEWS_DATA_MODE", self.raw.get("data_mode", "sandbox"))

    @property
    def is_sandbox(self) -> bool:
        return self.data_mode == "sandbox"

    @property
    def lead_days(self) -> list[int]:
        return list(self.raw["forecast"]["lead_days"])

    @property
    def domain(self) -> dict[str, Any]:
        return self.raw["domain"]

    @property
    def resolution(self) -> float:
        return float(self.raw["grid"]["resolution"])

    @property
    def sandbox_banner(self) -> str:
        return self.raw["sandbox"]["label"]


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.environ.get("FBEWS_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=PROJECT_ROOT)


@lru_cache(maxsize=2)
def load_regions(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    cfg_path = Path(path) if path else REGIONS_CONFIG
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)["regions"]


def credential(name: str, required: bool = False) -> str | None:
    """Read a credential from the environment.

    Credentials are never stored in configuration files or in the repository.
    """
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(
            f"Missing credential {name}. Copy .env.example to .env and set it, "
            f"then `export $(grep -v '^#' .env | xargs)`."
        )
    return value
