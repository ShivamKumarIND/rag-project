"""Load and parse llms.json configuration."""

import json
import os
from pathlib import Path
from typing import Any

_CONFIG: dict | None = None
_CONFIG_PATH = Path(__file__).parent / "llms.json"


def _load_config() -> dict:
    """Load and cache the llms.json configuration."""
    global _CONFIG
    if _CONFIG is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
    return _CONFIG


def get_all_llms() -> list[dict[str, str]]:
    """Return a list of {id, display_name} for every configured LLM."""
    cfg = _load_config()
    result = []
    for key, value in cfg.items():
        if key == "managerLLM":
            continue
        if isinstance(value, dict) and "display_name" in value:
            result.append({"id": key, "display_name": value["display_name"]})
    return result


def get_llm_config(llm_id: str) -> dict[str, Any]:
    """Return the full config dict for a given LLM ID."""
    cfg = _load_config()
    if llm_id not in cfg:
        raise ValueError(f"LLM '{llm_id}' not found in llms.json")
    return cfg[llm_id]


def get_default_llm_id() -> str:
    """Return the default LLM ID from the managerLLM key."""
    cfg = _load_config()
    return cfg.get("managerLLM", "")
