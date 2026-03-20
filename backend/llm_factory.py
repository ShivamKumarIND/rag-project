"""Dynamically load LLM instances from llms.json configuration."""

import importlib
import os
from typing import Any

from config import get_llm_config

# Cache instantiated LLMs to avoid re-initializing on every request
_LLM_CACHE: dict[str, Any] = {}

# Keys that are NOT constructor parameters
_NON_CONSTRUCTOR_KEYS = {"import_module", "import_class", "display_name", "max_input_chars", "config"}


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve any value prefixed with 'ENV:' from environment variables."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("ENV:"):
            env_var = value[4:]
            env_value = os.environ.get(env_var)
            if env_value is None:
                raise EnvironmentError(
                    f"Environment variable '{env_var}' is not set (required by LLM config key '{key}')"
                )
            resolved[key] = env_value
        else:
            resolved[key] = value
    return resolved


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert string booleans and strip non-constructor keys."""
    cleaned = {}
    for key, value in config.items():
        # Convert string booleans
        if isinstance(value, str) and value.lower() in ("true", "false"):
            cleaned[key] = value.lower() == "true"
        else:
            cleaned[key] = value
    return cleaned


def get_llm_instance(llm_id: str):
    """Get or create a cached LangChain chat model instance for the given LLM ID."""
    if llm_id in _LLM_CACHE:
        return _LLM_CACHE[llm_id]

    llm_config = get_llm_config(llm_id)

    import_module = llm_config["import_module"]
    import_class = llm_config["import_class"]

    # Get constructor kwargs from the "config" sub-key
    constructor_kwargs = dict(llm_config.get("config", {}))

    # Resolve ENV: prefixed values
    constructor_kwargs = _resolve_env_vars(constructor_kwargs)

    # Clean up string booleans
    constructor_kwargs = _clean_config(constructor_kwargs)

    # Dynamically import and instantiate
    module = importlib.import_module(import_module)
    cls = getattr(module, import_class)
    instance = cls(**constructor_kwargs)

    _LLM_CACHE[llm_id] = instance
    return instance
