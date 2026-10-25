"""YAML configuration loading and validation."""

from pathlib import Path

import yaml


def load_config(path):
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root in {config_path} must be a mapping")
    return config
