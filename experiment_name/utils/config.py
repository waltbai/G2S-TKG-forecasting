from typing import Dict

from yaml import load, Loader


def load_config(
        config_path: str,
) -> Dict:
    """Load config file."""
    with open(config_path, "r") as f:
        config = load(f, Loader=Loader)
    return config
