from typing import Dict, List, Tuple, Any

from yaml import load, Loader


def load_config(
        config_path: str,
) -> Dict:
    """Load config file."""
    with open(config_path, "r") as f:
        config = load(f, Loader=Loader)
    return config


def remove_brackets(ent: str) -> str:
    """Remove brackets in entity name."""
    # Simple strategy that cannot handle nested brackets,
    # however, it seems enough.
    start_idx = ent.find("(")
    end_idx = ent.find(")")
    if start_idx != -1 and end_idx != -1:
        return ent.replace(ent[start_idx:end_idx + 1], "").strip()
    else:
        return ent


def format_params(params: List[Tuple[str, Any]]) -> str:
    """Format parameter string."""
    num_char_key = max([len(_[0]) for _ in params]) + 1
    num_char_value = max([len(str(_[1])) for _ in params])
    result = ""
    for key, value in params:
        result += f"{key.ljust(num_char_key)}: {str(value).rjust(num_char_value)}\n"
    return result
