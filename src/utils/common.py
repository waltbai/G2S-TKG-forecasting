import importlib
from typing import Callable, Dict, Tuple, List, Any


def import_lib(path: str) -> Callable:
    """Import function or class via path."""
    module, target = path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), target)


def card2ord(n: int) -> str:
    """Convert a cardinal number to ordinal number."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return f"{n}{suffix}"


def format_params(params: List[Tuple[str, Any]]) -> str:
    """Format parameter string."""
    num_char_key = max([len(_[0]) for _ in params]) + 1
    num_char_value = max([len(str(_[1])) for _ in params])
    result = ""
    for key, value in params:
        result += f"{key.ljust(num_char_key)}: {str(value).rjust(num_char_value)}\n"
    return result
