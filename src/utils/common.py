import importlib
from typing import Callable


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
