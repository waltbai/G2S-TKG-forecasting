import importlib
from typing import Callable


def import_lib(path: str) -> Callable:
    """Import function or class via path."""
    module, target = path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), target)

