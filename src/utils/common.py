import os
from typing import Dict, List

from yaml import load, Loader


def load_config(
        config_path: str,
) -> Dict:
    """Load config file."""
    with open(config_path, "r") as f:
        config = load(f, Loader=Loader)
    return config


def read_index_file(fp: str) -> List[List[int]]:
    """Read index file."""
    result = []
    if os.path.exists(fp):
        with open(fp, 'r', encoding="utf-8") as f:
            for line in f:
                item = line.split("\t")
                head, rel, tail, time = item[:4]     # Filter 5th column if exists
                result.append([head, rel, tail, time])
    return result


def read_dict_file(
        fp: str,
) -> Dict[str, int]:
    """Read dictionary file."""
    result = {}
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            word, index = line.strip().split("\t")
            result.setdefault(word, index)
    return result
