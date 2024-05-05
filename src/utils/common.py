from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from yaml import load, Loader


# ===== Load file functions =====
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
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            item = list(map(int, line.split("\t")))
            head, rel, tail, time = item[:4]     # Filter 5th column if exists
            result.append([head, rel, tail, time])
    return result


def read_dict_file(
        fp: str,
        recover_space: bool = True,
        remove_bracket: bool = False,
        capitalize: bool = False,
        eventcode_dict: Dict[str, str] = None
) -> Dict[str, int]:
    """Read dictionary file."""
    result = {}
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            word, index = line.strip().split("\t")[:2]
            if eventcode_dict is not None:
                word = eventcode_dict[word]
            if recover_space:
                word = word.replace("_", " ")
            if remove_bracket:
                word = remove_brackets(word)
            if capitalize:
                word = word.capitalize()
            result.setdefault(word, int(index))
    return result


# ===== String functions =====
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


# ===== Time functions =====
YEAR = "year"
DAY = "day"
MINUTE_15 = "15min"


def time_id2str(
        idx: int,
        base_time: datetime,
        time_unit: str,
) -> str:
    """Convert time id to string."""
    if time_unit == YEAR:
        time = str(base_time.year + idx)
    elif time_unit == MINUTE_15:
        time = str(base_time + timedelta(minutes=15) * idx)
    else:  # time_unit == DAY:
        time = str((base_time + timedelta(days=idx)).date())
    return time


def time_str2id(
        time: str,
        base_time: datetime,
        time_unit: str,
) -> int:
    """Convert time string to id."""
    if time_unit == YEAR:
        idx = int(time) - base_time.year
    elif time_unit == MINUTE_15:
        idx = (datetime.fromisoformat(time) - base_time) // timedelta(minutes=15)
    else:  # time_unit == DAY:
        idx = (datetime.fromisoformat(time) - base_time) // timedelta(days=1)
    return idx
