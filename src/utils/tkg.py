import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.utils.common import load_config, remove_brackets
from src.utils.fact import Fact


@dataclass
class TKG:
    """Temporal Knowledge Graph class."""
    name: str
    # Facts
    train_facts: List[Fact]
    valid_facts: List[Fact]
    test_facts: List[Fact]
    # Time
    base_time: datetime
    time_unit: timedelta
    time_precision: str
    # Entity
    entities: List[str]
    entity2id: Dict[str, int]
    # Relation
    relations: List[str]
    relation2id: Dict[str, int]
    # Speedup
    search_history: Dict[str, Any]
    time2id: Dict[str, int]
    @classmethod
    def load(cls,
             dataset_dir: str,
             dataset: str,
             ):
        """Load TKG from file."""
        # Load basic config
        config = load_config("config/dataset.yml")
        # Set paths
        dataset_path = os.path.join(dataset_dir, dataset)
        train_path = os.path.join(dataset_path, "train.txt")
        valid_path = os.path.join(dataset_path, "valid.txt")
        test_path = os.path.join(dataset_path, "test.txt")
        entity2id_path = os.path.join(dataset_path, "entity2id.txt")
        relation2id_path = os.path.join(dataset_path, "relation2id.txt")
        # Load original files
        # In previous works, train/valid/unit_test files are processed index files.
        adjust_time = dataset in ["ICEWS14", "ICEWS05-15"]
        train_set_idx = _read_index_file(train_path, adjust_time=adjust_time)
        valid_set_idx = _read_index_file(valid_path, adjust_time=adjust_time)
        test_set_idx = _read_index_file(test_path, adjust_time=adjust_time)
        if dataset == "GDELT":
            entity2id = _read_dict_file(
                entity2id_path,
                remove_bracket=True,
                capitalize=True,
            )
            eventcode_dict = load_config("config/cameo.yml")["eventcode"]
            relation2id = _read_dict_file(
                relation2id_path,
                eventcode_dict=eventcode_dict,
            )
        else:
            entity2id = _read_dict_file(entity2id_path)
            relation2id = _read_dict_file(relation2id_path)
        entities = [_[0] for _ in sorted(entity2id.items(), key=lambda x: x[1])]
        relations = [_[0] for _ in sorted(relation2id.items(), key=lambda x: x[1])]
        # Convert indices to facts
        dataset_config = config[dataset]
        base_time = _str2datetime(dataset_config["base_time"])
        time_unit = _get_timedelta(dataset_config["time_unit"])
        time_precision = dataset_config["time_precision"]
        # Construct tkg
        train_facts = _idx2facts(
            indices=train_set_idx,
            entities=entities,
            relations=relations,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        valid_facts = _idx2facts(
            indices=valid_set_idx,
            entities=entities,
            relations=relations,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        test_facts = _idx2facts(
            indices=test_set_idx,
            entities=entities,
            relations=relations,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        search_history = construct_search_histories(
            train_facts + valid_facts + test_facts
        )
        time2id = construct_time_index(
            facts=train_facts + valid_facts + test_facts,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        return cls(
            name=dataset,
            train_facts=train_facts,
            valid_facts=valid_facts,
            test_facts=test_facts,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
            entities=entities,
            entity2id=entity2id,
            relations=relations,
            relation2id=relation2id,
            search_history=search_history,
            time2id=time2id,
        )


def _read_index_file(
        fp: str,
        adjust_time: bool = False,
) -> List[List[int]]:
    """Read index file."""
    result = []
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            item = list(map(int, line.split("\t")))
            head, rel, tail, time = item[:4]     # Filter 5th column if exists
            if adjust_time:
                time -= 1
            result.append([head, rel, tail, time])
    return result


def _read_dict_file(
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


def _str2datetime(
        time_str: str,
        time_precision: str = "day",
) -> datetime:
    """Convert time string to datetime object."""
    if time_precision == "year":
        return datetime.strptime(time_str, "%Y")
    else:
        return datetime.fromisoformat(time_str)


def _datetime2str(
        time: datetime,
        time_precision: str = "day",
) -> str:
    """Convert datetime object to string."""
    if time_precision == "year":
        return str(time.year)
    elif time_precision == "day":
        return str(time.date())
    else:
        return str(time)


def _get_timedelta(time_unit: str) -> timedelta:
    """Get time delta object by time_unit."""
    # By default, time unit is 1 day.
    ret_val = timedelta(days=1)
    if time_unit == "year":
        ret_val = timedelta(days=365)
    elif time_unit == "hour":
        ret_val = timedelta(hours=1)
    elif time_unit == "15min":
        ret_val = timedelta(minutes=15)
    return ret_val


def _idx2facts(
        indices: List[List[int]],
        entities: List[str],
        relations: List[str],
        base_time: datetime,
        time_unit: timedelta,
        time_precision: str = "day",
) -> List[Fact]:
    """Convert indices to quadruple."""
    result = []
    for head_idx, rel_idx, tail_idx, time_idx in indices:
        head = entities[head_idx]
        rel = relations[rel_idx]
        tail = entities[tail_idx]
        time = _datetime2str(
            base_time + time_unit * time_idx,
            time_precision=time_precision,
        )
        quad = Fact(
            head=head,
            rel=rel,
            tail=tail,
            time=time,
        )
        result.append(quad)
    return result


def construct_search_histories(facts):
    """Construct history dictionaries for search."""
    search_history = {
        "head": {},
        "tail": {},
        "both": {},
        "head_rel": {},
        "tail_rel": {},
        "both_rel": {},
    }
    for fact in facts:
        search_history["head"].setdefault(fact.head, []).append(fact)
        search_history["tail"].setdefault(fact.tail, []).append(fact)
        search_history["both"].setdefault(fact.head, []).append(fact)
        search_history["both"].setdefault(fact.tail, []).append(fact)
        search_history["head_rel"].setdefault((fact.head, fact.rel), []).append(fact)
        search_history["tail_rel"].setdefault((fact.tail, fact.rel), []).append(fact)
        search_history["both_rel"].setdefault((fact.head, fact.rel), []).append(fact)
        search_history["both_rel"].setdefault((fact.tail, fact.rel), []).append(fact)
    return search_history


def construct_time_index(
        facts: List[Fact],
        base_time: datetime,
        time_unit: timedelta,
        time_precision: str,
) -> Dict[str, int]:
    """Construct time index."""
    time2id = {}
    for fact in facts:
        time_str = fact.time
        time = _str2datetime(time_str, time_precision)
        idx = (time - base_time) // time_unit
        time2id.setdefault(time_str, idx)
    return time2id
