import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.utils.common import (
    load_config,
    read_index_file,
    read_dict_file,
    time_id2str, read_rule_file,
)
from src.utils.fact import Fact


@dataclass
class TKG:
    """Temporal Knowledge Graph class.

    Args:
        name (str): Name of the Knowledge Graph.
        train_facts (List[Fact]):
            a list of training facts ordered by time.
        valid_facts (List[Fact]):
            a list of validation facts ordered by time.
        test_facts (List[Fact]):
            a list of test facts ordered by time.
        base_time (datetime):
            the start time of the TKG.
        time_unit (timedelta):
            the time granularity of raw data.
        entities (List[str]):
            the list of entities.
        entity2id (Dict[str, int]):
            mapping between entities and ids.
        relations (List[str]):
            the list of relations.
        relation2id (Dict[str, int]):
            mapping between relations and ids.
        search_history (Dict[str, Any]):
            the indices to speed up searching,
        time2id (Dict[str, int]):
            mapping between time and ids.
        id2time (Dict[int, str]):
            mapping between ids and times.
    """
    name: str
    # Facts
    train_facts: List[Fact]
    valid_facts: List[Fact]
    test_facts: List[Fact]
    # Time
    base_time: datetime
    time_unit: str
    # Entity
    entities: List[str]
    entity2id: Dict[str, int]
    # Relation
    relations: List[str]
    relation2id: Dict[str, int]
    # Speedup
    search_history: Dict[str, Any]
    time2id: Dict[str, int]
    id2time: Dict[int, str]
    rules: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset_dir: str, dataset: str):
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
        rule_path = os.path.join(f"data/{dataset}-rules.json")
        # Load original files
        # In previous works, train/valid/unit_test files are processed index files.
        train_set_idx = read_index_file(train_path)
        valid_set_idx = read_index_file(valid_path)
        test_set_idx = read_index_file(test_path)
        if dataset == "GDELT":
            entity2id = read_dict_file(
                entity2id_path,
                remove_bracket=True,
                capitalize=True,
            )
            eventcode_dict = load_config("config/cameo.yml")["eventcode"]
            relation2id = read_dict_file(
                relation2id_path,
                eventcode_dict=eventcode_dict,
            )
        else:
            entity2id = read_dict_file(entity2id_path)
            relation2id = read_dict_file(relation2id_path)
        entities = [_[0] for _ in sorted(entity2id.items(), key=lambda x: x[1])]
        relations = [_[0] for _ in sorted(relation2id.items(), key=lambda x: x[1])]
        rules = read_rule_file(rule_path)
        # Normalize time and record time indices
        dataset_config = config[dataset]
        time_unit = dataset_config["time_unit"]
        base_time = datetime.fromisoformat(str(dataset_config["base_time"]))
        norm_factor = dataset_config["norm_factor"]
        adjust_time = dataset_config["adjust_time"]
        time2id = {}
        id2time = {}
        for ids in train_set_idx + valid_set_idx + test_set_idx:
            ids[3] = ids[3] // norm_factor
            if adjust_time:
                ids[3] -= 1
            time = time_id2str(ids[3], base_time, time_unit)
            time2id.setdefault(time, ids[3])
            id2time.setdefault(ids[3], time)
        # Construct tkg
        params = {
            "entities": entities,
            "relations": relations,
            "id2time": id2time,
        }
        train_facts = [Fact.from_ids(ids, **params) for ids in train_set_idx]
        valid_facts = [Fact.from_ids(ids, **params) for ids in valid_set_idx]
        test_facts = [Fact.from_ids(ids, **params) for ids in test_set_idx]
        facts = train_facts + valid_facts + test_facts
        search_history = construct_search_histories(facts)
        return cls(
            name=dataset,
            train_facts=train_facts,
            valid_facts=valid_facts,
            test_facts=test_facts,
            base_time=base_time,
            time_unit=time_unit,
            entities=entities,
            entity2id=entity2id,
            relations=relations,
            relation2id=relation2id,
            search_history=search_history,
            time2id=time2id,
            id2time=id2time,
            rules=rules
        )


def construct_search_histories(
        facts: List[Fact],
) -> Dict[str, Any]:
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
