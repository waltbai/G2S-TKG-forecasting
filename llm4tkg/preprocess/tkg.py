import os
import random
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple

from llm4tkg.preprocess.fact import Fact
from llm4tkg.utils.config import load_config


__all__ = [
    "TemporalKG",
]


def _read_index_file(fp: str) -> List[List[int]]:
    """Read index file."""
    result = []
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            item = list(map(int, line.split("\t")))
            result.append(item[:4])     # Filter 5th column if exists
    return result


def _read_dict_file(
        fp: str,
        recover_space: bool = True,
) -> Dict[int, str]:
    """Read dictionary file."""
    result = {}
    with open(fp, 'r', encoding="utf-8") as f:
        for line in f:
            word, index = line.strip().split("\t")[:2]
            if recover_space:
                word = word.replace("_", " ")
            result.setdefault(int(index), word)
    return result


def _remove_brackets(ent: str) -> str:
    """Remove brackets in entity name."""
    # Simple strategy that cannot handle nested brackets,
    # however, it seems enough.
    start_idx = ent.find("(")
    end_idx = ent.find(")")
    if start_idx != -1 and end_idx != -1:
        return ent.replace(ent[start_idx:end_idx+1], "").strip()
    else:
        return ent


def _idx2facts(
        indices: List[List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        base_time: str,
        time_unit: str = "day",
) -> List[Fact]:
    """Convert indices to quadruple."""
    result = []
    for head_idx, rel_idx, tail_idx, time_idx in indices:
        head = id2entity[head_idx]
        rel = id2relation[rel_idx]
        tail = id2entity[tail_idx]
        if time_unit == "day":
            # starts from 1
            time = str(date.fromisoformat(base_time) + timedelta(days=time_idx - 1))
        elif time_unit == "hour":
            time = str(date.fromisoformat(base_time) + timedelta(hours=time_idx))
        elif time_unit == "15min":
            # starts from 0
            time = str(datetime.fromisoformat(base_time) + timedelta(minutes=time_idx * 15))
        elif time_unit == "year":
            # starts from 0
            time = str(int(base_time) + time_idx)
        else:
            time = "unknown"
        quad = Fact(
            head=head,
            rel=rel,
            tail=tail,
            time=time,
            head_idx=head_idx,
            rel_idx=rel_idx,
            tail_idx=tail_idx,
            time_idx=time_idx,
        )
        result.append(quad)
    return result


@dataclass
class TemporalKG:
    """Universal class for Temporal Knowledge Graph."""
    # Basic graph elements
    entities: List[str]
    relations: List[str]
    # Datasets
    train_set: List[Fact]
    valid_set: List[Fact]
    test_set: List[Fact]
    # Time information
    base_time: str
    time_unit: str
    # Dataset name
    dataset_name: str = ""
    # Inner indices for search
    _head_history: Dict[str, List[Fact]] = None
    _tail_history: Dict[str, List[Fact]] = None
    _both_history: Dict[str, List[Fact]] = None
    _head_rel_history: Dict[Tuple[str, str], List[Fact]] = None
    _tail_rel_history: Dict[Tuple[str, str], List[Fact]] = None

    def construct_indices(self) -> None:
        """Construct indices for speedup."""
        _head_history = {}
        _tail_history = {}
        _both_history = {}
        _head_rel_history = {}
        _tail_rel_history = {}
        facts = self.train_set + self.valid_set + self.test_set
        for fact in facts:
            _head_history.setdefault(fact.head, []).append(fact)
            _tail_history.setdefault(fact.tail, []).append(fact)
            _both_history.setdefault(fact.head, []).append(fact)
            _both_history.setdefault(fact.tail, []).append(fact)
            _head_rel_history.setdefault((fact.head, fact.rel), []).append(fact)
            _tail_rel_history.setdefault((fact.tail, fact.rel), []).append(fact)
        self._head_history = _head_history
        self._tail_history = _tail_history
        self._both_history = _both_history
        self._head_rel_history = _head_rel_history
        self._tail_rel_history = _tail_rel_history

    def clear_indices(self):
        """Clear all indices."""
        self._head_history = None
        self._tail_history = None
        self._both_history = None
        self._head_rel_history = None
        self._tail_rel_history = None

    def find_history_by_head(self, head: str) -> List[Fact]:
        """Find historical facts by head entity."""
        if self._head_history is not None:
            return self._head_history[head]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.head == head]

    def find_history_by_tail(self, tail: str) -> List[Fact]:
        """Find historical facts by tail entity."""
        if self._tail_history is not None:
            return self._tail_history[tail]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.tail == tail]

    def find_history_by_both(self, ent: str) -> List[Fact]:
        """Find historical facts by both head and tail entity."""
        if self._both_history is not None:
            return self._both_history[ent]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.head == ent or _.tail == ent]

    def find_history_by_head_rel(
            self,
            head: str,
            rel: str,
    ) -> List[Fact]:
        """Find historical facts by head and relation."""
        if self._head_rel_history is not None:
            return self._head_rel_history[(head, rel)]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.head == head and _.rel == rel]

    def find_history_by_tail_rel(
            self,
            tail: str,
            rel: str,
    ) -> List[Fact]:
        """Find historical facts by tail and relation."""
        if self._tail_rel_history is not None:
            return self._tail_rel_history[(tail, rel)]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.tail == tail and _.rel == rel]

    def statistic(self):
        """Statistic dataset."""
        num_entities = len(self.entities)
        num_relations = len(self.relations)
        num_train = len(self.train_set)
        num_valid = len(self.valid_set)
        num_test = len(self.test_set)
        # examples = random.sample(self.train_set, 10)
        # examples_str = "\n".join([str(_) for _ in examples])
        report = (
            f"Total number of entities: {num_entities}\n"
            f"Total number of relations: {num_relations}\n"
            f"Total number of train facts: {num_train}\n"
            f"Total number of valid facts: {num_valid}\n"
            f"Total number of test facts: {num_test}\n\n"
            # f"Some examples:\n"
            # f"{examples_str}"
        )
        return report

    @classmethod
    def load(cls,
             dataset_name: str,
             data_dir: str=None,
             verbose: bool = False,
    ):
        """Construct a temporal KG dataset."""
        # Load basic config
        config = load_config("config/dataset.yml")
        if data_dir is None:
            data_dir = config["data_dir"]
        # Set paths
        dataset_dir = os.path.join(data_dir, dataset_name)
        train_path = os.path.join(dataset_dir, "train.txt")
        valid_path = os.path.join(dataset_dir, "valid.txt")
        test_path = os.path.join(dataset_dir, "test.txt")
        entity2id_path = os.path.join(dataset_dir, "entity2id.txt")
        relation2id_path = os.path.join(dataset_dir, "relation2id.txt")
        # Load original files
        # In previous works, train/valid/unit_test files are processed index files.
        train_set_idx = _read_index_file(train_path)
        valid_set_idx = _read_index_file(valid_path)
        test_set_idx = _read_index_file(test_path)
        id2entity = _read_dict_file(entity2id_path)
        id2relation = _read_dict_file(relation2id_path)
        if dataset_name == "GDELT":
            # GDELT dataset use CAMEO event code as relation name
            # Convert it back to actual relation name
            cameo_config = load_config("config/cameo.yml")
            for rid in id2relation.keys():
                id2relation[rid] = cameo_config["event_code"][id2relation[rid]]
            for eid in id2entity.keys():
                id2entity[eid] = _remove_brackets(id2entity[eid]).capitalize()
        entities = [_ for _ in id2entity.values()]
        relations = [_ for _ in id2relation.values()]
        # Convert indices to quadruples
        base_time = str(config["datasets"][dataset_name]["base_time"])
        time_unit = config["datasets"][dataset_name]["time_unit"]
        # Construct datasets
        train_set = _idx2facts(
            indices=train_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        valid_set = _idx2facts(
            indices=valid_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        test_set = _idx2facts(
            indices=test_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
        )
        obj = cls(
            entities=entities,
            relations=relations,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            base_time=base_time,
            time_unit=time_unit,
            dataset_name=dataset_name,
        )
        obj.construct_indices()
        if verbose:
            print(obj.statistic())
        return obj
