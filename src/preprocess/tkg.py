import logging
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple

from src.evaluation import Query
from src.preprocess.fact import Fact
from src.utils.common import format_params
from src.utils.config import load_config


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
            # starts from 0
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


def _construct_queries(dataset: List[Fact]) -> List[Query]:
    """Construct queries from dataset."""
    query_dict = {}
    for query_direction in ["tail", "head"]:
        for fact in dataset:
            query_rel = fact.rel
            query_rel_idx = fact.rel_idx
            query_time = fact.time
            query_time_idx = fact.time_idx
            if query_direction == "tail":
                query_entity = fact.head
                query_entity_idx = fact.head_idx
                answer = fact.tail
                answer_idx = fact.tail_idx
            else:   # query_direction == "head"
                query_entity = fact.tail
                query_entity_idx = fact.tail_idx
                answer = fact.head
                answer_idx = fact.head_idx
            key = (query_entity, query_rel, query_time, query_direction)
            if key in query_dict:
                query_dict[key].answers.append(answer)
                query_dict[key].answers_idx.append(answer_idx)
            else:
                query_dict.setdefault(
                    key,
                    Query(
                        entity=query_entity,
                        rel=query_rel,
                        answers=[answer],
                        time=query_time,
                        direction=query_direction,
                        entity_idx=query_entity_idx,
                        rel_idx=query_rel_idx,
                        answers_idx=[answer_idx],
                        time_idx=query_time_idx,
                    )
                )
    # Sort queries by time
    keys = sorted(query_dict.keys(), key=lambda x: x[2])
    queries = [query_dict[key] for key in keys]
    return queries


class TemporalKG:
    """Universal class for Temporal Knowledge Graph."""

    def __init__(
            self,
            entities: List[str],
            relations: List[str],
            train_set: List[Fact],
            valid_set: List[Fact],
            test_set: List[Fact],
            base_time: str,
            time_unit: str,
            dataset_name: str = "",
            indices: bool = True,
            train_queries: bool = False,
            valid_queries: bool = False,
    ):
        # Assign Arguments
        self.entities = entities
        self.relations = relations
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.base_time = base_time
        self.time_unit = time_unit
        self.dataset_name = dataset_name
        # Construct inner variables
        self.train_queries = None
        self.valid_queries = None
        self.test_queries = _construct_queries(test_set)
        if train_queries:
            self.train_queries = _construct_queries(train_set)
        if valid_queries:
            self.valid_queries = _construct_queries(valid_set)
        # Inner indices for search
        self._head_history = {}
        self._tail_history = {}
        self._both_history = {}
        self._head_rel_history = {}
        self._tail_rel_history = {}
        if indices:
            self.construct_indices()
        # Logger
        self.logger = logging.getLogger("TKG")

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
        self._head_history = {}
        self._tail_history = {}
        self._both_history = {}
        self._head_rel_history = {}
        self._tail_rel_history = {}

    def find_history_by_head(self, head: str) -> List[Fact]:
        """Find historical facts by head entity."""
        if self._head_history:
            return self._head_history[head]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.head == head]

    def find_history_by_tail(self, tail: str) -> List[Fact]:
        """Find historical facts by tail entity."""
        if self._tail_history:
            return self._tail_history[tail]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.tail == tail]

    def find_history_by_both(self, ent: str) -> List[Fact]:
        """Find historical facts by both head and tail entity."""
        if self._both_history:
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
        if self._head_rel_history:
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
        if self._tail_rel_history:
            return self._tail_rel_history[(tail, rel)]
        else:
            facts = self.train_set + self.valid_set + self.test_set
            return [_ for _ in facts if _.tail == tail and _.rel == rel]

    def statistic(self):
        """Statistic dataset."""
        params = [
            ("# entities", f"{len(self.entities):,}"),
            ("# relations", f"{len(self.relations):,}"),
            ("# train facts", f"{len(self.train_set):,}"),
            ("# valid facts", f"{len(self.valid_set):,}"),
            ("# test facts", f"{len(self.test_set):,}"),
        ]
        if self.train_queries is not None:
            params.append(("# train queries", f"{len(self.train_queries):,}"))
        else:
            pass
        if self.valid_queries is not None:
            params.append(("# valid queries", f"{len(self.valid_queries):,}"))
        else:
            pass
        params.append(("# test queries", f"{len(self.test_queries):,}"))
        return format_params(params)

    @classmethod
    def load(cls,
             dataset_name: str,
             data_dir: str=None,
             verbose: bool = False,
             train_queries: bool = False,
             valid_queries: bool = False,
    ):
        """Construct a temporal KG dataset."""
        logger = logging.getLogger("TKG.load")
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
        if verbose:
            logger.info("Facts loaded.")
        id2entity = _read_dict_file(
            entity2id_path,
            recover_space=True,
        )
        id2relation = _read_dict_file(
            relation2id_path,
            recover_space=True,
        )
        if verbose:
            logger.info("Dicts loaded.")
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
            train_queries=train_queries,
            valid_queries=valid_queries,
        )
        obj.construct_indices()
        if verbose:
            logger.info(f"TKG statistics:\n{obj.statistic()}")
        return obj
