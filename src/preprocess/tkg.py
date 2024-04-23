import logging
import os
from datetime import timedelta, datetime
from typing import List, Dict

from src.evaluation import Query
from src.preprocess.fact import Fact
from src.utils.common import format_params, card2ord
from src.utils.config import load_config


__all__ = [
    "TemporalKG",
]


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


def _idx2facts(
        indices: List[List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        base_time: datetime,
        time_unit: timedelta,
        time_precision: str = "day",
) -> List[Fact]:
    """Convert indices to quadruple."""
    result = []
    for head_idx, rel_idx, tail_idx, time_idx in indices:
        head = id2entity[head_idx]
        rel = id2relation[rel_idx]
        tail = id2entity[tail_idx]
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


def _construct_queries(dataset: List[Fact]) -> List[Query]:
    """Construct queries from dataset."""
    query_dict = {}
    for query_direction in ["tail", "head"]:
        for fact in dataset:
            query_rel = fact.rel
            query_time = fact.time
            if query_direction == "tail":
                query_entity = fact.head
                answer = fact.tail
            else:   # query_direction == "head"
                query_entity = fact.tail
                answer = fact.head
            key = (query_entity, query_rel, query_time, query_direction)
            if key in query_dict:
                query_dict[key].answers.append(answer)
            else:
                query_dict.setdefault(
                    key,
                    Query(
                        entity=query_entity,
                        rel=query_rel,
                        answers=[answer],
                        time=query_time,
                        direction=query_direction,
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
            base_time: datetime,
            time_unit: timedelta,
            time_precision: str = "day",
            dataset_name: str = "",
            anon_entity: str = None,
            anon_rel: str = None,
            anon_time: str = None,
            search_indices: bool = True,
            train_queries: bool = False,
            valid_queries: bool = False,
            entity2id: Dict[str, int] = None,
            relation2id: Dict[str, int] = None,
    ):
        # Assign Arguments
        self.entities = entities
        self.entity2id = entity2id or {v: k for k, v in enumerate(entities)}
        self.relations = relations
        self.relation2id = relation2id or {v: k for k, v in enumerate(relations)}
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.base_time = base_time
        self.time_unit = time_unit
        self.dataset_name = dataset_name
        self.time_precision = time_precision
        # Anonymization strategies
        self.anon_entity = anon_entity
        self.anon_rel = anon_rel
        self.anon_time = anon_time
        # Construct queries
        self.train_queries = None
        self.valid_queries = None
        self.test_queries = _construct_queries(test_set)
        if train_queries:
            self.train_queries = _construct_queries(train_set)
        if valid_queries:
            self.valid_queries = _construct_queries(valid_set)
        # Search indices
        self._head_history = {}
        self._tail_history = {}
        self._both_history = {}
        self._head_rel_history = {}
        self._tail_rel_history = {}
        if search_indices:
            self.construct_search_indices()
        # Logger
        self.logger = logging.getLogger("TKG")

    def anonymize_entity(
            self,
            entity: str,
    ) -> str:
        """Anonymize entity."""
        if self.anon_entity == "index":
            return str(self.entity2id[entity])
        elif self.anon_entity == "prefix":
            return f"ENT_{self.entity2id[entity]}"
        else:
            return entity

    def deanonymize_entity(
            self,
            entity: str,
    ) -> str:
        """Deanonymize entity."""
        if self.anon_entity == "index":
            return self.entities[int(entity)]
        elif self.anon_entity == "prefix":
            return self.entities[int(entity.replace("ENT_", ""))]
        else:
            return entity

    def anonymize_rel(
            self,
            rel: str
    ) -> str:
        """Anonymize relation."""
        if self.anon_rel == "index":
            return str(self.relation2id[rel])
        elif self.anon_rel == "prefix":
            return f"REL_{self.relation2id[rel]}"
        else:
            return rel

    def anonymize_time(
            self,
            time: str,
    ) -> str:
        """Anonymize time."""
        if self.anon_time == "index":
            time = _str2datetime(time, time_precision=self.time_precision)
            return str((time - self.base_time) // self.time_unit)
        elif self.anon_time == "suffix":
            time = _str2datetime(
                time,
                time_precision=self.time_precision,
            )
            time = card2ord((time - self.base_time) // self.time_unit)
            return f"on the {time} {self.time_precision}"
        else:
            return time

    def construct_search_indices(self) -> None:
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
             train_queries: bool = False,
             valid_queries: bool = False,
             anonymize_entity: str = None,
             anonymize_rel: str = None,
             anonymize_time: str = None,
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
        adjust_time = dataset_name in ["ICEWS14s", "ICEWS05-15"]
        train_set_idx = _read_index_file(train_path, adjust_time=adjust_time)
        valid_set_idx = _read_index_file(valid_path, adjust_time=adjust_time)
        test_set_idx = _read_index_file(test_path, adjust_time=adjust_time)
        logger.info("Facts loaded.")
        id2entity = _read_dict_file(entity2id_path, recover_space=True)
        id2relation = _read_dict_file(relation2id_path, recover_space=True)
        logger.info("Dicts loaded.")
        if dataset_name == "GDELT":
            # GDELT dataset use CAMEO event code as relation name
            # Convert it back to actual relation name
            cameo_config = load_config("config/cameo.yml")
            for rid in id2relation.keys():
                id2relation[rid] = cameo_config["event_code"][id2relation[rid]]
            for eid in id2entity.keys():
                id2entity[eid] = _remove_brackets(id2entity[eid]).capitalize()
        entities = [id2entity[_] for _ in range(len(id2entity))]
        entity2id = {v: k for k, v in id2entity.items()}
        relations = [id2relation[_] for _ in range(len(id2relation))]
        relation2id = {v: k for k, v in id2relation.items()}
        # Convert indices to quadruples
        dataset_config = config["datasets"][dataset_name]
        time_precision = dataset_config["time_precision"]
        base_time = _str2datetime(str(dataset_config["base_time"]))
        time_unit = _get_timedelta(dataset_config["time_unit"])
        # Construct datasets
        train_set = _idx2facts(
            indices=train_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        valid_set = _idx2facts(
            indices=valid_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        test_set = _idx2facts(
            indices=test_set_idx,
            id2entity=id2entity,
            id2relation=id2relation,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
        )
        obj = cls(
            entities=entities,
            relations=relations,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            base_time=base_time,
            time_unit=time_unit,
            time_precision=time_precision,
            dataset_name=dataset_name,
            anon_entity=anonymize_entity,
            anon_rel=anonymize_rel,
            anon_time=anonymize_time,
            train_queries=train_queries,
            valid_queries=valid_queries,
            entity2id=entity2id,
            relation2id=relation2id,
        )
        obj.construct_search_indices()
        logger.info(f"TKG statistics:\n{obj.statistic()}")
        return obj
