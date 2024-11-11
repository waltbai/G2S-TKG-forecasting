import json
import os
from dataclasses import field, dataclass
from typing import List, Dict, Any

from ..common import read_index_file, read_dict_file
from .fact import Fact
from .query import Query


@dataclass
class TKG:
    """
    Temporal Knowledge Graph class.
    """
    # Dataset name
    name: str
    # Facts
    train_facts: List[Fact] = field(default_factory=list)
    valid_facts: List[Fact] = field(default_factory=list)
    test_facts: List[Fact] = field(default_factory=list)
    # Dicts
    entity2id: Dict[str, Any] = field(default_factory=dict)
    relation2id: Dict[str, Any] = field(default_factory=dict)
    # Rules
    rules: Dict[str, Any] = field(default_factory=dict)
    # Indices
    index: Dict[str, Any] = field(default_factory=dict)

    # ===== Methods =====
    @classmethod
    def load(cls, dataset_dir: str, dataset: str):
        """
        Load TKG from directory.
        """
        # Set paths
        dataset_path = os.path.join(dataset_dir, dataset)
        train_path = os.path.join(dataset_path, "train.txt")
        valid_path = os.path.join(dataset_path, "valid.txt")
        test_path = os.path.join(dataset_path, "test.txt")
        entity2id_path = os.path.join(dataset_path, "entity2id.txt")
        relation2id_path = os.path.join(dataset_path, "relation2id.txt")
        rule_path = os.path.join(dataset_dir, "rules", f"{dataset}.json")

        # Load original files
        train_idx = read_index_file(train_path)
        valid_idx = read_index_file(valid_path)
        test_idx = read_index_file(test_path)
        entity2id = read_dict_file(entity2id_path)
        id2entity = {v: k for k, v in entity2id.items()}
        relation2id = read_dict_file(relation2id_path)
        id2relation = {v: k for k, v in relation2id.items()}
        if os.path.exists(rule_path):
            with open(rule_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
        else:
            rules = {}

        # Convert Facts
        map_fact = lambda x: Fact(
            head=id2entity[x[0]],
            rel=id2relation[x[1]],
            tail=id2entity[x[2]],
            time=int(x[3]),
        )
        train_facts = [map_fact(quad) for quad in train_idx]
        valid_facts = [map_fact(quad) for quad in valid_idx]
        test_facts = [map_fact(quad) for quad in test_idx]

        # Construct TKG
        tkg = TKG(
            name=dataset,
            train_facts=train_facts,
            valid_facts=valid_facts,
            test_facts=test_facts,
            entity2id=entity2id,
            relation2id=relation2id,
            rules=rules,
        )
        tkg.build_index()
        return tkg

    def build_index(self):
        """
        Build index for facts.
        """
        # For single one-hop history speedup
        self.index["head"] = {}
        self.index["tail"] = {}
        # For rule-based history speedup
        self.index["head+rel"] = {}
        self.index["tail+rel"] = {}
        for fact in self.train_facts + self.valid_facts + self.test_facts:
            self.index["head"].setdefault(fact.head, []).append(fact)
            self.index["tail"].setdefault(fact.tail, []).append(fact)
            self.index["head+rel"].setdefault((fact.head, fact.rel), []).append(fact)
            self.index["tail+rel"].setdefault((fact.tail, fact.rel), []).append(fact)

    def construct_queries(self, part: str = "test") -> List[Query]:
        """
        Construct queries via facts.
        """
        # Select facts
        assert part in ["train", "valid", "test"]
        if part == "train":
            facts = self.train_facts
        elif part == "valid":
            facts = self.valid_facts
        elif part == "test":
            facts = self.test_facts
        else:
            raise ValueError(f"Unknown part '{part}'.")
        # Generate queries
        queries = []
        answers = {}
        roles = ["head", "tail"]
        for role in roles:
            for fact in facts:
                query = Query.from_fact(fact=fact, role=role)
                key = query.key()
                answers.setdefault(key, []).append(query.answer)
                queries.append(query)
        # Add filter answers
        for query in queries:
            key = query.key()
            query.filters = [_ for _ in answers[key] if _ != query.answer]
        queries = sorted(queries, key=lambda q: q.time)
        return queries

    def find_history(self, query: Query, strategy="rule"):
        """
        Find history for query.
        """
        if strategy == "rule" and len(self.rules):
            query.history = self._find_history_by_rule(query)
        elif strategy == "hop":
            query.history = self._find_history_by_hop(query)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    def _find_history_by_hop(
            self,
            query: Query,
            history_length: int = 50,
    ) -> List[Fact]:
        """
        Find history by rule.
        """
        facts = self.index[query.role][query.entity]
        history = [fact for fact in facts if fact.time < query.time]
        history = history[-history_length:]
        return history

    def _find_history_by_rule(
            self,
            query: Query,
            history_length: int = 50,
    ) -> List[Fact]:
        """
        Find history by hop.
        """
        rule_head = query.rel
        if rule_head in self.rules:
            rule_bodies = sorted(
                self.rules[rule_head].items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            rule_bodies = []
        history = []
        for rule_body, confidence in rule_bodies:
            key1 = f"{query.role}+rel"
            key2 = (query.entity, rule_body)
            if key2 in self.index[key1]:
                temp_history = self.index[key1][key2]
            else:
                temp_history = []
            temp_history = [
                fact for fact in temp_history
                if fact.time < query.time
            ]
            history = temp_history + history
        history = history[-history_length:]
        history = sorted(history, key=lambda x: x.time)
        return history
