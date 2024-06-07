import random
import unittest
from datetime import datetime

from src.utils.anonymizer import get_anonymizer
from src.stage1.prepare import construct_queries, construct_history
from src.utils.fact import Fact
from src.utils.tkg import construct_search_histories, TKG


entities = ["A", "B", "C", "D"]
entity2id = {"A": 0, "B": 1, "C": 2, "D": 3}
relations = ["R1", "R2"]
relation2id = {"R1": 0, "R2": 1}
train_facts = [
    Fact("A", "R1", "B", "2024-01-01"),
    Fact("A", "R2", "B", "2024-01-01"),
    Fact("A", "R1", "C", "2024-01-02"),
    Fact("C", "R2", "B", "2024-01-02"),
    Fact("C", "R1", "D", "2024-01-03"),
]
valid_facts = [
    Fact("A", "R1", "B", "2024-01-04"),
    Fact("A", "R2", "C", "2024-01-04"),
    Fact("A", "R1", "B", "2024-01-05"),
    Fact("A", "R2", "B", "2024-01-05"),
]
test_facts = [
    Fact("A", "R1", "B", "2024-01-06"),
    Fact("A", "R1", "C", "2024-01-06"),
    Fact("A", "R1", "D", "2024-01-07"),
]
base_time = datetime.fromisoformat("2024-01-01")
facts = train_facts + valid_facts + test_facts
search_history = construct_search_histories(facts)
time2id = {
    "2024-01-01": 0,
    "2024-01-02": 1,
    "2024-01-03": 2,
    "2024-01-04": 3,
    "2024-01-05": 4,
    "2024-01-06": 5,
    "2024-01-07": 6,
}
id2time = {v: k for k, v in time2id.items()}
tkg = TKG(
    name="test",
    train_facts=train_facts,
    valid_facts=valid_facts,
    test_facts=test_facts,
    base_time=base_time,
    time_unit="day",
    entities=entities,
    entity2id=entity2id,
    relations=relations,
    relation2id=relation2id,
    search_history=search_history,
    time2id=time2id,
    id2time=id2time,
)


class TestAnonymizer(unittest.TestCase):
    def test_global_anonymize_strategy(self):
        anonymizer = get_anonymizer("global")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1: Without prefix
        anonymizer(queries, tkg, False)
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "0", "B": "1", "C": "2"}
        )
        self.assertDictEqual(
            queries[0].rel_mapping,
            {"R1": "0", "R2": "1"}
        )

        # Case 2: With prefix
        anonymizer(queries, tkg, True)
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "ENT_0", "B": "ENT_1", "C": "ENT_2"}
        )
        self.assertDictEqual(
            queries[0].rel_mapping,
            {"R1": "REL_0", "R2": "REL_1"}
        )

    def test_session_anonymize_strategy(self):
        anonymizer = get_anonymizer("session")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1: without prefix
        anonymizer(queries, tkg, False)
        self.assertDictEqual(
            queries[2].entity_mapping,
            {"B": "0", "A": "1", "C": "2"}
        )
        self.assertDictEqual(
            queries[2].rel_mapping,
            {"R1": "0", "R2": "1"}
        )

        # Case 2: with prefix
        anonymizer(queries, tkg, True)
        self.assertDictEqual(
            queries[2].entity_mapping,
            {"B": "ENT_0", "A": "ENT_1", "C": "ENT_2"}
        )
        self.assertDictEqual(
            queries[2].rel_mapping,
            {"R1": "REL_0", "R2": "REL_1"}
        )

    def test_session_random_anonymize_strategy(self):
        anonymizer = get_anonymizer("random")
        queries = construct_queries(test_facts)
        construct_history(queries=queries, tkg=tkg)

        # Case 1: without prefix
        random.seed(0)
        anonymizer(queries, tkg, False)
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "0", "C": "1", "B": "2"}
        )
        self.assertDictEqual(
            queries[0].rel_mapping,
            {"R2": "0", "R1": "1"}
        )

        # Case 2: with prefix
        random.seed(0)
        anonymizer(queries, tkg, True)
        self.assertDictEqual(
            queries[0].entity_mapping,
            {"A": "ENT_0", "C": "ENT_1", "B": "ENT_2"}
        )
        self.assertDictEqual(
            queries[0].rel_mapping,
            {"R2": "REL_0", "R1": "REL_1"}
        )
