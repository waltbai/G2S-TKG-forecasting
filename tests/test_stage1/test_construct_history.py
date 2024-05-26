import unittest
from datetime import datetime

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


class TestConstructHistories(unittest.TestCase):
    def test_construct_histories(self):
        # Case 1: entity, uni
        test_queries = construct_queries(test_facts)
        construct_history(test_queries, tkg)
        query = test_queries[0]
        self.assertEqual(len(query.history), 7)

        # Case 2: entity, bi
        construct_history(test_queries, tkg, history_direction="bi")
        query = test_queries[2]
        self.assertEqual(len(query.history), 6)

        # Case 3: pair, uni
        construct_history(test_queries, tkg, history_type="pair")
        query = test_queries[0]
        self.assertEqual(len(query.history), 4)

        # Case 4: pair, bi
        construct_history(
            test_queries, tkg,
            history_type="pair",
            history_direction="bi"
        )
        query = test_queries[2]
        self.assertEqual(len(query.history), 3)
